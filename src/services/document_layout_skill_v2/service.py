import json
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from azure.ai.documentintelligence.models import DocumentContentFormat

from src.conf.conf import get_config
from src.services.document_intelligence.service import DocumentIntelligenceService
from src.services.storage_account import AzureStorageAccountService
from src.storage import LocalOutputStore

SEARCH_API_VERSION = "2024-07-01"
VISION_API_VERSION = "2024-02-01"
VECTOR_ALGORITHM_NAME = "vector-hnsw"
VECTOR_PROFILE_NAME = "vector-profile"
DEFAULT_DEMO_DIR = Path("documents/layout_skill_demo")
DEFAULT_CHUNK_CONTAINER = "chunk-container"
DEFAULT_NAME_PREFIX = "document-layout-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


class SearchApiError(ValueError):
    """Structured Azure AI Search REST error."""

    def __init__(self, *, method: str, path: str, status_code: int | None, detail: str) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        self.detail = detail
        status_text = f"status={status_code}" if status_code is not None else ""
        super().__init__(f"Search API {method} {path} failed: {status_text} detail={detail}")


class DocumentLayoutSkillV2Service:
    """Proof-of-concept layout and chunk ingestion flow targeting one final index."""

    def __init__(self) -> None:
        config = get_config()
        search_endpoint = config.get("ai_search_endpoint")
        search_api_key = config.get("ai_search_api_key")
        storage_blob_endpoint = config.get("storage_blob_endpoint")
        storage_blob_api_key = config.get("storage_blob_api_key")
        embedding_provider = config.get("embedding_provider")
        ai_vision_endpoint = config.get("ai_vision_endpoint")
        ai_vision_api_key = config.get("ai_vision_api_key")
        ai_vision_model_version = config.get("ai_vision_model_version") or "2023-04-15"
        ai_vision_embedding_dimensions = config.get("ai_vision_embedding_dimensions") or 1024
        ai_vision_timeout_seconds = config.get("ai_vision_timeout_seconds") or 300

        if not search_endpoint:
            raise ValueError("Missing AZURE_AI_SEARCH_ENDPOINT.")
        if not search_api_key:
            raise ValueError("Missing AZURE_AI_SEARCH_API_KEY (key-based auth required).")
        if not ai_vision_endpoint:
            raise ValueError("Missing AZURE_AI_VISION_ENDPOINT.")
        if not ai_vision_api_key:
            raise ValueError("Missing AZURE_AI_VISION_API_KEY.")

        self.search_endpoint = search_endpoint.rstrip("/")
        self.search_api_key = search_api_key
        self.embedding_provider = (embedding_provider or "azure_ai_vision").strip().lower()
        self.ai_vision_endpoint = ai_vision_endpoint.rstrip("/")
        self.ai_vision_api_key = ai_vision_api_key
        self.ai_vision_model_version = ai_vision_model_version
        self.embedding_dimensions = int(ai_vision_embedding_dimensions)
        self.ai_vision_timeout_seconds = int(ai_vision_timeout_seconds)
        self.di_service = DocumentIntelligenceService()
        self.storage_service: AzureStorageAccountService | None = None
        if storage_blob_endpoint:
            self.storage_service = AzureStorageAccountService(
                endpoint=storage_blob_endpoint,
                api_key=storage_blob_api_key,
            )
        self.local_output_store = LocalOutputStore()

    @staticmethod
    def _log(message: str) -> None:
        print(f"[document-layout-skill-v2] {message}", flush=True)

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9-]+", "-", value.strip().lower())
        return slug.strip("-") or DEFAULT_NAME_PREFIX

    @staticmethod
    def _searchable_text(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    @classmethod
    def _chunk_text(cls, text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
        normalized = cls._searchable_text(text)
        if not normalized:
            return []

        size = max(100, int(chunk_size))
        overlap = max(0, min(int(chunk_overlap), size // 2))
        start = 0
        chunks: list[str] = []
        while start < len(normalized):
            end = min(len(normalized), start + size)
            if end < len(normalized):
                split = normalized.rfind(" ", start, end)
                if split > start + 40:
                    end = split
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(end - overlap, start + 1)
        return chunks

    @staticmethod
    def _make_record_id(source_name: str, record_kind: str, ordinal: int) -> str:
        safe_source = re.sub(r"[^a-z0-9]+", "-", source_name.lower()).strip("-")
        return f"{safe_source}-{record_kind}-{ordinal:04d}"

    def _search_request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.search_endpoint}{path}?api-version={SEARCH_API_VERSION}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(url, data=data, method=method)
        req.add_header("api-key", self.search_api_key)
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=120) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SearchApiError(
                method=method,
                path=path,
                status_code=exc.code,
                detail=detail,
            ) from exc
        except URLError as exc:
            raise SearchApiError(
                method=method,
                path=path,
                status_code=None,
                detail=str(exc),
            ) from exc

    def _search_delete_if_exists(self, path: str) -> None:
        url = f"{self.search_endpoint}{path}?api-version={SEARCH_API_VERSION}"
        req = Request(url, method="DELETE")
        req.add_header("api-key", self.search_api_key)
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=120):
                return
        except HTTPError as exc:
            if exc.code == 404:
                return
            detail = exc.read().decode("utf-8", errors="replace")
            raise SearchApiError(
                method="DELETE",
                path=path,
                status_code=exc.code,
                detail=detail,
            ) from exc
        except URLError as exc:
            raise SearchApiError(
                method="DELETE",
                path=path,
                status_code=None,
                detail=str(exc),
            ) from exc

    def _target_index_name(self, name_prefix: str) -> str:
        return self._slug(name_prefix)

    def _vision_vectorize(self, *, route: str, payload: dict[str, Any]) -> list[float]:
        if self.embedding_provider != "azure_ai_vision":
            raise ValueError(
                f"Unsupported AZURE_EMBEDDING_PROVIDER '{self.embedding_provider}'. "
                "Currently supported: azure_ai_vision."
            )

        url = (
            f"{self.ai_vision_endpoint}/computervision/{route}"
            f"?api-version={VISION_API_VERSION}&model-version={quote(self.ai_vision_model_version)}"
        )
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, method="POST")
        req.add_header("Ocp-Apim-Subscription-Key", self.ai_vision_api_key)
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=self.ai_vision_timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(
                f"Azure AI Vision {route} failed: status={exc.code} detail={detail}"
            ) from exc
        except URLError as exc:
            raise ValueError(f"Azure AI Vision {route} failed: {exc}") from exc
        except TimeoutError as exc:
            raise ValueError(
                f"Azure AI Vision {route} timed out after {self.ai_vision_timeout_seconds}s."
            ) from exc

        vector = body.get("vector")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Azure AI Vision {route} returned no vector.")
        return [float(value) for value in vector]

    def _vision_vectorize_image_stream(self, image_bytes: bytes, content_type: str) -> list[float]:
        if self.embedding_provider != "azure_ai_vision":
            raise ValueError(
                f"Unsupported AZURE_EMBEDDING_PROVIDER '{self.embedding_provider}'. "
                "Currently supported: azure_ai_vision."
            )

        url = (
            f"{self.ai_vision_endpoint}/computervision/retrieval:vectorizeImage"
            f"?overload=stream&api-version={VISION_API_VERSION}"
            f"&model-version={quote(self.ai_vision_model_version)}"
        )
        req = Request(url, data=image_bytes, method="POST")
        req.add_header("Ocp-Apim-Subscription-Key", self.ai_vision_api_key)
        req.add_header("Content-Type", content_type)

        try:
            with urlopen(req, timeout=self.ai_vision_timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(
                f"Azure AI Vision retrieval:vectorizeImage(stream) failed: "
                f"status={exc.code} detail={detail}"
            ) from exc
        except URLError as exc:
            raise ValueError(
                f"Azure AI Vision retrieval:vectorizeImage(stream) failed: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise ValueError(
                "Azure AI Vision retrieval:vectorizeImage(stream) "
                f"timed out after {self.ai_vision_timeout_seconds}s."
            ) from exc

        vector = body.get("vector")
        if not isinstance(vector, list) or not vector:
            raise ValueError("Azure AI Vision retrieval:vectorizeImage(stream) returned no vector.")
        return [float(value) for value in vector]

    def _vision_describe_image(self, image_bytes: bytes) -> dict[str, Any]:
        url = f"{self.ai_vision_endpoint}/vision/v3.2/analyze?visualFeatures=Description,Tags,Objects"
        req = Request(url, data=image_bytes, method="POST")
        req.add_header("Ocp-Apim-Subscription-Key", self.ai_vision_api_key)
        req.add_header("Content-Type", "application/octet-stream")

        try:
            with urlopen(req, timeout=self.ai_vision_timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._log(
                "Image Analysis captioning failed; continuing without model-generated image summary. "
                f"status={exc.code} detail={detail}"
            )
            return {}
        except URLError as exc:
            self._log(
                "Image Analysis captioning failed; continuing without model-generated image summary. "
                f"detail={exc}"
            )
            return {}
        except TimeoutError:
            self._log(
                "Image Analysis captioning timed out; continuing without model-generated image summary."
            )
            return {}

    def _embed_text(self, text: str) -> list[float]:
        normalized = self._searchable_text(text)
        if not normalized:
            raise ValueError("Cannot embed empty text.")
        return self._vision_vectorize(
            route="retrieval:vectorizeText",
            payload={"text": normalized},
        )

    def _embed_image_bytes(self, image_bytes: bytes, content_type: str = "image/png") -> list[float]:
        if not image_bytes:
            raise ValueError("Cannot embed empty image bytes.")
        return self._vision_vectorize_image_stream(image_bytes=image_bytes, content_type=content_type)

    def _ensure_target_index(self, *, index_name: str, hard_refresh: bool) -> None:
        if hard_refresh:
            self._log(f"Hard refresh enabled; deleting index '{index_name}'")
            self._search_delete_if_exists(f"/indexes/{quote(index_name)}")

        self._log(f"Creating or updating index '{index_name}'")
        self._search_request(
            "PUT",
            f"/indexes/{quote(index_name)}",
            {
                "name": index_name,
                "fields": [
                    {
                        "name": "id",
                        "type": "Edm.String",
                        "key": True,
                        "filterable": True,
                        "retrievable": True,
                    },
                    {
                        "name": "metadata",
                        "type": "Edm.ComplexType",
                        "fields": [
                            {
                                "name": "source_type",
                                "type": "Edm.String",
                                "searchable": True,
                                "filterable": True,
                                "retrievable": True,
                                "facetable": True,
                            },
                            {
                                "name": "category",
                                "type": "Edm.String",
                                "searchable": True,
                                "filterable": True,
                                "retrievable": True,
                                "facetable": True,
                            },
                            {
                                "name": "topic",
                                "type": "Edm.String",
                                "searchable": True,
                                "filterable": True,
                                "retrievable": True,
                                "facetable": True,
                            },
                            {
                                "name": "subtopic",
                                "type": "Edm.String",
                                "searchable": True,
                                "filterable": True,
                                "retrievable": True,
                                "facetable": True,
                            },
                            {
                                "name": "source_url",
                                "type": "Edm.String",
                                "searchable": False,
                                "filterable": True,
                                "retrievable": True,
                            },
                            {
                                "name": "source_url_text",
                                "type": "Edm.String",
                                "searchable": True,
                                "filterable": True,
                                "retrievable": True,
                            },
                        ],
                    },
                    {
                        "name": "content",
                        "type": "Edm.String",
                        "searchable": True,
                        "retrievable": True,
                    },
                    {
                        "name": "contentVector",
                        "type": "Collection(Edm.Single)",
                        "searchable": True,
                        "retrievable": True,
                        "dimensions": self.embedding_dimensions,
                        "vectorSearchProfile": VECTOR_PROFILE_NAME,
                    },
                ],
                "vectorSearch": {
                    "algorithms": [
                        {
                            "name": VECTOR_ALGORITHM_NAME,
                            "kind": "hnsw",
                            "hnswParameters": {
                                "metric": "cosine",
                            },
                        }
                    ],
                    "profiles": [
                        {
                            "name": VECTOR_PROFILE_NAME,
                            "algorithm": VECTOR_ALGORITHM_NAME,
                        }
                    ],
                },
            },
        )

    def _upload_records(self, *, index_name: str, records: list[dict[str, Any]]) -> None:
        for start in range(0, len(records), 500):
            batch = records[start:start + 500]
            body = {
                "value": [
                    {
                        "@search.action": "mergeOrUpload",
                        **record,
                    }
                    for record in batch
                ]
            }
            self._search_request("POST", f"/indexes/{quote(index_name)}/docs/index", body)

    def _save_artifact(self, *, container_name: str, blob_name: str, payload: Any) -> str:
        if self.storage_service:
            return self.storage_service.upload_json(
                container_name=container_name,
                blob_name=blob_name,
                payload=payload,
            )

        local_path = f"local_documents/{container_name}/{blob_name}"
        self.local_output_store.save(payload, local_path)
        return local_path

    def _load_demo_files(self, demo_dir: Path) -> list[Path]:
        if not demo_dir.exists():
            raise FileNotFoundError(f"Demo folder not found: {demo_dir}")
        files = [path for path in demo_dir.iterdir() if path.is_file()]
        if not files:
            raise ValueError(f"No demo files found in {demo_dir}")
        return sorted(files)

    @staticmethod
    def _metadata_from_payload(payload: dict[str, Any], source_name: str) -> dict[str, str]:
        metadata = payload.get("metadata") or {}
        return {
            "source_type": str(metadata.get("source_type") or "text"),
            "category": str(metadata.get("category") or ""),
            "topic": str(metadata.get("topic") or ""),
            "subtopic": str(metadata.get("subtopic") or ""),
            "source_url": str(metadata.get("source_url") or source_name),
            "source_url_text": str(metadata.get("source_url_text") or source_name),
        }

    def _json_records(
        self,
        *,
        path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_name = path.stem
        metadata = self._metadata_from_payload(payload, source_name)
        chunks = self._chunk_text(
            str(payload.get("content") or payload.get("fulltext") or ""),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        records: list[dict[str, Any]] = []
        for ordinal, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "id": self._make_record_id(source_name, "text", ordinal),
                    "metadata": metadata,
                    "content": chunk,
                    "contentVector": self._embed_text(chunk),
                }
            )
        return records

    def _write_binary_artifact(
        self,
        *,
        container_name: str,
        blob_name: str,
        data: bytes,
        content_type: str,
    ) -> str:
        if not self.storage_service:
            raise ValueError(
                "Blob storage is required for PDF/image processing in layout-skill-v2 so "
                "source URLs remain navigable and Azure AI Vision can vectorize extracted images."
            )
        return self.storage_service.upload_bytes(
            container_name=container_name,
            blob_name=blob_name,
            data=data,
            content_type=content_type,
        )

    def _pdf_source_url(self, *, chunk_container: str, path: Path) -> str:
        with path.open("rb") as f:
            return self._write_binary_artifact(
                container_name=chunk_container,
                blob_name=f"sources/{path.name}",
                data=f.read(),
                content_type="application/pdf",
            )

    def _extract_figure_bytes(self, *, result_id: str, figure_id: str) -> bytes:
        stream = self.di_service.client.get_analyze_result_figure(
            model_id="prebuilt-layout",
            result_id=result_id,
            figure_id=figure_id,
        )
        return b"".join(stream)

    def _extract_figure_text(self, figure_bytes: bytes) -> str:
        result = self.di_service.analyze_bytes(
            data=figure_bytes,
            model_id="prebuilt-read",
            content_format=DocumentContentFormat.TEXT,
        )
        content = self._searchable_text(getattr(result, "content", "") or "")
        return content

    @classmethod
    def _summarize_figure(
        cls,
        *,
        model_description: str,
        model_tags: list[str],
        caption: str,
        page_context: str,
        figure_ocr_text: str,
    ) -> str:
        combined = cls._searchable_text(
            " ".join([model_description, " ".join(model_tags), caption, page_context, figure_ocr_text])
        )
        lower = combined.lower()

        visual_type = "visual"
        if any(term in lower for term in ["graph", "line chart", "line graph"]):
            visual_type = "graph"
        elif any(term in lower for term in ["bar chart", "bar graph", "bars"]):
            visual_type = "bar chart"
        elif any(term in lower for term in ["chart", "plot"]):
            visual_type = "chart"
        elif any(term in lower for term in ["table"]):
            visual_type = "table"

        topics: list[str] = []
        topic_terms = [
            "gdp",
            "growth",
            "inflation",
            "output",
            "trade",
            "loss",
            "employment",
            "economy",
            "forecast",
            "real gdp",
        ]
        for term in topic_terms:
            if term in lower and term not in topics:
                topics.append(term)

        topic_text = ", ".join(topics[:4]) if topics else "document content"

        if model_description:
            return cls._searchable_text(
                f"{model_description}. "
                f"Relevant concepts: {topic_text}. "
                f"Tags: {', '.join(model_tags[:8])}."
            )

        if caption:
            return cls._searchable_text(
                f"{visual_type} about {topic_text}. "
                f"Caption summary: {caption}."
            )

        return cls._searchable_text(
            f"{visual_type} about {topic_text}."
        )

    @staticmethod
    def _page_number_from_regions(item: Any) -> int | None:
        regions = getattr(item, "bounding_regions", None) or []
        for region in regions:
            page_number = getattr(region, "page_number", None)
            if page_number is not None:
                return int(page_number)
        return None

    def _pdf_records(
        self,
        *,
        path: Path,
        chunk_container: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict[str, Any]]:
        result, operation_id = self.di_service.analyze_file_with_figures(
            path=path,
            model_id="prebuilt-layout",
            content_format=DocumentContentFormat.TEXT,
        )
        source_name = path.stem
        source_url = self._pdf_source_url(chunk_container=chunk_container, path=path)
        metadata = {
            "source_type": "text",
            "category": "document-layout-demo",
            "topic": "layout extraction",
            "subtopic": "pdf text and figures",
            "source_url": source_url,
            "source_url_text": path.name,
        }

        page_text: dict[int, list[str]] = {}
        for paragraph in getattr(result, "paragraphs", None) or []:
            content = self._searchable_text(getattr(paragraph, "content", "") or "")
            page_number = self._page_number_from_regions(paragraph)
            if not content or page_number is None:
                continue
            page_text.setdefault(page_number, []).append(content)

        records: list[dict[str, Any]] = []
        ordinal = 1
        for page_number in sorted(page_text):
            page_content = " ".join(page_text[page_number])
            for chunk in self._chunk_text(
                page_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ):
                records.append(
                    {
                        "id": self._make_record_id(source_name, "text", ordinal),
                        "metadata": dict(metadata),
                        "content": f"Page {page_number}: {chunk}",
                        "contentVector": self._embed_text(chunk),
                    }
                )
                ordinal += 1

        figures = getattr(result, "figures", None) or []
        for figure in figures:
            if not operation_id:
                raise ValueError("Document Intelligence analyze result did not return operation_id for figures.")
            figure_id = getattr(figure, "id", None) or f"figure-{ordinal}"
            caption = getattr(getattr(figure, "caption", None), "content", None) or ""
            page_number = self._page_number_from_regions(figure) or 0
            page_context = " ".join(page_text.get(page_number, [])[:2])
            figure_bytes = self._extract_figure_bytes(result_id=operation_id, figure_id=figure_id)
            figure_ocr_text = self._extract_figure_text(figure_bytes)
            analysis = self._vision_describe_image(figure_bytes)
            description_block = analysis.get("description") or {}
            captions = description_block.get("captions") or []
            model_description = ""
            if captions:
                first_caption = captions[0] or {}
                model_description = self._searchable_text(str(first_caption.get("text") or ""))
            model_tags = [
                self._searchable_text(str(tag.get("name") or ""))
                for tag in (analysis.get("tags") or [])
                if self._searchable_text(str(tag.get("name") or ""))
            ]
            figure_summary = self._summarize_figure(
                model_description=model_description,
                model_tags=model_tags,
                caption=caption,
                page_context=page_context,
                figure_ocr_text=figure_ocr_text,
            )
            figure_url = self._write_binary_artifact(
                container_name=chunk_container,
                blob_name=f"figures/{source_name}/{figure_id}.png",
                data=figure_bytes,
                content_type="image/png",
            )
            figure_text = self._searchable_text(
                f"Image summary: {figure_summary}. "
                f"Figure extracted from {path.name}. "
                f"Figure id: {figure_id}. "
                f"Page: {page_number}. "
                f"Model description: {model_description}. "
                f"Model tags: {', '.join(model_tags[:8])}. "
                f"Caption: {caption}. "
                f"Context: {page_context}. "
                f"Image OCR: {figure_ocr_text}"
            )
            if not figure_text:
                continue
            records.append(
                {
                    "id": self._make_record_id(source_name, "image", ordinal),
                    "metadata": {
                        **metadata,
                        "source_type": "image",
                        "source_url": figure_url,
                        "source_url_text": f"{path.name} figure {figure_id}",
                    },
                    "content": figure_text,
                    "contentVector": self._embed_image_bytes(figure_bytes, content_type="image/png"),
                }
            )
            ordinal += 1

        if not records:
            raise ValueError(f"No extractable content found in PDF: {path}")
        return records

    def _records_for_source(
        self,
        *,
        path: Path,
        chunk_container: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._json_records(path=path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if suffix == ".pdf":
            return self._pdf_records(
                path=path,
                chunk_container=chunk_container,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        raise ValueError(f"Unsupported demo file type: {path.suffix}")

    def _write_source_artifact(
        self,
        *,
        container_name: str,
        source_path: Path,
        records: list[dict[str, Any]],
    ) -> str:
        artifact = {
            "source": source_path.name,
            "recordCount": len(records),
            "records": records,
        }
        blob_name = f"{source_path.stem}.json"
        return self._save_artifact(container_name=container_name, blob_name=blob_name, payload=artifact)

    def run(
        self,
        *,
        src: str,
        chunk_container: str = DEFAULT_CHUNK_CONTAINER,
        name_prefix: str = DEFAULT_NAME_PREFIX,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        hard_refresh: bool = False,
    ) -> dict[str, Any]:
        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        records = self._records_for_source(
            path=path,
            chunk_container=chunk_container,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        artifact_uri = self._write_source_artifact(
            container_name=chunk_container,
            source_path=path,
            records=records,
        )
        index_name = self._target_index_name(name_prefix)
        self._ensure_target_index(index_name=index_name, hard_refresh=hard_refresh)
        self._upload_records(index_name=index_name, records=records)

        return {
            "pipeline": "document-layout-skill-v2",
            "mode": "single-source",
            "source": str(path),
            "chunk_container": chunk_container,
            "derived_artifact": artifact_uri,
            "target_index": index_name,
            "record_count": len(records),
            "embedding": {
                "mode": self.embedding_provider,
                "field": "contentVector",
                "dimensions": self.embedding_dimensions,
            },
            "records": records,
        }

    def run_demo(
        self,
        *,
        demo_dir: str | Path = DEFAULT_DEMO_DIR,
        chunk_container: str = DEFAULT_CHUNK_CONTAINER,
        name_prefix: str = DEFAULT_NAME_PREFIX,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        hard_refresh: bool = True,
    ) -> dict[str, Any]:
        demo_path = Path(demo_dir)
        files = self._load_demo_files(demo_path)
        derived_artifacts: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []

        self._log(f"Processing {len(files)} demo file(s) from '{demo_path}'")
        for path in files:
            self._log(f"Deriving records for '{path.name}'")
            records = self._records_for_source(
                path=path,
                chunk_container=chunk_container,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            artifact_uri = self._write_source_artifact(
                container_name=chunk_container,
                source_path=path,
                records=records,
            )
            derived_artifacts.append(
                {
                    "source": path.name,
                    "artifact": artifact_uri,
                    "record_count": len(records),
                }
            )
            all_records.extend(records)

        index_name = self._target_index_name(name_prefix)
        self._ensure_target_index(index_name=index_name, hard_refresh=hard_refresh)
        self._upload_records(index_name=index_name, records=all_records)
        self._log(f"Demo finished with {len(all_records)} indexed record(s)")

        return {
            "pipeline": "document-layout-skill-v2",
            "mode": "demo",
            "demo_dir": str(demo_path),
            "chunk_container": chunk_container,
            "target_index": index_name,
            "source_count": len(files),
            "record_count": len(all_records),
            "derived_artifacts": derived_artifacts,
            "embedding": {
                "mode": self.embedding_provider,
                "field": "contentVector",
                "dimensions": self.embedding_dimensions,
            },
            "records": all_records,
        }
