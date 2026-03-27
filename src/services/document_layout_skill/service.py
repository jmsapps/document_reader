import json
import hashlib
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from src.conf.conf import get_config
from src.services.storage_account import AzureStorageAccountService

SEARCH_API_VERSION = "2025-11-01-preview"
EMBEDDING_DIMENSIONS = 1024
AI_VISION_MODEL_VERSION = "2023-04-15"
MAX_MULTIMODAL_TEXT_CHARS = 450
VECTOR_ALGORITHM_NAME = "vector-hnsw"
VECTOR_PROFILE_NAME = "vector-profile"
VECTOR_VECTORIZER_NAME = "vision-vectorizer"
TEXT_EMBEDDING_SKILL_NAME = "#text-vectorize"
IMAGE_EMBEDDING_SKILL_NAME = "#image-vectorize"
TEXT_VECTOR_FIELD_NAME = "chunk_vector"
IMAGE_VECTOR_FIELD_NAME = "image_vector"


class SearchApiError(ValueError):
    """Structured Azure AI Search REST error."""

    def __init__(self, *, method: str, path: str, status_code: int | None, detail: str) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        self.detail = detail
        status_text = f"status={status_code}" if status_code is not None else ""
        super().__init__(f"Search API {method} {path} failed: {status_text} detail={detail}")


class DocumentLayoutSkillService:
    """Orchestrates Azure AI Search DocumentIntelligenceLayoutSkill indexing flow."""

    def __init__(self) -> None:
        config = get_config()
        search_endpoint = config.get("ai_search_endpoint")
        search_api_key = config.get("ai_search_api_key")
        storage_blob_endpoint = config.get("storage_blob_endpoint")
        storage_blob_api_key = config.get("storage_blob_api_key")
        foundry_endpoint = config.get("foundry_endpoint")
        foundry_api_key = config.get("foundry_api_key")
        ai_vision_model_version = config.get("ai_vision_model_version") or AI_VISION_MODEL_VERSION
        ai_vision_embedding_dimensions = config.get("ai_vision_embedding_dimensions") or EMBEDDING_DIMENSIONS

        if not search_endpoint:
            raise ValueError("Missing AZURE_AI_SEARCH_ENDPOINT.")
        if not search_api_key:
            raise ValueError("Missing AZURE_AI_SEARCH_API_KEY (key-based auth required).")
        if not storage_blob_endpoint:
            raise ValueError("Missing AZURE_STORAGE_BLOB_ENDPOINT.")
        if not storage_blob_api_key:
            raise ValueError("Missing AZURE_STORAGE_BLOB_API_KEY (key-based auth required).")
        if not foundry_endpoint:
            raise ValueError("Missing AZURE_FOUNDRY_ENDPOINT for multimodal layout skillset.")
        if not foundry_api_key:
            raise ValueError("Missing AZURE_FOUNDRY_API_KEY for multimodal layout skillset.")

        self.search_endpoint: str = search_endpoint.rstrip("/")
        self.search_api_key: str = search_api_key
        self.storage_blob_endpoint: str = storage_blob_endpoint.rstrip("/")
        self.storage_blob_api_key: str = storage_blob_api_key
        self.foundry_resource_uri: str = self._normalize_foundry_resource_uri(foundry_endpoint)
        self.foundry_api_key: str = foundry_api_key
        self.ai_vision_model_version: str = ai_vision_model_version
        self.embedding_dimensions: int = int(ai_vision_embedding_dimensions)
        self.storage_service = AzureStorageAccountService(
            endpoint=storage_blob_endpoint,
            api_key=storage_blob_api_key,
        )

    @staticmethod
    def _log(message: str) -> None:
        print(f"[document-layout-skill] {message}", flush=True)

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9-]+", "-", value.strip().lower())
        return slug.strip("-") or "document-layout"

    @staticmethod
    def _normalize_foundry_resource_uri(endpoint: str) -> str:
        normalized = endpoint.rstrip("/")
        if "/api/projects/" in normalized:
            normalized = normalized.split("/api/projects/", 1)[0]
        return normalized

    @staticmethod
    def _account_name_from_blob_endpoint(blob_endpoint: str) -> str:
        host = urlparse(blob_endpoint).netloc
        if not host:
            raise ValueError(f"Invalid blob endpoint: {blob_endpoint}")
        return host.split(".", 1)[0]

    @classmethod
    def _build_connection_string(cls, blob_endpoint: str, blob_api_key: str) -> str:
        account_name = cls._account_name_from_blob_endpoint(blob_endpoint)
        return (
            "DefaultEndpointsProtocol=https;"
            f"AccountName={account_name};"
            f"AccountKey={blob_api_key};"
            "EndpointSuffix=core.windows.net"
        )

    def _search_request(self, method: str, path: str, body: Dict[str, Any] | None = None) -> Dict[str, Any]:
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

    def _vector_search_config(self) -> dict[str, Any]:
        return {
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
                    "vectorizer": VECTOR_VECTORIZER_NAME,
                }
            ],
            "vectorizers": [
                {
                    "name": VECTOR_VECTORIZER_NAME,
                    "kind": "aiServicesVision",
                    "aiServicesVisionParameters": {
                        "resourceUri": self.foundry_resource_uri,
                        "apiKey": self.foundry_api_key,
                        "modelVersion": self.ai_vision_model_version,
                    },
                }
            ],
        }

    def _effective_chunk_size(self, chunk_size: int) -> int:
        requested_chunk_size = max(200, int(chunk_size))
        if requested_chunk_size > MAX_MULTIMODAL_TEXT_CHARS:
            self._log(
                "Requested chunk_size exceeds Azure Vision multimodal text limits; "
                f"capping from {requested_chunk_size} to {MAX_MULTIMODAL_TEXT_CHARS} characters"
            )
        return min(requested_chunk_size, MAX_MULTIMODAL_TEXT_CHARS)

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

    def _create_or_update_index(self, *, index_name: str, body: dict[str, Any], hard_refresh: bool) -> None:
        self._log(f"Creating or updating index '{index_name}'")
        try:
            self._search_request("PUT", f"/indexes/{quote(index_name)}", body)
        except SearchApiError as exc:
            detail = exc.detail or ""
            requires_recreate = (
                exc.status_code == 400
                and "cannot have the 'stored' property set to false" in detail.lower()
                and not hard_refresh
            )
            if requires_recreate:
                raise ValueError(
                    "Existing Azure AI Search indexes must be recreated to add vector fields. "
                    "Re-run with --hard-refresh to delete and rebuild the layout-skill indexes."
                ) from exc
            raise

    def _run_indexer_with_backoff(self, *, indexer_name: str, max_wait_seconds: float = 5.0) -> None:
        """Run indexer with bounded exponential backoff for transient 409 conflicts."""
        start = time.time()
        attempt = 0

        while True:
            try:
                self._log(f"Starting indexer '{indexer_name}'")
                self._search_request("POST", f"/indexers/{quote(indexer_name)}/run")
                self._log(f"Indexer '{indexer_name}' accepted run request")
                return
            except SearchApiError as exc:
                detail = (exc.detail or "").lower()
                is_concurrent_conflict = (
                    exc.status_code == 409
                    and "concurrent invocations are not allowed" in detail
                )
                if not is_concurrent_conflict:
                    raise

                elapsed = time.time() - start
                remaining = max_wait_seconds - elapsed
                if remaining <= 0:
                    raise SearchApiError(
                        method="POST",
                        path=f"/indexers/{quote(indexer_name)}/run",
                        status_code=409,
                        detail=(
                            "Indexer invocation remained in concurrent-invocation conflict "
                            f"after {max_wait_seconds:.1f}s of retry."
                        ),
                    ) from exc

                delay = min(0.5 * (2**attempt), remaining)
                self._log(
                    f"Indexer '{indexer_name}' is busy; retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1})"
                )
                time.sleep(delay)
                attempt += 1

    @staticmethod
    def _tiny_markdown_to_html(md_text: str) -> str:
        lines = md_text.splitlines()
        html_lines: list[str] = []
        in_list = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{stripped[2:].strip()}</li>")
                continue

            if in_list:
                html_lines.append("</ul>")
                in_list = False

            if stripped:
                html_lines.append(f"<p>{stripped}</p>")
            else:
                html_lines.append("<br/>")

        if in_list:
            html_lines.append("</ul>")

        return "<!doctype html><html><head><meta charset='utf-8'></head><body>" + "\n".join(html_lines) + "</body></html>"

    @classmethod
    def _load_source_bytes(cls, src: str) -> tuple[bytes, str]:
        parsed = urlparse(src)
        if parsed.scheme in ("http", "https"):
            req = Request(src, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
            filename = Path(parsed.path).name or "document.bin"
            return data, filename

        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() == ".md":
            html = cls._tiny_markdown_to_html(path.read_text(encoding="utf-8"))
            return html.encode("utf-8"), f"{path.stem}.html"

        return path.read_bytes(), path.name

    @staticmethod
    def _source_identity(src: str) -> str:
        parsed = urlparse(src)
        if parsed.scheme in ("http", "https"):
            normalized = src
        else:
            normalized = str(Path(src).resolve())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _extract_blob_name_from_url(blob_url: str) -> str:
        parsed = urlparse(blob_url)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        if len(path_parts) != 2 or not path_parts[1]:
            raise ValueError(f"Invalid blob URL: {blob_url}")
        return path_parts[1]

    @staticmethod
    def _wait_for_indexer(get_status_fn, timeout_seconds: int = 900) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        last: dict[str, Any] = {}
        while time.time() < deadline:
            status = get_status_fn()
            last = status
            last_result = status.get("lastResult") or {}
            state = (last_result.get("status") or "").lower()
            execution_status = status.get("status") or "unknown"
            print(
                "[document-layout-skill] "
                f"Indexer status: execution={execution_status} result={state or 'pending'}",
                flush=True,
            )
            if state in {"success", "transientfailure", "persistentfailure", "reset"}:
                if state != "success":
                    errors = last_result.get("errors") or []
                    raise ValueError(f"Indexer failed with status={state} errors={errors}")
                return status
            time.sleep(5)

        raise TimeoutError(f"Timed out waiting for indexer. Last status: {last}")

    def run(
        self,
        *,
        src: str,
        input_container: str = "layout-input",
        name_prefix: str = "document-layout",
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        hard_refresh: bool = True,
    ) -> Dict[str, Any]:
        self._log(f"Run started for source '{src}'")
        safe_prefix = self._slug(name_prefix)
        text_index_name = f"{safe_prefix}-chunks"
        image_index_name = f"{safe_prefix}-images"
        datasource_name = f"{safe_prefix}-ds"
        skillset_name = f"{safe_prefix}-skillset"
        indexer_name = f"{safe_prefix}-indexer"

        if hard_refresh:
            self._log("Hard refresh enabled; deleting existing search objects and storage container")
            self._search_delete_if_exists(f"/indexers/{quote(indexer_name)}")
            self._search_delete_if_exists(f"/skillsets/{quote(skillset_name)}")
            self._search_delete_if_exists(f"/datasources/{quote(datasource_name)}")
            self._search_delete_if_exists(f"/indexes/{quote(text_index_name)}")
            self._search_delete_if_exists(f"/indexes/{quote(image_index_name)}")
            self.storage_service.delete_container_if_exists(input_container)

        self._log(f"Ensuring storage container '{input_container}' exists")
        self.storage_service.ensure_container(input_container)

        self._log("Loading source document")
        source_bytes, source_filename = self._load_source_bytes(src)
        effective_chunk_size = self._effective_chunk_size(chunk_size)
        source_id = self._source_identity(src)
        blob_prefix = f"source_files/{source_id}"
        blob_name = f"{blob_prefix}/{source_filename}"

        source_blob_url = self.storage_service.get_blob_url(
            container_name=input_container,
            blob_name=blob_name,
        )

        if not self.storage_service.blob_exists(
            container_name=input_container,
            blob_name=blob_name,
        ):
            self._log(f"Uploading source blob '{blob_name}'")
            content_type = mimetypes.guess_type(source_filename)[0] or "application/octet-stream"
            source_blob_url = self.storage_service.upload_bytes(
                container_name=input_container,
                blob_name=blob_name,
                data=source_bytes,
                content_type=content_type,
                overwrite=False,
            )
        else:
            self._log(f"Source blob already exists at '{blob_name}'")

        connection_string = self._build_connection_string(
            self.storage_blob_endpoint,
            self.storage_blob_api_key,
        )

        self._log(f"Creating or updating datasource '{datasource_name}'")
        self._search_request(
            "PUT",
            f"/datasources/{quote(datasource_name)}",
            {
                "name": datasource_name,
                "type": "azureblob",
                "credentials": {"connectionString": connection_string},
                "container": {"name": input_container, "query": blob_prefix},
            },
        )

        self._create_or_update_index(
            index_name=text_index_name,
            hard_refresh=hard_refresh,
            body={
                "name": text_index_name,
                "vectorSearch": self._vector_search_config(),
                "fields": [
                    {"name": "chunk_id", "type": "Edm.String", "key": True, "searchable": True, "analyzer": "keyword", "filterable": True, "retrievable": True},
                    {"name": "parent_id", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True},
                    {"name": "source_path", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True},
                    {"name": "chunk", "type": "Edm.String", "searchable": True, "retrievable": True},
                    {"name": "page_number", "type": "Edm.Int32", "searchable": False, "filterable": True, "sortable": True, "retrievable": True},
                    {"name": "ordinal_position", "type": "Edm.Int32", "searchable": False, "filterable": True, "sortable": True, "retrievable": True},
                    {"name": TEXT_VECTOR_FIELD_NAME, "type": "Collection(Edm.Single)", "searchable": True, "retrievable": True, "dimensions": self.embedding_dimensions, "vectorSearchProfile": VECTOR_PROFILE_NAME},
                ],
            },
        )

        self._create_or_update_index(
            index_name=image_index_name,
            hard_refresh=hard_refresh,
            body={
                "name": image_index_name,
                "fields": [
                    {"name": "image_id", "type": "Edm.String", "key": True, "searchable": True, "analyzer": "keyword", "filterable": True, "retrievable": True},
                    {"name": "parent_id", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True},
                    {"name": "source_path", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True},
                    {"name": "image_path", "type": "Edm.String", "searchable": False, "filterable": True, "retrievable": True},
                    {"name": "page_number", "type": "Edm.Int32", "searchable": False, "filterable": True, "sortable": True, "retrievable": True},
                    {"name": "ordinal_position", "type": "Edm.Int32", "searchable": False, "filterable": True, "sortable": True, "retrievable": True},
                    {"name": IMAGE_VECTOR_FIELD_NAME, "type": "Collection(Edm.Single)", "searchable": True, "retrievable": True, "dimensions": self.embedding_dimensions, "vectorSearchProfile": VECTOR_PROFILE_NAME},
                ],
                "vectorSearch": self._vector_search_config(),
            },
        )

        self._log(f"Creating or updating skillset '{skillset_name}'")
        self._search_request(
            "PUT",
            f"/skillsets/{quote(skillset_name)}",
            {
                "name": skillset_name,
                "description": "Layout skillset for multimodal text and image extraction and vectorization.",
                "skills": [
                    {
                        "@odata.type": "#Microsoft.Skills.Util.DocumentIntelligenceLayoutSkill",
                        "name": "#layout",
                        "description": "Extract text sections and normalized images.",
                        "context": "/document",
                        "outputMode": "oneToMany",
                        "outputFormat": "text",
                        "extractionOptions": ["images", "locationMetadata"],
                        "chunkingProperties": {
                            "unit": "characters",
                            "maximumLength": effective_chunk_size,
                            "overlapLength": max(0, int(chunk_overlap)),
                        },
                        "inputs": [
                            {"name": "file_data", "source": "/document/file_data"},
                        ],
                        "outputs": [
                            {"name": "text_sections", "targetName": "text_sections"},
                            {"name": "normalized_images", "targetName": "normalized_images"},
                        ],
                    },
                    {
                        "@odata.type": "#Microsoft.Skills.Vision.VectorizeSkill",
                        "name": TEXT_EMBEDDING_SKILL_NAME,
                        "description": "Generate multimodal text embeddings for each extracted text section.",
                        "context": "/document/text_sections/*",
                        "modelVersion": self.ai_vision_model_version,
                        "inputs": [
                            {"name": "text", "source": "/document/text_sections/*/content"},
                        ],
                        "outputs": [
                            {"name": "vector", "targetName": TEXT_VECTOR_FIELD_NAME},
                        ],
                    },
                    {
                        "@odata.type": "#Microsoft.Skills.Vision.VectorizeSkill",
                        "name": IMAGE_EMBEDDING_SKILL_NAME,
                        "description": "Generate multimodal image embeddings for each extracted image.",
                        "context": "/document/normalized_images/*",
                        "modelVersion": self.ai_vision_model_version,
                        "inputs": [
                            {"name": "image", "source": "/document/normalized_images/*"},
                        ],
                        "outputs": [
                            {"name": "vector", "targetName": IMAGE_VECTOR_FIELD_NAME},
                        ],
                    }
                ],
                "cognitiveServices": {
                    "@odata.type": "#Microsoft.Azure.Search.AIServicesByKey",
                    "subdomainUrl": self.foundry_resource_uri,
                    "key": self.foundry_api_key,
                },
                "indexProjections": {
                    "selectors": [
                        {
                            "targetIndexName": text_index_name,
                            "parentKeyFieldName": "parent_id",
                            "sourceContext": "/document/text_sections/*",
                            "mappings": [
                                {"name": "chunk", "source": "/document/text_sections/*/content"},
                                {"name": "source_path", "source": "/document/metadata_storage_path"},
                                {"name": "page_number", "source": "/document/text_sections/*/locationMetadata/pageNumber"},
                                {"name": "ordinal_position", "source": "/document/text_sections/*/locationMetadata/ordinalPosition"},
                                {"name": TEXT_VECTOR_FIELD_NAME, "source": f"/document/text_sections/*/{TEXT_VECTOR_FIELD_NAME}"},
                            ],
                        },
                        {
                            "targetIndexName": image_index_name,
                            "parentKeyFieldName": "parent_id",
                            "sourceContext": "/document/normalized_images/*",
                            "mappings": [
                                {"name": "image_path", "source": "/document/normalized_images/*/imagePath"},
                                {"name": "source_path", "source": "/document/metadata_storage_path"},
                                {"name": "page_number", "source": "/document/normalized_images/*/locationMetadata/pageNumber"},
                                {"name": "ordinal_position", "source": "/document/normalized_images/*/locationMetadata/ordinalPosition"},
                                {"name": IMAGE_VECTOR_FIELD_NAME, "source": f"/document/normalized_images/*/{IMAGE_VECTOR_FIELD_NAME}"},
                            ],
                        },
                    ],
                    "parameters": {"projectionMode": "skipIndexingParentDocuments"},
                },
                "knowledgeStore": {
                    "storageConnectionString": connection_string,
                    "projections": [
                        {
                            "files": [
                                {
                                    "storageContainer": input_container,
                                    "source": "/document/normalized_images/*",
                                }
                            ]
                        }
                    ],
                },
            },
        )

        self._log(f"Creating or updating indexer '{indexer_name}'")
        self._search_request(
            "PUT",
            f"/indexers/{quote(indexer_name)}",
            {
                "name": indexer_name,
                "dataSourceName": datasource_name,
                "skillsetName": skillset_name,
                "targetIndexName": text_index_name,
                "parameters": {
                    "configuration": {
                        "allowSkillsetToReadFileData": True,
                    }
                },
            },
        )

        self._run_indexer_with_backoff(indexer_name=indexer_name, max_wait_seconds=5.0)

        self._log(f"Waiting for indexer '{indexer_name}' to finish")
        status = self._wait_for_indexer(
            lambda: self._search_request("GET", f"/indexers/{quote(indexer_name)}/status")
        )
        self._log(f"Indexer '{indexer_name}' completed successfully")

        escaped_source_blob_url = source_blob_url.replace("'", "''")
        source_filter = f"source_path eq '{escaped_source_blob_url}'"
        self._log("Fetching indexed chunks")
        chunks = self._search_request(
            "POST",
            f"/indexes/{quote(text_index_name)}/docs/search",
            {
                "search": "*",
                "filter": source_filter,
                "top": 1000,
                "select": f"chunk_id,parent_id,source_path,chunk,page_number,ordinal_position,{TEXT_VECTOR_FIELD_NAME}",
                "orderby": "ordinal_position asc",
            },
        ).get("value", [])

        self._log("Fetching indexed images")
        images = self._search_request(
            "POST",
            f"/indexes/{quote(image_index_name)}/docs/search",
            {
                "search": "*",
                "filter": source_filter,
                "top": 1000,
                "select": f"image_id,parent_id,source_path,image_path,page_number,ordinal_position,{IMAGE_VECTOR_FIELD_NAME}",
                "orderby": "ordinal_position asc",
            },
        ).get("value", [])
        self._log(f"Run finished with {len(chunks)} chunks and {len(images)} images")

        return {
            "pipeline": "document-layout-skill",
            "source": src,
            "source_blob_url": source_blob_url,
            "source_blob_name": self._extract_blob_name_from_url(source_blob_url),
            "objects": {
                "datasource": datasource_name,
                "text_index": text_index_name,
                "image_index": image_index_name,
                "skillset": skillset_name,
                "indexer": indexer_name,
            },
            "status": status.get("lastResult", {}),
            "embeddings": {
                "mode": "skillset-native-multimodal",
                "text_field": TEXT_VECTOR_FIELD_NAME,
                "image_field": IMAGE_VECTOR_FIELD_NAME,
                "resource_uri": self.foundry_resource_uri,
                "model_version": self.ai_vision_model_version,
                "dimensions": self.embedding_dimensions,
            },
            "chunks": chunks,
            "images": images,
        }
