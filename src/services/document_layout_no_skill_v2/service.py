import json
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from azure.ai.documentintelligence.models import DocumentContentFormat

from src.conf.conf import get_config
from src.services.ai_search.service import AISearchService
from src.services.document_intelligence.service import DocumentIntelligenceService
from src.services.openai import OpenAIService, OpenAIServiceError
from src.services.shared import (
    DEFAULT_CHUNK_CONTAINER,
    DEFAULT_TARGET_INDEX_NAME,
    build_shared_index,
)
from src.services.storage_account import AzureStorageAccountService
from src.storage import LocalOutputStore

SEARCH_API_VERSION = "2024-07-01"
DEFAULT_DEMO_DIR = Path("documents/demo_files")
DEFAULT_NAME_PREFIX = DEFAULT_TARGET_INDEX_NAME
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_TEXT_RECORD_CATEGORY = "document-layout-demo"
DEFAULT_TEXT_RECORD_TOPIC = "layout extraction"
DEFAULT_TEXT_RECORD_SUBTOPIC = "pdf text and figures"
VERTICAL_PROXIMITY_THRESHOLD = 0.35
CAPTION_BAND_THRESHOLD = 0.12
MIN_HORIZONTAL_OVERLAP = 0.2
MAX_RELEVANT_PARAGRAPHS = 2
MAX_SURROUNDING_PARAGRAPHS = 2
FIGURE_INTERPRETATION_SCHEMA: dict[str, Any] = {
    "name": "figure_interpretation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "figure_type": {"type": "string"},
            "what_it_shows": {"type": "string"},
            "key_relationships": {
                "type": "array",
                "items": {"type": "string"},
            },
            "supporting_context": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "image_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "ocr_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "relevant_text_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "surrounding_text_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "document_summary_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "image_evidence",
                    "ocr_evidence",
                    "relevant_text_evidence",
                    "surrounding_text_evidence",
                    "document_summary_evidence",
                ],
            },
            "uncertainties": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence_notes": {"type": "string"},
        },
        "required": [
            "figure_type",
            "what_it_shows",
            "key_relationships",
            "supporting_context",
            "uncertainties",
            "confidence_notes",
        ],
    },
}


class SearchApiError(ValueError):
    """Structured Azure AI Search REST error."""

    def __init__(
        self, *, method: str, path: str, status_code: int | None, detail: str
    ) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        self.detail = detail
        status_text = f"status={status_code}" if status_code is not None else ""
        super().__init__(
            f"Search API {method} {path} failed: {status_text} detail={detail}"
        )


class OpenAIApiError(ValueError):
    """Structured Azure OpenAI REST error."""

    def __init__(self, *, path: str, status_code: int | None, detail: str) -> None:
        self.path = path
        self.status_code = status_code
        self.detail = detail
        status_text = f"status={status_code}" if status_code is not None else ""
        super().__init__(f"Azure OpenAI {path} failed: {status_text} detail={detail}")

    @classmethod
    def from_service_error(cls, exc: OpenAIServiceError) -> "OpenAIApiError":
        return cls(path=exc.path, status_code=exc.status_code, detail=exc.detail)


class DocumentLayoutNoSkillV2Service:
    """Sibling no-skill path that uses Azure OpenAI grounded figure verbalization."""

    def __init__(self) -> None:
        config = get_config()
        search_endpoint = config.get("ai_search_endpoint")
        search_api_key = config.get("ai_search_api_key")
        storage_blob_endpoint = config.get("storage_blob_endpoint")
        storage_blob_api_key = config.get("storage_blob_api_key")
        openai_endpoint = config.get("openai_endpoint")
        openai_api_key = config.get("openai_api_key")
        openai_api_version = config.get("openai_api_version")
        openai_chat_deployment = config.get("openai_chat_deployment")
        openai_interpret_deployment = config.get("openai_interpret_deployment")
        openai_verbalization_deployment = config.get("openai_verbalization_deployment")
        openai_embedding_deployment = config.get("openai_embedding_deployment")
        openai_embedding_dimensions = config.get("openai_embedding_dimensions")

        if not search_endpoint:
            raise ValueError("Missing AZURE_AI_SEARCH_ENDPOINT.")
        if not search_api_key:
            raise ValueError(
                "Missing AZURE_AI_SEARCH_API_KEY (key-based auth required)."
            )
        if not openai_endpoint:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT.")
        if not openai_api_key:
            raise ValueError("Missing AZURE_OPENAI_API_KEY.")
        if openai_api_version not in (None, "", "v1"):
            raise ValueError(
                "layout-no-skill-v2 requires Azure OpenAI v1 semantics. "
                "Set AZURE_OPENAI_API_VERSION=v1."
            )
        if not openai_interpret_deployment:
            raise ValueError("Missing AZURE_OPENAI_INTERPRET_DEPLOYMENT.")
        if not openai_verbalization_deployment:
            raise ValueError("Missing AZURE_OPENAI_VERBALIZATION_DEPLOYMENT.")
        if not openai_embedding_deployment:
            raise ValueError("Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT.")

        self.search_endpoint = search_endpoint.rstrip("/")
        self.search_api_key = search_api_key
        self.chat_deployment = openai_chat_deployment or openai_verbalization_deployment
        self.interpret_deployment = openai_interpret_deployment
        self.verbalization_deployment = openai_verbalization_deployment
        self.embedding_deployment = openai_embedding_deployment
        self.embedding_dimensions = int(
            openai_embedding_dimensions or DEFAULT_EMBEDDING_DIMENSIONS
        )
        self.di_service = DocumentIntelligenceService()
        self.ai_search_service = AISearchService()
        self.storage_service: AzureStorageAccountService | None = None
        if storage_blob_endpoint:
            self.storage_service = AzureStorageAccountService(
                endpoint=storage_blob_endpoint,
                api_key=storage_blob_api_key,
            )
        self.local_output_store = LocalOutputStore()
        self.openai_service = OpenAIService()

    @staticmethod
    def _log(message: str) -> None:
        print(f"[document-layout-no-skill-v2] {message}", flush=True)

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9-]+", "-", value.strip().lower())
        return slug.strip("-") or DEFAULT_NAME_PREFIX

    @staticmethod
    def _searchable_text(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    @staticmethod
    def _optional_text(value: Any) -> str:
        return str(value or "").strip()

    @classmethod
    def _metadata_record(
        cls,
        *,
        source_type: str,
        category: str = "",
        topic: str = "",
        subtopic: str = "",
        source_url: str = "",
        source_url_text: str = "",
        source_name: str = "",
        image: dict[str, Any] | None = None,
        page_number: int | None = None,
        figure_id: str = "",
        summary_method: str = "",
        bounding_regions: list[dict[str, Any]] | None = None,
        ocr_text: str = "",
        caption: str = "",
    ) -> dict[str, Any]:
        image_metadata = (
            image
            if image is not None
            else cls._image_metadata_record(
                page_number=page_number,
                figure_id=figure_id,
                summary_method=summary_method,
                bounding_regions=bounding_regions,
                ocr_text=ocr_text,
                caption=caption,
            )
        )
        return {
            "source_type": cls._optional_text(source_type) or "text",
            "category": cls._optional_text(category),
            "topic": cls._optional_text(topic),
            "subtopic": cls._optional_text(subtopic),
            "source_url": cls._optional_text(source_url),
            "source_url_text": cls._optional_text(source_url_text),
            "source_name": cls._optional_text(source_name),
            "image": image_metadata or None,
        }

    @classmethod
    def _image_metadata_record(
        cls,
        *,
        page_number: int | None = None,
        figure_id: str = "",
        summary_method: str = "",
        bounding_regions: list[dict[str, Any]] | None = None,
        ocr_text: str = "",
        caption: str = "",
    ) -> dict[str, Any]:
        record: dict[str, Any] = {}
        if page_number is not None:
            record["page_number"] = int(page_number)
        if figure_id:
            record["figure_id"] = cls._optional_text(figure_id)
        if summary_method:
            record["summary_method"] = cls._optional_text(summary_method)
        if bounding_regions:
            record["bounding_regions"] = bounding_regions
        if ocr_text:
            record["ocr_text"] = cls._optional_text(ocr_text)
        if caption:
            record["caption"] = cls._optional_text(caption)
        return record

    @classmethod
    def _bounding_regions_record(cls, regions: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for region in regions or []:
            polygon_values: list[float] = []
            polygon = getattr(region, "polygon", None) or []
            if polygon and all(isinstance(value, (int, float)) for value in polygon):
                polygon_values = [float(value) for value in polygon]
            for point in polygon if not polygon_values else []:
                x = getattr(point, "x", None)
                y = getattr(point, "y", None)
                if x is None and isinstance(point, (list, tuple)) and len(point) >= 2:
                    x, y = point[0], point[1]
                if x is None or y is None:
                    continue
                polygon_values.extend([float(x), float(y)])
            record: dict[str, Any] = {}
            page_number = getattr(region, "page_number", None)
            if page_number is not None and polygon_values:
                record["page_number"] = int(page_number)
            if polygon_values:
                record["polygon"] = polygon_values
            if record:
                records.append(record)
        return records

    @classmethod
    def _chunk_text(
        cls, text: str, *, chunk_size: int, chunk_overlap: int
    ) -> list[str]:
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

    @staticmethod
    def _page_number_from_regions(item: Any) -> int | None:
        regions = getattr(item, "bounding_regions", None) or []
        for region in regions:
            page_number = getattr(region, "page_number", None)
            if page_number is not None:
                return int(page_number)
        return None

    @staticmethod
    def _page_dimensions_map(result: Any) -> dict[int, dict[str, float]]:
        page_dimensions: dict[int, dict[str, float]] = {}
        for page in getattr(result, "pages", None) or []:
            page_number = getattr(page, "page_number", None)
            width = getattr(page, "width", None)
            height = getattr(page, "height", None)
            if page_number is None or width in (None, 0) or height in (None, 0):
                continue
            page_dimensions[int(page_number)] = {
                "width": float(width),
                "height": float(height),
            }
        return page_dimensions

    @classmethod
    def _bbox_from_bounding_regions(
        cls,
        *,
        bounding_regions: list[dict[str, Any]],
        page_dimensions: dict[int, dict[str, float]],
    ) -> dict[str, float] | None:
        xs: list[float] = []
        ys: list[float] = []
        normalized_xs: list[float] = []
        normalized_ys: list[float] = []
        region_page_number: int | None = None

        for region in bounding_regions:
            page_number = region.get("page_number")
            polygon = region.get("polygon") or []
            if (
                not isinstance(page_number, int)
                or not isinstance(polygon, list)
                or len(polygon) < 4
            ):
                continue
            dims = page_dimensions.get(page_number)
            if not dims:
                continue
            region_page_number = page_number
            for idx in range(0, len(polygon), 2):
                if idx + 1 >= len(polygon):
                    break
                x = polygon[idx]
                y = polygon[idx + 1]
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    continue
                x_value = float(x)
                y_value = float(y)
                xs.append(x_value)
                ys.append(y_value)
                normalized_xs.append(x_value / dims["width"])
                normalized_ys.append(y_value / dims["height"])

        if not normalized_xs or not normalized_ys or region_page_number is None:
            return None

        left = min(normalized_xs)
        right = max(normalized_xs)
        top = min(normalized_ys)
        bottom = max(normalized_ys)
        return {
            "page_number": float(region_page_number),
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "center_x": (left + right) / 2,
            "center_y": (top + bottom) / 2,
        }

    @classmethod
    def _build_paragraph_record(
        cls,
        *,
        paragraph: Any,
        index: int,
        page_dimensions: dict[int, dict[str, float]],
    ) -> dict[str, Any] | None:
        text = cls._searchable_text(getattr(paragraph, "content", "") or "")
        if not text:
            return None
        bounding_regions = cls._bounding_regions_record(
            getattr(paragraph, "bounding_regions", None)
        )
        bbox = cls._bbox_from_bounding_regions(
            bounding_regions=bounding_regions,
            page_dimensions=page_dimensions,
        )
        if not bbox:
            return None
        page_number = int(bbox["page_number"])
        return {
            "text": text,
            "page_number": page_number,
            "index": index,
            "bounding_regions": bounding_regions,
            "bbox": bbox,
            "top": bbox["top"],
            "bottom": bbox["bottom"],
            "left": bbox["left"],
            "right": bbox["right"],
            "center_x": bbox["center_x"],
            "center_y": bbox["center_y"],
        }

    @staticmethod
    def _normalize_figure_id(figure_id: str) -> str:
        return re.sub(r"[^0-9]+", ".", (figure_id or "").lower()).strip(".")

    @classmethod
    def _figure_reference_patterns(cls, figure_id: str) -> list[re.Pattern[str]]:
        normalized = cls._normalize_figure_id(figure_id)
        if not normalized:
            return []
        dashed = normalized.replace(".", "-")
        escaped_normalized = re.escape(normalized)
        escaped_dashed = re.escape(dashed)
        return [
            re.compile(rf"\bfigure\s+{escaped_normalized}\b", re.IGNORECASE),
            re.compile(rf"\bfig\.?\s*{escaped_normalized}\b", re.IGNORECASE),
            re.compile(rf"\bfigure\s+{escaped_dashed}\b", re.IGNORECASE),
            re.compile(rf"\bfig\.?\s*{escaped_dashed}\b", re.IGNORECASE),
            re.compile(rf"\({escaped_normalized}\)", re.IGNORECASE),
        ]

    @classmethod
    def _extract_figure_reference_ids(cls, text: str) -> set[str]:
        references: set[str] = set()
        for pattern in [
            re.compile(r"\bfigure\s+([0-9]+(?:[.-][0-9]+)*)\b", re.IGNORECASE),
            re.compile(r"\bfig\.?\s*([0-9]+(?:[.-][0-9]+)*)\b", re.IGNORECASE),
            re.compile(r"\(([0-9]+(?:\.[0-9]+)+)\)", re.IGNORECASE),
        ]:
            for match in pattern.finditer(text):
                references.add(cls._normalize_figure_id(match.group(1)))
        return references

    @staticmethod
    def _horizontal_overlap_ratio(
        figure_bbox: dict[str, float], paragraph_bbox: dict[str, float]
    ) -> float:
        overlap = min(figure_bbox["right"], paragraph_bbox["right"]) - max(
            figure_bbox["left"],
            paragraph_bbox["left"],
        )
        if overlap <= 0:
            return 0.0
        figure_width = max(figure_bbox["right"] - figure_bbox["left"], 1e-9)
        paragraph_width = max(paragraph_bbox["right"] - paragraph_bbox["left"], 1e-9)
        return overlap / min(figure_width, paragraph_width)

    @staticmethod
    def _spatial_distance(
        figure_bbox: dict[str, float], paragraph_bbox: dict[str, float]
    ) -> float:
        return abs(paragraph_bbox["center_y"] - figure_bbox["center_y"])

    @classmethod
    def _is_spatial_candidate(
        cls,
        *,
        figure_bbox: dict[str, float],
        paragraph_bbox: dict[str, float],
    ) -> bool:
        overlap = cls._horizontal_overlap_ratio(figure_bbox, paragraph_bbox)
        if overlap < MIN_HORIZONTAL_OVERLAP:
            return False

        vertically_near = (
            abs(paragraph_bbox["center_y"] - figure_bbox["center_y"])
            < VERTICAL_PROXIMITY_THRESHOLD
        )
        below_band = (
            0 <= paragraph_bbox["top"] - figure_bbox["bottom"] < CAPTION_BAND_THRESHOLD
        )
        above_band = (
            0 <= figure_bbox["top"] - paragraph_bbox["bottom"] < CAPTION_BAND_THRESHOLD
        )
        return vertically_near or below_band or above_band

    @classmethod
    def _score_paragraph_for_figure(
        cls,
        *,
        paragraph_text: str,
        figure_id: str,
        caption_terms: list[str],
    ) -> tuple[int, bool]:
        lower = paragraph_text.lower()
        score = 0
        matched_figure = False

        patterns = cls._figure_reference_patterns(figure_id)
        if any(pattern.search(paragraph_text) for pattern in patterns):
            score += 10
            matched_figure = True

        if caption_terms and any(term in lower for term in caption_terms):
            score += 5

        if (
            "figure" in lower
            or "fig." in lower
            or "chart" in lower
            or "diagram" in lower
            or "table" in lower
        ):
            score += 1

        normalized_figure_id = cls._normalize_figure_id(figure_id)
        referenced_ids = cls._extract_figure_reference_ids(paragraph_text)
        if referenced_ids and normalized_figure_id not in referenced_ids:
            return -100, False
        elif "figure" in lower and not matched_figure and normalized_figure_id:
            score -= 5

        return score, matched_figure

    def _search_request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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
        return self._slug(DEFAULT_NAME_PREFIX)

    def _ensure_target_index(self, *, index_name: str, hard_refresh: bool) -> None:
        if hard_refresh:
            self._log(f"Hard refresh enabled; deleting index '{index_name}'")
            self._search_delete_if_exists(f"/indexes/{quote(index_name)}")

        self._log(f"Creating or updating index '{index_name}'")
        search_index_client = self.ai_search_service.get_search_index_client()
        search_index_client.create_or_update_index(
            build_shared_index(
                name=index_name, embedding_dimensions=self.embedding_dimensions
            )
        )

    def _upload_records(
        self, *, index_name: str, records: list[dict[str, Any]]
    ) -> None:
        self._log(f"Uploading {len(records)} record(s) to index '{index_name}'")
        for start in range(0, len(records), 500):
            batch = records[start : start + 500]
            self._log(
                f"Uploading batch {(start // 500) + 1} with {len(batch)} record(s) to '{index_name}'"
            )
            body = {
                "value": [
                    {"@search.action": "mergeOrUpload", **record} for record in batch
                ]
            }
            self._search_request(
                "POST", f"/indexes/{quote(index_name)}/docs/index", body
            )

    def _save_artifact(
        self, *, container_name: str, blob_name: str, payload: Any
    ) -> str:
        if self.storage_service:
            return self.storage_service.upload_json(
                container_name=container_name,
                blob_name=blob_name,
                payload=payload,
            )

        local_path = Path("local_documents") / container_name / blob_name
        self.local_output_store.save(payload, str(local_path))
        return str(local_path)

    def _save_text_artifact(
        self, *, container_name: str, blob_name: str, text: str
    ) -> str:
        if self.storage_service:
            return self.storage_service.upload_text(
                container_name=container_name,
                blob_name=blob_name,
                text=text,
            )

        local_path = Path("local_documents") / container_name / blob_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(text, encoding="utf-8")
        return str(local_path)

    def _write_binary_artifact(
        self,
        *,
        container_name: str,
        blob_name: str,
        data: bytes,
        content_type: str,
    ) -> str:
        if self.storage_service:
            return self.storage_service.upload_bytes(
                container_name=container_name,
                blob_name=blob_name,
                data=data,
                content_type=content_type,
            )

        local_path = Path("local_documents") / container_name / blob_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return str(local_path)

    def _load_demo_files(self, demo_dir: Path) -> list[Path]:
        if not demo_dir.exists():
            raise FileNotFoundError(f"Demo folder not found: {demo_dir}")
        files = [path for path in demo_dir.iterdir() if path.is_file()]
        if not files:
            raise ValueError(f"No demo files found in {demo_dir}")
        return sorted(files)

    @staticmethod
    def _metadata_from_payload(
        payload: dict[str, Any], source_name: str
    ) -> dict[str, Any]:
        metadata = payload.get("metadata") or {}
        image_metadata = metadata.get("image") or {}
        return DocumentLayoutNoSkillV2Service._metadata_record(
            source_type=str(metadata.get("source_type") or "text"),
            category=str(metadata.get("category") or ""),
            topic=str(metadata.get("topic") or ""),
            subtopic=str(metadata.get("subtopic") or ""),
            source_url=str(metadata.get("source_url") or source_name),
            source_url_text=str(metadata.get("source_url_text") or source_name),
            source_name=str(metadata.get("source_name") or source_name),
            image=DocumentLayoutNoSkillV2Service._image_metadata_record(
                page_number=image_metadata.get(
                    "page_number", metadata.get("page_number")
                ),
                figure_id=str(
                    image_metadata.get("figure_id") or metadata.get("figure_id") or ""
                ),
                summary_method=str(
                    image_metadata.get("summary_method")
                    or metadata.get("summary_method")
                    or ""
                ),
                bounding_regions=image_metadata.get("bounding_regions") or [],
                ocr_text=str(image_metadata.get("ocr_text") or ""),
                caption=str(image_metadata.get("caption") or ""),
            ),
        )

    def _pdf_source_url(self, *, chunk_container: str, path: Path) -> str:
        self._log(f"Persisting PDF source artifact for '{path.name}'")
        with path.open("rb") as f:
            return self._write_binary_artifact(
                container_name=chunk_container,
                blob_name=f"sources/{path.name}",
                data=f.read(),
                content_type="application/pdf",
            )

    def _responses_text(
        self,
        *,
        deployment: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self._log(
            f"Azure OpenAI start: text response generation using deployment '{deployment}'"
        )
        try:
            text = self.openai_service.responses_text(
                deployment=deployment,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except OpenAIServiceError as exc:
            raise OpenAIApiError.from_service_error(exc) from exc
        self._log(
            f"Azure OpenAI complete: text response generation using deployment '{deployment}'"
        )
        return text

    def _responses_json_with_image(
        self,
        *,
        deployment: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> dict[str, Any]:
        self._log(
            f"Azure OpenAI start: multimodal grounded interpretation using deployment '{deployment}'"
        )
        try:
            payload = self.openai_service.responses_multimodal_json(
                deployment=deployment,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_bytes=image_bytes,
                content_type=mime_type,
            )
        except OpenAIServiceError as exc:
            raise OpenAIApiError.from_service_error(exc) from exc
        self._log(
            f"Azure OpenAI complete: multimodal grounded interpretation using deployment '{deployment}'"
        )
        return payload

    def _responses_structured_with_image(
        self,
        *,
        deployment: str,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> dict[str, Any]:
        self._log(
            f"Azure OpenAI start: multimodal grounded interpretation using deployment '{deployment}'"
        )
        try:
            payload = self.openai_service.responses_multimodal_structured(
                deployment=deployment,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
                image_bytes=image_bytes,
                content_type=mime_type,
            )
        except OpenAIServiceError as exc:
            raise OpenAIApiError.from_service_error(exc) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                "Structured interpretation response was not a JSON object."
            )
        self._log(
            f"Azure OpenAI complete: multimodal grounded interpretation using deployment '{deployment}'"
        )
        return payload

    def _embed_text(self, text: str) -> list[float]:
        normalized = self._searchable_text(text)
        if not normalized:
            raise ValueError("Cannot embed empty text.")
        self._log(
            "Generating text embedding "
            f"(deployment='{self.embedding_deployment}', chars={len(normalized)})"
        )
        try:
            return self.openai_service.embeddings(
                deployment=self.embedding_deployment,
                text=normalized,
            )
        except OpenAIServiceError as exc:
            raise OpenAIApiError.from_service_error(exc) from exc

    @classmethod
    def _guess_visual_heuristics(
        cls,
        *,
        caption: str,
        figure_ocr_text: str,
        relevant_text: str,
        surrounding_text: str,
    ) -> dict[str, Any]:
        combined = cls._searchable_text(
            " ".join([caption, figure_ocr_text, relevant_text, surrounding_text])
        )
        lower = combined.lower()
        figure_type = "figure"
        if any(term in lower for term in ["line chart", "line graph"]):
            figure_type = "line-chart"
        elif any(term in lower for term in ["bar chart", "bar graph", "bars"]):
            figure_type = "bar-chart"
        elif any(term in lower for term in ["pie chart", "donut chart"]):
            figure_type = "pie-chart"
        elif any(term in lower for term in ["diagram", "workflow", "process flow"]):
            figure_type = "diagram"
        elif any(term in lower for term in ["table", "tabular"]):
            figure_type = "table"
        elif any(term in lower for term in ["map"]):
            figure_type = "map"

        keywords: list[str] = []
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", lower):
            if token not in keywords:
                keywords.append(token)
            if len(keywords) >= 8:
                break

        return {
            "likely_figure_type": figure_type,
            "keywords": keywords,
        }

    @classmethod
    def _caption_keywords(cls, caption: str) -> list[str]:
        terms: list[str] = []
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", caption.lower()):
            if token not in terms:
                terms.append(token)
        return terms[:8]

    def _select_relevant_text(
        self,
        *,
        figure_id: str,
        caption: str,
        figure_bbox: dict[str, float] | None,
        page_number: int,
        page_paragraphs: dict[int, list[dict[str, Any]]],
    ) -> tuple[str, str]:
        same_page = page_paragraphs.get(page_number, [])
        candidate_count = len(same_page)
        if not same_page or not figure_bbox:
            self._log(
                f"Figure '{figure_id}' has no geometry-aware paragraph candidates "
                f"(same_page_candidates={candidate_count}, figure_bbox={bool(figure_bbox)})"
            )
            return "", ""

        caption_terms = self._caption_keywords(caption)
        filtered = [
            paragraph
            for paragraph in same_page
            if self._is_spatial_candidate(
                figure_bbox=figure_bbox,
                paragraph_bbox=paragraph["bbox"],
            )
        ]
        self._log(
            f"Figure '{figure_id}' candidate filtering "
            f"(same_page_candidates={candidate_count}, spatial_candidates={len(filtered)})"
        )
        if not filtered:
            self._log(
                f"Figure '{figure_id}' has no figure-local spatial context candidates"
            )
            return "", ""

        scored: list[tuple[int, float, int, dict[str, Any], bool]] = []
        figure_match_found = False
        for paragraph in filtered:
            score, matched_figure = self._score_paragraph_for_figure(
                paragraph_text=paragraph["text"],
                figure_id=figure_id,
                caption_terms=caption_terms,
            )
            figure_match_found = figure_match_found or matched_figure
            distance = self._spatial_distance(figure_bbox, paragraph["bbox"])
            scored.append(
                (score, distance, paragraph["index"], paragraph, matched_figure)
            )

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        eligible = [item for item in scored if item[0] > 0]
        if not eligible:
            self._log(
                f"Figure '{figure_id}' has no positively scored figure-local context candidates"
            )
            return "", ""

        relevant_records = [item[3] for item in eligible[:MAX_RELEVANT_PARAGRAPHS]]
        relevant_indexes = {record["index"] for record in relevant_records}

        remaining = [
            item for item in eligible if item[3]["index"] not in relevant_indexes
        ]
        remaining.sort(key=lambda item: (item[1], item[2]))
        surrounding_records = [
            item[3] for item in remaining[:MAX_SURROUNDING_PARAGRAPHS]
        ]

        self._log(
            f"Figure '{figure_id}' context selection "
            f"(figure_match_found={figure_match_found}, relevant_count={len(relevant_records)}, "
            f"surrounding_count={len(surrounding_records)})"
        )
        relevant = " ".join(record["text"] for record in relevant_records)
        surrounding = " ".join(record["text"] for record in surrounding_records)
        return self._searchable_text(relevant), self._searchable_text(surrounding)

    def _generate_document_summary(
        self, *, source_name: str, document_text: str
    ) -> str:
        normalized = self._searchable_text(document_text)
        if not normalized:
            self._log(
                f"Skipping document summary for '{source_name}' because no normalized text was available"
            )
            return ""

        deployment = self.chat_deployment or self.verbalization_deployment
        if not deployment:
            self._log(
                f"Skipping document summary for '{source_name}' because no chat deployment is configured"
            )
            return ""

        sample = normalized[:12000]
        self._log(
            f"Generating document summary for '{source_name}' "
            f"(sample_chars={len(sample)}, deployment='{deployment}')"
        )
        system_prompt = (
            "You write short grounded document summaries for indexing pipelines. "
            "Use only the provided text. Keep the result concise and factual."
        )
        user_prompt = (
            f"Source: {source_name}\n\n"
            "Write a short summary of the document's topic, major sections, and purpose in 4 sentences or fewer. "
            "Do not speculate beyond the provided text.\n\n"
            f"DOCUMENT TEXT:\n{sample}"
        )
        try:
            summary = self._searchable_text(
                self._responses_text(
                    deployment=deployment,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            )
            self._log(
                f"Generated document summary for '{source_name}' "
                f"(summary_chars={len(summary)})"
            )
            return summary
        except ValueError as exc:
            self._log(
                f"Document summary generation failed for '{source_name}'; continuing without it. {exc}"
            )
            return ""

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
        return self._searchable_text(getattr(result, "content", "") or "")

    def _build_figure_analysis_payload(
        self,
        *,
        source_name: str,
        source_url: str,
        page_number: int,
        figure_id: str,
        caption: str,
        bounding_regions: list[dict[str, Any]],
        figure_ocr_text: str,
        relevant_text: str,
        surrounding_text: str,
        document_summary: str,
        visual_heuristics: dict[str, Any],
        image_artifact_uri: str,
    ) -> dict[str, Any]:
        self._log(
            f"Building figure-analysis payload for figure '{figure_id}' on page {page_number} "
            f"(ocr_chars={len(figure_ocr_text)}, relevant_chars={len(relevant_text)}, "
            f"surrounding_chars={len(surrounding_text)}, summary_chars={len(document_summary)})"
        )
        return {
            "source_name": source_name,
            "source_url": source_url,
            "page_number": page_number,
            "figure_id": figure_id,
            "caption": caption,
            "bounding_regions": bounding_regions,
            "ocr_text": figure_ocr_text,
            "relevant_associated_text": relevant_text,
            "surrounding_text": surrounding_text,
            "document_summary": document_summary,
            "visual_heuristics": visual_heuristics,
            "image_artifact_uri": image_artifact_uri,
            "image_reference_mode": "memory-first",
        }

    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            text = cls._searchable_text(str(item))
            if text:
                normalized.append(text)
        return normalized

    @classmethod
    def _validate_grounded_interpretation(
        cls, grounded: dict[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(grounded, dict):
            raise ValueError("Structured interpretation payload was not a JSON object.")

        figure_type = cls._optional_text(grounded.get("figure_type")) or "figure"
        what_it_shows = cls._optional_text(grounded.get("what_it_shows"))
        if not what_it_shows:
            raise ValueError(
                "Structured interpretation payload is missing non-empty 'what_it_shows'."
            )

        supporting_context = grounded.get("supporting_context")
        if not isinstance(supporting_context, dict):
            raise ValueError(
                "Structured interpretation payload is missing 'supporting_context'."
            )

        normalized_supporting_context = {
            key: cls._normalize_string_list(supporting_context.get(key))
            for key in [
                "image_evidence",
                "ocr_evidence",
                "relevant_text_evidence",
                "surrounding_text_evidence",
                "document_summary_evidence",
            ]
        }

        return {
            "figure_type": figure_type,
            "what_it_shows": what_it_shows,
            "key_relationships": cls._normalize_string_list(
                grounded.get("key_relationships")
            ),
            "supporting_context": normalized_supporting_context,
            "uncertainties": cls._normalize_string_list(grounded.get("uncertainties")),
            "confidence_notes": cls._optional_text(grounded.get("confidence_notes")),
        }

    def _interpret_figure(
        self, *, figure_bytes: bytes, analysis_payload: dict[str, Any]
    ) -> dict[str, Any]:
        self._log(
            f"Starting grounded interpretation for figure '{analysis_payload['figure_id']}' "
            f"on page {analysis_payload['page_number']}"
        )
        system_prompt = (
            "You are a grounded figure interpretation model. Interpret the image visually first, "
            "then ground your interpretation using OCR, caption, relevant associated text, surrounding text, "
            "and document summary in that order of authority. Never let weaker context override stronger evidence. "
            "If evidence is ambiguous, say so."
        )
        user_prompt = (
            "Interpret this figure using only the evidence in this request.\n\n"
            f"SOURCE NAME:\n{analysis_payload['source_name']}\n\n"
            f"FIGURE ID:\n{analysis_payload['figure_id']}\n\n"
            f"PAGE NUMBER:\n{analysis_payload['page_number']}\n\n"
            f"CAPTION:\n{analysis_payload['caption']}\n\n"
            f"OCR TEXT:\n{analysis_payload['ocr_text']}\n\n"
            f"RELEVANT ASSOCIATED TEXT:\n{analysis_payload['relevant_associated_text']}\n\n"
            f"SURROUNDING TEXT:\n{analysis_payload['surrounding_text']}\n\n"
            f"DOCUMENT SUMMARY:\n{analysis_payload['document_summary']}\n\n"
            f"VISUAL HEURISTICS:\n{json.dumps(analysis_payload['visual_heuristics'], ensure_ascii=False)}\n\n"
            "Priority order:\n"
            "1. Image + OCR/caption\n"
            "2. Relevant associated text\n"
            "3. Surrounding text\n"
            "4. Document summary\n\n"
            "Return a structured interpretation that matches the required schema."
        )
        grounded = self._responses_structured_with_image(
            deployment=self.interpret_deployment,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=FIGURE_INTERPRETATION_SCHEMA,
            image_bytes=figure_bytes,
        )
        grounded = self._validate_grounded_interpretation(grounded)
        self._log(
            f"Completed grounded interpretation for figure '{analysis_payload['figure_id']}' "
            f"(type='{grounded.get('figure_type')}', uncertainties={len(grounded.get('uncertainties') or [])})"
        )
        return grounded

    @classmethod
    def _markdown_from_grounded(
        cls,
        *,
        grounded: dict[str, Any],
        analysis_payload: dict[str, Any],
    ) -> str:
        relationships = grounded.get("key_relationships") or []
        uncertainties = grounded.get("uncertainties") or []
        supporting_context = grounded.get("supporting_context") or {}
        evidence_lines: list[str] = []
        for key in [
            "image_evidence",
            "ocr_evidence",
            "relevant_text_evidence",
            "surrounding_text_evidence",
            "document_summary_evidence",
        ]:
            values = supporting_context.get(key) or []
            for value in values[:3]:
                normalized = cls._searchable_text(str(value))
                if normalized:
                    evidence_lines.append(f"- {normalized}")

        context_lines: list[str] = []
        if analysis_payload.get("relevant_associated_text"):
            context_lines.append(analysis_payload["relevant_associated_text"])
        if analysis_payload.get("document_summary"):
            context_lines.append(analysis_payload["document_summary"])

        lines = [
            "# Figure Summary",
            "",
            cls._searchable_text(
                str(grounded.get("what_it_shows") or "No summary available.")
            ),
            "",
            "## Interpretation",
            cls._searchable_text(
                " ".join(
                    str(value)
                    for value in relationships
                    if cls._searchable_text(str(value))
                )
            )
            or "Interpretation details were limited to the grounded evidence available.",
            "",
            "## Evidence",
        ]
        if evidence_lines:
            lines.extend(evidence_lines)
        else:
            lines.append("- Evidence was limited to the extracted figure inputs.")

        lines.extend(["", "## Context"])
        if context_lines:
            lines.append(cls._searchable_text(" ".join(context_lines)))
        else:
            lines.append(
                "Minimal contextual text was available beyond the figure-local evidence."
            )

        if uncertainties:
            lines.extend(["", "## Uncertainties"])
            lines.extend(
                f"- {cls._searchable_text(str(value))}"
                for value in uncertainties
                if cls._searchable_text(str(value))
            )

        return "\n".join(lines).strip()

    def _verbalize_figure(
        self,
        *,
        grounded: dict[str, Any],
        analysis_payload: dict[str, Any],
    ) -> str:
        self._log(
            f"Starting figure verbalization for '{analysis_payload['figure_id']}' "
            f"using deployment '{self.verbalization_deployment}'"
        )
        system_prompt = (
            "You convert grounded figure interpretations into concise semantic markdown for retrieval. "
            "Do not invent facts. Keep metadata out of the prose unless it materially helps retrieval."
        )
        user_prompt = (
            "Write markdown with sections: Figure Summary, Interpretation, Evidence, Context. "
            "Base it strictly on the grounded interpretation and extracted evidence below.\n\n"
            f"GROUNDED INTERPRETATION:\n{json.dumps(grounded, ensure_ascii=False, indent=2)}\n\n"
            f"EXTRACTED EVIDENCE:\n{json.dumps(analysis_payload, ensure_ascii=False, indent=2)}"
        )
        try:
            markdown = self._responses_text(
                deployment=self.verbalization_deployment,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ).strip()
            self._log(
                f"Completed figure verbalization for '{analysis_payload['figure_id']}' "
                f"(markdown_chars={len(markdown)})"
            )
            return markdown
        except ValueError as exc:
            self._log(
                f"Figure verbalization failed for '{analysis_payload['figure_id']}'; using deterministic fallback. {exc}"
            )
            return self._markdown_from_grounded(
                grounded=grounded, analysis_payload=analysis_payload
            )

    def _json_records(
        self,
        *,
        path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_name = path.stem
        metadata = self._metadata_from_payload(payload, source_name)
        chunks = self._chunk_text(
            str(payload.get("content") or payload.get("fulltext") or ""),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._log(f"Derived {len(chunks)} text chunk(s) from JSON source '{path.name}'")

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
        return records, []

    def _pdf_records(
        self,
        *,
        path: Path,
        chunk_container: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._log(f"Analyzing PDF source '{path.name}' with Document Intelligence")
        result, operation_id = self.di_service.analyze_file_with_figures(
            path=path,
            model_id="prebuilt-layout",
            content_format=DocumentContentFormat.TEXT,
        )
        source_name = path.stem
        source_url = self._pdf_source_url(chunk_container=chunk_container, path=path)
        base_metadata = self._metadata_record(
            source_type="text",
            category=DEFAULT_TEXT_RECORD_CATEGORY,
            topic=DEFAULT_TEXT_RECORD_TOPIC,
            subtopic=DEFAULT_TEXT_RECORD_SUBTOPIC,
            source_url=source_url,
            source_url_text=path.name,
            source_name=path.name,
            summary_method="document_text",
        )
        page_dimensions = self._page_dimensions_map(result)

        page_text: dict[int, list[str]] = {}
        page_paragraphs: dict[int, list[dict[str, Any]]] = {}
        full_document_parts: list[str] = []
        for index, paragraph in enumerate(getattr(result, "paragraphs", None) or []):
            content = self._searchable_text(getattr(paragraph, "content", "") or "")
            if not content:
                continue
            full_document_parts.append(content)
            paragraph_record = self._build_paragraph_record(
                paragraph=paragraph,
                index=index,
                page_dimensions=page_dimensions,
            )
            if not paragraph_record:
                continue
            page_number = paragraph_record["page_number"]
            page_text.setdefault(page_number, []).append(paragraph_record["text"])
            page_paragraphs.setdefault(page_number, []).append(paragraph_record)
        self._log(
            f"Collected paragraph text for '{path.name}' "
            f"(pages_with_text={len(page_text)}, paragraph_count={len(full_document_parts)}, "
            f"geometry_paragraph_count={sum(len(items) for items in page_paragraphs.values())})"
        )

        document_summary = self._generate_document_summary(
            source_name=path.name,
            document_text=" ".join(full_document_parts),
        )

        records: list[dict[str, Any]] = []
        support_artifacts: list[dict[str, Any]] = []
        ordinal = 1
        text_chunk_count = 0
        for page_number in sorted(page_text):
            page_content = " ".join(page_text[page_number])
            for chunk in self._chunk_text(
                page_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ):
                content = f"Page {page_number}: {chunk}"
                records.append(
                    {
                        "id": self._make_record_id(source_name, "text", ordinal),
                        "metadata": {
                            **base_metadata,
                            "image": self._image_metadata_record(
                                page_number=page_number
                            ),
                        },
                        "content": content,
                        "contentVector": self._embed_text(content),
                    }
                )
                ordinal += 1
                text_chunk_count += 1
        self._log(
            f"Derived {text_chunk_count} text chunk record(s) from PDF source '{path.name}'"
        )

        figures = getattr(result, "figures", None) or []
        self._log(f"Found {len(figures)} figure(s) in PDF source '{path.name}'")
        for figure in figures:
            if not operation_id:
                raise ValueError(
                    "Document Intelligence analyze result did not return operation_id for figures."
                )
            figure_id = getattr(figure, "id", None) or f"figure-{ordinal}"
            caption = self._searchable_text(
                getattr(getattr(figure, "caption", None), "content", None) or ""
            )
            page_number = self._page_number_from_regions(figure) or 0
            bounding_regions = self._bounding_regions_record(
                getattr(figure, "bounding_regions", None)
            )
            figure_bbox = self._bbox_from_bounding_regions(
                bounding_regions=bounding_regions,
                page_dimensions=page_dimensions,
            )
            self._log(
                f"Processing figure '{figure_id}' from '{path.name}' "
                f"(page={page_number}, caption_present={bool(caption)}, regions={len(bounding_regions)})"
            )
            figure_bytes = self._extract_figure_bytes(
                result_id=operation_id, figure_id=figure_id
            )
            figure_ocr_text = self._extract_figure_text(figure_bytes)
            relevant_text, surrounding_text = self._select_relevant_text(
                figure_id=figure_id,
                caption=caption,
                figure_bbox=figure_bbox,
                page_number=page_number,
                page_paragraphs=page_paragraphs,
            )
            visual_heuristics = self._guess_visual_heuristics(
                caption=caption,
                figure_ocr_text=figure_ocr_text,
                relevant_text=relevant_text,
                surrounding_text=surrounding_text,
            )
            self._log(
                f"Figure '{figure_id}' context prepared "
                f"(ocr_chars={len(figure_ocr_text)}, relevant_chars={len(relevant_text)}, "
                f"surrounding_chars={len(surrounding_text)})"
            )
            image_artifact_uri = self._write_binary_artifact(
                container_name=chunk_container,
                blob_name=f"figures-v2/{source_name}/{figure_id}.png",
                data=figure_bytes,
                content_type="image/png",
            )
            self._log(
                f"Persisted figure image for '{figure_id}' to '{image_artifact_uri}'"
            )
            analysis_payload = self._build_figure_analysis_payload(
                source_name=path.name,
                source_url=source_url,
                page_number=page_number,
                figure_id=figure_id,
                caption=caption,
                bounding_regions=bounding_regions,
                figure_ocr_text=figure_ocr_text,
                relevant_text=relevant_text,
                surrounding_text=surrounding_text,
                document_summary=document_summary,
                visual_heuristics=visual_heuristics,
                image_artifact_uri=image_artifact_uri,
            )
            analysis_artifact = self._save_artifact(
                container_name=chunk_container,
                blob_name=f"figure-analysis-v2/{source_name}/{figure_id}.json",
                payload=analysis_payload,
            )
            self._log(
                f"Persisted figure-analysis artifact for '{figure_id}' to '{analysis_artifact}'"
            )
            grounded = self._interpret_figure(
                figure_bytes=figure_bytes,
                analysis_payload=analysis_payload,
            )
            grounded_artifact = self._save_artifact(
                container_name=chunk_container,
                blob_name=f"figure-grounded-v2/{source_name}/{figure_id}.json",
                payload=grounded,
            )
            self._log(
                f"Persisted grounded interpretation artifact for '{figure_id}' to '{grounded_artifact}'"
            )
            figure_markdown = self._verbalize_figure(
                grounded=grounded,
                analysis_payload=analysis_payload,
            )
            markdown_artifact = self._save_text_artifact(
                container_name=chunk_container,
                blob_name=f"figure-markdown-v2/{source_name}/{figure_id}.md",
                text=figure_markdown,
            )
            self._log(
                f"Persisted markdown artifact for '{figure_id}' to '{markdown_artifact}'"
            )
            support_artifacts.append(
                {
                    "source": path.name,
                    "figure_id": figure_id,
                    "analysis_artifact": analysis_artifact,
                    "grounded_artifact": grounded_artifact,
                    "markdown_artifact": markdown_artifact,
                    "image_artifact": image_artifact_uri,
                }
            )
            records.append(
                {
                    "id": self._make_record_id(source_name, "image", ordinal),
                    "metadata": self._metadata_record(
                        source_type="image",
                        category=base_metadata["category"],
                        topic=base_metadata["topic"],
                        subtopic=base_metadata["subtopic"],
                        source_url=image_artifact_uri,
                        source_url_text=f"{path.name} figure {figure_id}",
                        source_name=path.name,
                        page_number=page_number,
                        figure_id=figure_id,
                        summary_method="aoai-grounded-interpretation+semantic-markdown",
                        bounding_regions=bounding_regions,
                        ocr_text=figure_ocr_text,
                        caption=caption,
                    ),
                    "content": figure_markdown,
                    "contentVector": self._embed_text(figure_markdown),
                }
            )
            ordinal += 1
            self._log(
                f"Completed figure record for '{figure_id}' "
                f"(content_chars={len(figure_markdown)}, total_support_artifacts={len(support_artifacts)})"
            )

        if not records:
            raise ValueError(f"No extractable content found in PDF: {path}")
        self._log(
            f"Completed PDF source '{path.name}' "
            f"(record_count={len(records)}, support_artifact_count={len(support_artifacts)})"
        )
        return records, support_artifacts

    def _process_source(
        self,
        *,
        path: Path,
        chunk_container: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        suffix = path.suffix.lower()
        self._log(f"Processing source '{path.name}' as '{suffix}'")
        if suffix == ".json":
            return self._json_records(
                path=path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
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
        support_artifacts: list[dict[str, Any]],
    ) -> str:
        artifact = {
            "source": source_path.name,
            "recordCount": len(records),
            "supportArtifactCount": len(support_artifacts),
            "supportArtifacts": support_artifacts,
            "records": records,
        }
        blob_name = f"{source_path.stem}.json"
        artifact_uri = self._save_artifact(
            container_name=container_name, blob_name=blob_name, payload=artifact
        )
        self._log(
            f"Persisted derived source artifact for '{source_path.name}' to '{artifact_uri}' "
            f"(records={len(records)}, support_artifacts={len(support_artifacts)})"
        )
        return artifact_uri

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
        self._log(
            f"Starting single-source v2 run for '{path}' "
            f"(chunk_container='{chunk_container}', chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
        )

        records, support_artifacts = self._process_source(
            path=path,
            chunk_container=chunk_container,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        artifact_uri = self._write_source_artifact(
            container_name=chunk_container,
            source_path=path,
            records=records,
            support_artifacts=support_artifacts,
        )
        index_name = self._target_index_name(name_prefix)
        self._ensure_target_index(index_name=index_name, hard_refresh=hard_refresh)
        self._upload_records(index_name=index_name, records=records)
        self._log(
            f"Single-source v2 run complete for '{path.name}' "
            f"(target_index='{index_name}', records={len(records)}, support_artifacts={len(support_artifacts)})"
        )

        return {
            "pipeline": "document-layout-no-skill-v2",
            "mode": "single-source",
            "source": str(path),
            "chunk_container": chunk_container,
            "derived_artifact": artifact_uri,
            "support_artifact_count": len(support_artifacts),
            "support_artifacts": support_artifacts,
            "target_index": index_name,
            "record_count": len(records),
            "embedding": {
                "mode": "azure_openai",
                "deployment": self.embedding_deployment,
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
        all_support_artifacts: list[dict[str, Any]] = []

        self._log(
            f"Starting v2 demo run from '{demo_path}' "
            f"(source_count={len(files)}, chunk_container='{chunk_container}', "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
        )
        for path in files:
            self._log(f"Deriving v2 records for '{path.name}'")
            records, support_artifacts = self._process_source(
                path=path,
                chunk_container=chunk_container,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            artifact_uri = self._write_source_artifact(
                container_name=chunk_container,
                source_path=path,
                records=records,
                support_artifacts=support_artifacts,
            )
            derived_artifacts.append(
                {
                    "source": path.name,
                    "artifact": artifact_uri,
                    "record_count": len(records),
                    "support_artifact_count": len(support_artifacts),
                }
            )
            all_records.extend(records)
            all_support_artifacts.extend(support_artifacts)

        index_name = self._target_index_name(name_prefix)
        self._ensure_target_index(index_name=index_name, hard_refresh=hard_refresh)
        self._upload_records(index_name=index_name, records=all_records)
        self._log(
            f"Demo finished with {len(all_records)} indexed record(s) "
            f"and {len(all_support_artifacts)} support artifact(s)"
        )

        return {
            "pipeline": "document-layout-no-skill-v2",
            "mode": "demo",
            "demo_dir": str(demo_path),
            "chunk_container": chunk_container,
            "target_index": index_name,
            "source_count": len(files),
            "record_count": len(all_records),
            "support_artifact_count": len(all_support_artifacts),
            "derived_artifacts": derived_artifacts,
            "support_artifacts": all_support_artifacts,
            "embedding": {
                "mode": "azure_openai",
                "deployment": self.embedding_deployment,
                "field": "contentVector",
                "dimensions": self.embedding_dimensions,
            },
            "records": all_records,
        }
