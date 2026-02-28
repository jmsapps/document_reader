from pathlib import Path
from typing import Any, Dict, Literal, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.core.exceptions import HttpResponseError

from ...document_intelligence.normalize import get_metadata, to_html_payload, to_raw_json
from .service import DocumentIntelligenceService

ContentFormat = Literal["text", "markdown", "html"]

SUPPORTED_DIRECT = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".heif",
    ".bmp",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
}


def _is_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def _detect_kind(src: str) -> Tuple[str, str]:
    if _is_url(src):
        return "url", Path(urlparse(src).path).suffix.lower()
    return "file", Path(src).suffix.lower()


def _load_url_bytes(url: str) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; document-reader/1.0)",
            "Accept": "*/*",
        },
    )
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def _tiny_markdown_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    html_lines = []
    in_code = False
    in_list = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            in_code = not in_code
            html_lines.append("<pre><code>" if in_code else "</code></pre>")
            continue

        if in_code:
            escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(escaped)
            continue

        if line.startswith("#"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            level = min(6, len(line) - len(line.lstrip("#")))
            html_lines.append(f"<h{level}>{line[level:].strip()}</h{level}>")
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{stripped[2:].strip()}</li>")
            continue

        if in_list:
            html_lines.append("</ul>")
            in_list = False

        if stripped == "":
            html_lines.append("<br/>")
        else:
            html_lines.append(f"<p>{line}</p>")

    if in_list:
        html_lines.append("</ul>")

    body = "\n".join(html_lines)
    return (
        "<!doctype html><html><head><meta charset='utf-8'></head>"
        f"<body>{body}</body></html>"
    )


def _to_di_content_format(content_format: ContentFormat) -> DocumentContentFormat:
    # For HTML output we render from extracted structure, so TEXT is the fastest DI format.
    if content_format == "markdown":
        return DocumentContentFormat.MARKDOWN

    return DocumentContentFormat.TEXT


def _serialize_raw(result: Any, content_format: ContentFormat) -> Dict[str, Any]:
    payload = to_raw_json(result)
    payload["contentFormat"] = content_format
    payload["metadata"] = get_metadata(result)

    if content_format == "html":
        payload["content"] = to_html_payload(result)

    return payload


def analyze_any(
    src: str,
    model_id: str = "prebuilt-layout",
    content_format: ContentFormat = "text",
) -> Dict[str, Any]:
    service = DocumentIntelligenceService()
    kind, ext = _detect_kind(src)
    di_content_format = _to_di_content_format(content_format)

    if kind == "url":
        try:
            result = service.analyze_url(
                url=src, model_id=model_id, content_format=di_content_format
            )
        except HttpResponseError as exc:
            message = (exc.message or "").lower()
            # DI sometimes cannot fetch externally accessible URLs due to source-side restrictions.
            if exc.status_code == 400 and "could not download the file" in message:
                try:
                    data = _load_url_bytes(src)
                except (HTTPError, URLError, TimeoutError) as fetch_exc:
                    raise ValueError(f"Failed to download URL locally: {fetch_exc}") from fetch_exc
                result = service.analyze_bytes(
                    data=data,
                    model_id=model_id,
                    content_format=di_content_format,
                )
            else:
                raise
        return _serialize_raw(result, content_format)

    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if ext == ".md":
        html = _tiny_markdown_to_html(path.read_text(encoding="utf-8"))
        result = service.analyze_bytes(
            data=html.encode("utf-8"),
            model_id=model_id,
            content_format=di_content_format,
        )

        return _serialize_raw(result, content_format)

    if ext not in SUPPORTED_DIRECT:
        raise ValueError(
            f"Unsupported file extension: {ext or '<none>'}. Convert to PDF/HTML first."
        )

    result = service.analyze_file(path=path, model_id=model_id, content_format=di_content_format)

    return _serialize_raw(result, content_format)
