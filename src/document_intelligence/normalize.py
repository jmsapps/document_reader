import re
from typing import Any, Dict
from html import escape


def _span_list(obj: Any) -> list[dict]:
    spans = getattr(obj, "spans", None) or []
    return [
        {"offset": getattr(s, "offset", None), "length": getattr(s, "length", None)}
        for s in spans
    ]


def _bounding_regions(obj: Any) -> list[dict]:
    regions = getattr(obj, "bounding_regions", None) or []
    out = []
    for r in regions:
        out.append(
            {
                "page_number": getattr(r, "page_number", None),
                "polygon": getattr(r, "polygon", None),
            }
        )
    return out


def to_normalized_json(result: Any) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "model_id": result.model_id,
        "content": result.content,
        "pages": [],
        "paragraphs": [],
        "tables": [],
        "sections": [],
        "key_value_pairs": [],
        "figures": [],
    }

    for page in (result.pages or []):
        output["pages"].append(
            {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "words": [
                    {"text": word.content, "confidence": word.confidence}
                    for word in (page.words or [])
                ],
            }
        )

    for paragraph in (result.paragraphs or []):
        output["paragraphs"].append(
            {
                "role": paragraph.role,
                "text": paragraph.content,
                "spans": _span_list(paragraph),
                "bounding_regions": _bounding_regions(paragraph),
            }
        )

    for table in (result.tables or []):
        matrix = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
        for cell in table.cells:
            matrix[cell.row_index][cell.column_index] = cell.content or ""
        output["tables"].append(
            {
                "rows": table.row_count,
                "columns": table.column_count,
                "data": matrix,
                "spans": _span_list(table),
                "bounding_regions": _bounding_regions(table),
            }
        )

    for section in (getattr(result, "sections", None) or []):
        output["sections"].append(
            {
                "spans": _span_list(section),
                "elements": list(getattr(section, "elements", None) or []),
            }
        )

    for kv in (getattr(result, "key_value_pairs", None) or []):
        output["key_value_pairs"].append(
            {
                "key": getattr(kv.key, "content", None) if kv.key else None,
                "value": getattr(kv.value, "content", None) if kv.value else None,
                "confidence": kv.confidence,
            }
        )

    for figure in (getattr(result, "figures", None) or []):
        caption = getattr(figure, "caption", None)
        output["figures"].append(
            {
                "id": figure.id,
                "caption": getattr(caption, "content", None) if caption else None,
                "caption_spans": _span_list(caption) if caption else [],
                "spans": _span_list(figure),
                "bounding_regions": _bounding_regions(figure),
                "page_number": (
                    figure.bounding_regions[0].page_number
                    if figure.bounding_regions
                    else None
                ),
            }
        )

    return output


def to_raw_json(result: Any) -> Dict[str, Any]:
    return result.as_dict()


def _render_html(normalized: Dict[str, Any]) -> str:
    paragraphs = normalized.get("paragraphs", [])
    tables = normalized.get("tables", [])
    figures = normalized.get("figures", [])
    sections = normalized.get("sections", [])

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Document Intelligence Output</title>",
        "<style>body{font-family:Arial,sans-serif;line-height:1.45}"
        "table{border-collapse:collapse;margin:12px 0;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px;vertical-align:top}"
        "th{text-align:left;background:#f6f6f6}"
        "</style>",
        "</head><body>",
    ]

    value_like_re = re.compile(r"^(?:N/A|\$?\d[\d,]*(?:\.\d+)?%?|\.\d+%|•+)$")

    def _is_value_like(text: str) -> bool:
        return bool(value_like_re.match(text.strip()))

    def _span_start(spans: list[dict]) -> int | None:
        starts: list[Any] = [s.get("offset") for s in spans if isinstance(s.get("offset"), int)]

        if not starts:
            return None

        return min(starts)

    def _span_end(spans: list[dict]) -> int | None:
        ends = []
        for s in spans:
            offset = s.get("offset")
            length = s.get("length")
            if isinstance(offset, int) and isinstance(length, int):
                ends.append(offset + length)
        if not ends:
            return None
        return max(ends)

    table_ranges = []
    for t in tables:
        spans = t.get("spans", [])
        start = _span_start(spans)
        end = _span_end(spans)
        if start is not None and end is not None:
            table_ranges.append((start, end))

    figure_ranges = []
    for f in figures:
        spans = f.get("spans", [])
        start = _span_start(spans)
        end = _span_end(spans)
        if start is not None and end is not None:
            figure_ranges.append((start, end))

    def _in_table_span(spans: list[dict]) -> bool:
        p_start = _span_start(spans)
        p_end = _span_end(spans)
        if p_start is None or p_end is None:
            return False
        for t_start, t_end in table_ranges:
            if p_start < t_end and t_start < p_end:
                return True
        return False

    def _in_figure_span(spans: list[dict]) -> bool:
        p_start = _span_start(spans)
        p_end = _span_end(spans)
        if p_start is None or p_end is None:
            return False
        for f_start, f_end in figure_ranges:
            if p_start < f_end and f_start < p_end:
                return True
        return False

    def _page_hint(item: dict) -> int:
        regions = item.get("bounding_regions", [])
        for r in regions:
            page = r.get("page_number")
            if isinstance(page, int):
                return page
        return 10**9

    def _y_hint(item: dict) -> float:
        regions = item.get("bounding_regions", [])
        ys: list[float] = []
        for r in regions:
            polygon = r.get("polygon") or []
            if isinstance(polygon, list) and len(polygon) >= 2:
                for idx in range(1, len(polygon), 2):
                    y = polygon[idx]
                    if isinstance(y, (int, float)):
                        ys.append(float(y))
        if not ys:
            return 10**9
        return min(ys)

    blocks = []
    para_seq = 0
    table_seq = 0
    figure_caption_texts = {
        (f.get("caption") or "").strip()
        for f in figures
        if isinstance(f.get("caption"), str) and (f.get("caption") or "").strip()
    }
    for p_index, p in enumerate(paragraphs):
        text = (p.get("text") or "").strip()
        if not text:
            continue
        if text in figure_caption_texts:
            continue
        if _in_table_span(p.get("spans", [])):
            continue
        if _in_figure_span(p.get("spans", [])):
            continue
        blocks.append(
            {
                "type": "paragraph",
                "role": p.get("role"),
                "text": text,
                "spans": p.get("spans", []),
                "bounding_regions": p.get("bounding_regions", []),
                "seq": para_seq,
                "source_ref": f"/paragraphs/{p_index}",
            }
        )
        para_seq += 1

    for t_index, t in enumerate(tables):
        if _in_figure_span(t.get("spans", [])):
            continue
        blocks.append(
            {
                "type": "table",
                "data": t.get("data", []),
                "spans": t.get("spans", []),
                "bounding_regions": t.get("bounding_regions", []),
                "seq": table_seq,
                "source_ref": f"/tables/{t_index}",
            }
        )
        table_seq += 1

    for f_index, f in enumerate(figures):
        fig_id = (f.get("id") or "").strip() or f"figure-{f_index + 1}"
        fig_content = (f.get("caption") or "").strip()
        anchor_spans = f.get("caption_spans") or f.get("spans", [])
        blocks.append(
            {
                "type": "figure",
                "figure_id": fig_id,
                "figure_content": fig_content,
                "spans": anchor_spans,
                "bounding_regions": f.get("bounding_regions", []),
                "seq": f_index,
                "source_ref": f"/figures/{f_index}",
            }
        )

    ref_to_section_rank: dict[str, int] = {}
    if isinstance(sections, list) and sections:
        visited_sections: set[int] = set()
        active_sections: set[int] = set()
        ordered_refs: list[str] = []
        seen_refs: set[str] = set()

        def _walk_section(section_idx: int) -> None:
            if section_idx < 0 or section_idx >= len(sections):
                return
            if section_idx in active_sections:
                return
            if section_idx in visited_sections:
                return

            active_sections.add(section_idx)
            section = sections[section_idx] if isinstance(sections[section_idx], dict) else {}
            for ref in section.get("elements", []):
                if not isinstance(ref, str):
                    continue
                if ref.startswith("/sections/"):
                    child = ref.rsplit("/", 1)[-1]
                    if child.isdigit():
                        _walk_section(int(child))
                    continue
                if ref.startswith("/paragraphs/") or ref.startswith("/tables/"):
                    if ref not in seen_refs:
                        seen_refs.add(ref)
                        ordered_refs.append(ref)

            active_sections.remove(section_idx)
            visited_sections.add(section_idx)

        for idx in range(len(sections)):
            _walk_section(idx)

        for rank, ref in enumerate(ordered_refs):
            ref_to_section_rank[ref] = rank

    def _block_sort_key(block: dict):
        start = _span_start(block.get("spans", []))
        span_missing = 1 if start is None else 0
        span_key = start if start is not None else 10**12
        section_key = ref_to_section_rank.get(block.get("source_ref", ""), 10**9)
        page_key = _page_hint(block)
        y_key = _y_hint(block)
        # Prefer global text-span flow first. Fall back to section/page geometry when spans are missing.
        return (span_missing, span_key, section_key, page_key, y_key, block.get("seq", 0))

    blocks.sort(key=_block_sort_key)

    in_list = False
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.get("type") == "figure":
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(
                f"<figure id=\"{escape(block.get('figure_id', ''))}\">"
                f"{escape(block.get('figure_content', ''))}</figure>"
            )
            i += 1
            continue

        if block.get("type") == "table":
            if in_list:
                parts.append("</ul>")
                in_list = False
            rows = block.get("data", [])
            if rows:
                parts.append("<table><tbody>")
                for r_i, row in enumerate(rows):
                    parts.append("<tr>")
                    cell_tag = "th" if r_i == 0 else "td"
                    for cell in row:
                        parts.append(f"<{cell_tag}>{escape(str(cell))}</{cell_tag}>")
                    parts.append("</tr>")
                parts.append("</tbody></table>")
            i += 1
            continue

        role = block.get("role")
        text = block.get("text", "")

        if role == "title":
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h1>{escape(text)}</h1>")
            i += 1
            continue

        if role == "sectionHeading":
            if in_list:
                parts.append("</ul>")
                in_list = False
            parts.append(f"<h2>{escape(text)}</h2>")
            i += 1
            continue

        bullet = text.startswith("· ") or text.startswith("- ")
        if bullet:
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{escape(text[2:].strip())}</li>")
            i += 1
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False

            # Conservative key-value pairing for fee-style lines.
            if i + 1 < len(blocks) and blocks[i + 1].get("type") == "paragraph":
                nxt = blocks[i + 1]
                nxt_role = nxt.get("role")
                nxt_text = (nxt.get("text") or "").strip()
                if (
                    role is None
                    and nxt_role is None
                    and nxt_text
                    and not nxt_text.startswith(("· ", "- "))
                    and _is_value_like(nxt_text)
                    and not _is_value_like(text)
                ):
                    parts.append("<table><tbody>")
                    parts.append("<tr>")
                    parts.append(f"<th>{escape(text)}</th>")
                    parts.append(f"<td>{escape(nxt_text)}</td>")
                    parts.append("</tr></tbody></table>")
                    i += 2
                    continue

            parts.append(f"<p>{escape(text)}</p>")
            i += 1

    if in_list:
        parts.append("</ul>")

    parts.append("</body></html>")
    return "".join(parts)


def to_html_payload(result: Dict) -> str:
    normalized = to_normalized_json(result)
    return _render_html(normalized)

def get_metadata(result: Dict) -> Dict[str, Any]:
    normalized = to_normalized_json(result)
    paragraphs = normalized.get("paragraphs", [])

    title = next((p.get("text") for p in paragraphs if p.get("role") == "title"), None)
    first_section = next(
        (p.get("text") for p in paragraphs if p.get("role") == "sectionHeading"),
        None,
    )

    return {
        "subject": title,
        "category": first_section,
        "model_id": normalized.get("model_id"),
        "pages_count": len(normalized.get("pages", [])),
        "paragraphs_count": len(paragraphs),
        "tables_count": len(normalized.get("tables", [])),
        "figures_count": len(normalized.get("figures", [])),
    }
