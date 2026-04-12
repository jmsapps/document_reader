import re


def _normalize_text_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def strip_figure_blocks_from_markdown(markdown: str) -> str:
    normalized = (markdown or "").strip()
    if not normalized:
        return ""

    without_figures = re.sub(
        r"<figure\b[^>]*>.*?</figure>",
        "",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    without_headers = re.sub(
        r"^\s*<!--\s*PageHeader=.*?-->\s*$",
        "",
        without_figures,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    compacted = re.sub(r"\n{3,}", "\n\n", without_headers)
    return compacted.strip()


def chunk_text_deterministic(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    normalized = _normalize_text_whitespace(text)
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


def _split_markdown_sections(markdown: str) -> list[tuple[str | None, str]]:
    lines = markdown.splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    current_header: str | None = None
    current_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if re.match(r"^#{1,6}\s+", line):
            if current_header is not None or current_lines:
                sections.append((current_header, current_lines))
            current_header = line.strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_header is not None or current_lines:
        sections.append((current_header, current_lines))

    result: list[tuple[str | None, str]] = []
    for header, section_lines in sections:
        section_body = "\n".join(section_lines).strip()
        if header or section_body:
            result.append((header, section_body))
    return result


def _split_blocks(section_body: str) -> list[str]:
    if not section_body.strip():
        return []
    return [
        block.strip()
        for block in re.split(r"\n\s*\n+", section_body)
        if block.strip()
    ]


def _is_table_block(block: str) -> bool:
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if "|" not in lines[0]:
        return False
    return bool(re.match(r"^\s*\|?[\s:-]+(?:\|[\s:-]+)+\|?\s*$", lines[1]))


def _split_table_block(
    block: str,
    *,
    header: str | None,
    chunk_size: int,
) -> list[str]:
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    if len(lines) <= 2:
        return [block.strip()]

    table_header = [lines[0], lines[1]]
    rows = lines[2:]
    prefix = [header] if header else []
    chunks: list[str] = []
    current_rows: list[str] = []

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate_parts = prefix + table_header + candidate_rows
        candidate_text = "\n".join(candidate_parts).strip()
        if current_rows and len(candidate_text) > chunk_size:
            chunks.append("\n".join(prefix + table_header + current_rows).strip())
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        chunks.append("\n".join(prefix + table_header + current_rows).strip())
    return chunks


def _is_list_line(line: str) -> bool:
    stripped = line.lstrip()
    return bool(
        re.match(r"^([-*+]\s+|\d+[.)]\s+)", stripped)
    )


def _split_list_items(block: str) -> list[str]:
    lines = block.splitlines()
    items: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if _is_list_line(line):
            if current:
                items.append(current)
            current = [line.rstrip()]
        else:
            if current:
                current.append(line.rstrip())
            else:
                current = [line.rstrip()]
    if current:
        items.append(current)
    return ["\n".join(item).strip() for item in items if "\n".join(item).strip()]


def _is_list_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return bool(lines) and all(
        _is_list_line(lines[0]) or _is_list_line(line) or line.startswith(" ")
        for line in lines
    )


def _split_list_block(
    block: str,
    *,
    header: str | None,
    chunk_size: int,
) -> list[str]:
    items = _split_list_items(block)
    if not items:
        return [block.strip()]

    prefix = [header] if header else []
    chunks: list[str] = []
    current_items: list[str] = []
    for item in items:
        candidate_items = current_items + [item]
        candidate_text = "\n\n".join(prefix + candidate_items).strip()
        if current_items and len(candidate_text) > chunk_size:
            chunks.append("\n\n".join(prefix + current_items).strip())
            current_items = [item]
        else:
            current_items = candidate_items
    if current_items:
        chunks.append("\n\n".join(prefix + current_items).strip())
    return chunks


def _split_prose_block(
    block: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    paragraphs = [
        _normalize_text_whitespace(part)
        for part in re.split(r"\n+", block)
        if _normalize_text_whitespace(part)
    ]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > chunk_size:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current.strip())

    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(
                chunk_text_deterministic(
                    chunk,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
    return final_chunks


def chunk_markdown_deterministic(
    markdown: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    normalized_markdown = (markdown or "").strip()
    if not normalized_markdown:
        return []

    size = max(100, int(chunk_size))
    overlap = max(0, min(int(chunk_overlap), size // 2))
    sections = _split_markdown_sections(normalized_markdown)
    rendered_chunks: list[str] = []

    for header, section_body in sections:
        prefix = [header] if header else []
        blocks = _split_blocks(section_body)
        if not blocks:
            if header:
                rendered_chunks.append(header)
            continue

        section_chunks: list[str] = []
        current_blocks: list[str] = []
        for block in blocks:
            normalized_block = block.strip()
            candidate_blocks = current_blocks + [normalized_block]
            candidate_text = "\n\n".join(prefix + candidate_blocks).strip()
            if current_blocks and len(candidate_text) > size:
                section_chunks.append("\n\n".join(prefix + current_blocks).strip())
                current_blocks = [normalized_block]
            else:
                current_blocks = candidate_blocks

            current_text = "\n\n".join(prefix + current_blocks).strip()
            if len(current_text) > size:
                current_blocks.pop()
                if current_blocks:
                    section_chunks.append("\n\n".join(prefix + current_blocks).strip())
                if _is_table_block(normalized_block):
                    section_chunks.extend(
                        _split_table_block(
                            normalized_block,
                            header=header,
                            chunk_size=size,
                        )
                    )
                elif _is_list_block(normalized_block):
                    section_chunks.extend(
                        _split_list_block(
                            normalized_block,
                            header=header,
                            chunk_size=size,
                        )
                    )
                else:
                    prose_chunks = _split_prose_block(
                        normalized_block,
                        chunk_size=size - len(header or ""),
                        chunk_overlap=overlap,
                    )
                    for prose_chunk in prose_chunks:
                        section_chunks.append(
                            "\n\n".join(prefix + [prose_chunk]).strip()
                        )
                current_blocks = []

        if current_blocks:
            section_chunks.append("\n\n".join(prefix + current_blocks).strip())

        if len(section_chunks) == 1:
            rendered_chunks.extend(section_chunks)
            continue

        total_parts = len(section_chunks)
        for index, section_chunk in enumerate(section_chunks, start=1):
            chunk_lines = section_chunk.splitlines()
            if header and chunk_lines and chunk_lines[0].strip() == header:
                body = "\n".join(chunk_lines[1:]).strip()
                rendered = "\n".join(
                    [
                        header,
                        f"Part {index} of {total_parts}",
                        "",
                        body,
                    ]
                ).strip()
            else:
                rendered = "\n".join(
                    [
                        f"Part {index} of {total_parts}",
                        "",
                        section_chunk,
                    ]
                ).strip()
            rendered_chunks.append(rendered)

    return [chunk for chunk in rendered_chunks if chunk.strip()]
