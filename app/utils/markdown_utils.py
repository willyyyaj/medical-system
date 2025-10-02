import re


SECTION_TITLES = [
    "看診原因",
    "診斷結果",
    "治療計畫",
    "注意事項",
]


def _normalize_line_endings(text: str) -> str:
    if not text:
        return text
    # Normalize CRLF/CR to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # If literal \n appears a lot (common from JSON transport), carefully convert
    # Only do this when we detect many literal sequences and very few real newlines
    if text.count("\\n") >= 3 and text.count("\n") <= 1:
        text = text.replace("\\n", "\n")
    return text


def _is_heading_variant(line: str, expected: str) -> bool:
    s = line.strip()
    # Remove markdown markers and trailing colons
    s = re.sub(r"^[#*\s]+", "", s)
    s = re.sub(r"[：:]+$", "", s)
    s = s.strip()
    return s == expected


def normalize_summary_markdown(text: str) -> str:
    """
    Force the summary into the exact Markdown structure we expect without altering content:
    - Ensure first heading is "## 看診重點摘要"
    - Ensure section titles are bold: **看診原因**, **診斷結果**, **治療計畫**, **注意事項**
    - Ensure a blank line between blocks
    """
    if not text:
        return text

    text = _normalize_line_endings(text)
    lines = text.split("\n")

    # Determine if the first non-empty line is the summary title; if not, we will insert it
    first_idx = 0
    while first_idx < len(lines) and lines[first_idx].strip() == "":
        first_idx += 1

    result_lines = []

    # Always start with the standardized heading
    result_lines.append("## 看診重點摘要")
    result_lines.append("")

    # Skip an existing variant of the title in the original
    skip_first_line = False
    if first_idx < len(lines):
        first_line = lines[first_idx]
        if _is_heading_variant(first_line, "看診重點摘要"):
            skip_first_line = True

    i = first_idx + (1 if skip_first_line else 0)

    def append_blank_line_once():
        if len(result_lines) == 0 or result_lines[-1] != "":
            result_lines.append("")

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # Collapse excessive blank lines to a single blank
        if stripped == "":
            append_blank_line_once()
            i += 1
            continue

        # Normalize section headings to bold
        matched_section = None
        for title in SECTION_TITLES:
            if _is_heading_variant(stripped, title):
                matched_section = title
                break

        if matched_section is not None:
            # Ensure one blank line before a section (but avoid duplicating directly after title)
            if len(result_lines) > 0 and result_lines[-1] != "":
                result_lines.append("")
            result_lines.append(f"**{matched_section}**")
            result_lines.append("")
            i += 1
            # Skip following duplicate blank lines in source to avoid extra spacing
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            continue

        # Normal content line: keep as-is (trim right whitespace only)
        result_lines.append(raw.rstrip())
        i += 1

    # Trim extra blank lines at the end
    while len(result_lines) > 0 and result_lines[-1] == "":
        result_lines.pop()

    # Collapse any double blanks in the middle (ensure at most one)
    collapsed = []
    for line in result_lines:
        if line == "" and len(collapsed) > 0 and collapsed[-1] == "":
            continue
        collapsed.append(line)

    return "\n".join(collapsed)


