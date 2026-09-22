import re
from pathlib import Path

import fitz  # PyMuPDF

HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _pdf(path: Path) -> list[tuple[str, str]]:
    sections = []
    for page_no, page in enumerate(fitz.open(path), start=1):
        text = page.get_text().strip()
        if text:
            sections.append((f"p.{page_no}", text))
    return sections


def _markdown(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(HEADING.finditer(text))

    if not matches:
        return [("document", text.strip())] if text.strip() else []

    sections = []
    intro = text[:matches[0].start()].strip()
    if intro:
        sections.append(("intro", intro))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        title = m.group(2).strip()[:100]
        if body:
            sections.append((title or "section", body))
    return sections


def _plain(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [("document", text)] if text else []


def extract(path: Path) -> list[tuple[str, str]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf(path)
    if ext in {".md", ".markdown"}:
        return _markdown(path)
    if ext == ".txt":
        return _plain(path)
    raise ValueError(f"Unsupported file type: {ext}")
