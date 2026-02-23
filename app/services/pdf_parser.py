"""
PDF and document parsing service.

Uses PyMuPDF (fitz) for PDF extraction — fast, accurate, handles complex layouts.
Supports: PDF, DOCX, TXT, MD
"""

import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks by character count, respecting sentence boundaries."""
    if not text or not text.strip():
        return []

    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ".!?\n" and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: keep last portion
            words = current_chunk.split()
            overlap_words = words[-overlap // 4:] if len(words) > overlap // 4 else words
            current_chunk = " ".join(overlap_words) + " " + sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def parse_pdf(filepath: str) -> List[Dict[str, Any]]:
    """Parse PDF and return list of {text, page, metadata}."""
    import fitz
    doc = fitz.open(filepath)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_num + 1,
                "char_count": len(text),
            })
    doc.close()
    logger.info(f"Parsed PDF: {filepath} → {len(pages)} pages")
    return pages


def parse_docx(filepath: str) -> List[Dict[str, Any]]:
    """Parse DOCX and return list of {text, page, metadata}."""
    from docx import Document
    doc = Document(filepath)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": full_text, "page": 1, "char_count": len(full_text)}]


def parse_txt(filepath: str) -> List[Dict[str, Any]]:
    """Parse plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text.strip(), "page": 1, "char_count": len(text)}]


def parse_markdown(filepath: str) -> List[Dict[str, Any]]:
    """Parse markdown file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text.strip(), "page": 1, "char_count": len(text)}]


PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_txt,
    ".md": parse_markdown,
}


def parse_document(filepath: str) -> List[Dict[str, Any]]:
    """Auto-detect format and parse document."""
    ext = os.path.splitext(filepath)[1].lower()
    parser = PARSERS.get(ext)
    if not parser:
        raise ValueError(f"Unsupported format: {ext}. Supported: {list(PARSERS.keys())}")
    return parser(filepath)
