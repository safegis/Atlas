"""Extract plain text from uploaded bytes for RAG ingest (txt, md, pdf, images with OCR)."""

from __future__ import annotations

import io
from typing import Any


def _ext(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower().strip()


def extract_text_from_upload(filename: str, raw: bytes) -> tuple[str, dict[str, Any]]:
    """
    Returns (text, meta) where meta may include extraction hints.
    Raises ValueError with user-facing message on failure / unsupported type.
    """
    ext = _ext(filename)
    meta: dict[str, Any] = {"extension": ext or "unknown"}

    if ext in ("txt", "md", "markdown"):
        text = raw.decode("utf-8", errors="replace")
        meta["method"] = "utf-8"
        return text, meta

    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ValueError("PDF support requires the 'pypdf' package.") from e
        reader = PdfReader(io.BytesIO(raw))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n\n".join(parts).strip()
        meta["method"] = "pypdf"
        meta["pages"] = len(reader.pages)
        if not text:
            raise ValueError(
                "No text extracted from PDF (it may be scanned images only). "
                "Try OCR software or export as text."
            )
        return text, meta

    if ext in ("png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "tif"):
        try:
            from PIL import Image
        except ImportError as e:
            raise ValueError("Image support requires the 'Pillow' package.") from e
        try:
            import pytesseract
        except ImportError as e:
            raise ValueError(
                "Image OCR requires 'pytesseract' and a system Tesseract install "
                "(e.g. brew install tesseract)."
            ) from e
        try:
            img = Image.open(io.BytesIO(raw))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            text = (pytesseract.image_to_string(img) or "").strip()
        except Exception as e:
            raise ValueError(
                f"Image OCR failed: {e}. Ensure Tesseract is installed and on PATH."
            ) from e
        meta["method"] = "pytesseract"
        if not text:
            raise ValueError(
                "No text read from image (OCR empty). Try a clearer scan or a text-based PDF."
            )
        return text, meta

    raise ValueError(
        f"Unsupported type '.{ext}'. Use .txt, .md, .pdf, or an image (.png, .jpg, …)."
    )
