from pathlib import Path
import fitz
from docx import Document


def extract_text(file_path: str) -> str:
    """
    Extract text from PDF, DOCX, or TXT files.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    # PDF processing
    if extension == ".pdf":
        extracted_pages = []

        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text()

                if page_text.strip():
                    extracted_pages.append(
                        f"[Page {page_number}]\n{page_text}"
                    )

        text = "\n".join(extracted_pages)

    # DOCX processing
    elif extension == ".docx":
        document = Document(path)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

    # TXT processing
    elif extension == ".txt":
        text = path.read_text(encoding="utf-8")

    else:
        raise ValueError(
            "Unsupported file type. Upload PDF, DOCX, or TXT."
        )

    if not text.strip():
        raise ValueError(
            "No readable text found. The document may be empty "
            "or scanned."
        )

    return text


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks