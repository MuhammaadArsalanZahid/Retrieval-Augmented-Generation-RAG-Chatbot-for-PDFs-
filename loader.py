"""loader.py - PDF reading and chunking (Steps 3 and 4 of the guide)."""
from pypdf import PdfReader


def load_pdf(file_obj, file_name):
    """Read a PDF page by page. Returns a list of {"file", "page", "text"}.
    Empty pages are skipped. If the list is empty, the PDF is probably scanned."""
    reader = PdfReader(file_obj)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        text = " ".join(text.split())  # clean extra spaces / line breaks
        if text:
            pages.append({"file": file_name, "page": i, "text": text})
    return pages


def chunk_pages(pages, chunk_size=500, overlap=100):
    """Split every page into overlapping chunks.
    Each chunk remembers its file name and page number (needed for citations)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size - overlap
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # try not to cut a word in half: move end back to the last space
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start + step // 2:
                    end = space
            piece = text[start:end].strip()
            if len(piece) > 30:  # ignore tiny useless chunks
                chunks.append({"text": piece, "file": p["file"], "page": p["page"]})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
