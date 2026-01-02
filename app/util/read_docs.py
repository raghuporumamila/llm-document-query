from pypdf import PdfReader
import io

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

from pathlib import Path

def read_pdfs_as_bytes(folder_path: str):
    folder = Path(folder_path)

    for pdf_file in folder.glob("*.pdf"):
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
        yield pdf_file.name, pdf_bytes

