from pypdf import PdfReader
import io
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

def read_pdfs_as_bytes(folder_path: str):
    folder = Path(folder_path)

    for pdf_file in folder.glob("*.pdf"):
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
        yield pdf_file.name, pdf_bytes

def split_multiple_pdfs(pdf_texts: dict):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    all_chunks = []
    for filename, text in pdf_texts.items():
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)

    return all_chunks