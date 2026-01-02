from fastapi import FastAPI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

import app.util.read_docs as pdf_util
import app.service.rag_service as rag_service

app = FastAPI()
user_id = 'raghu'
embeddings = OpenAIEmbeddings()

def create_vector_store(chunks):
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=[{"user_id": user_id}] * len(chunks)
    )
    vectorstore.save_local(f"vectorstore/{user_id}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

def split_multiple_pdfs(pdf_texts: dict):
    all_chunks = []
    for filename, text in pdf_texts.items():
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)

    return all_chunks

pdf_texts: dict = dict()
for filename, pdf_bytes in pdf_util.read_pdfs_as_bytes("./docs"):
    print(filename, len(pdf_bytes))
    pdf_texts[filename] = pdf_util.extract_text_from_pdf(pdf_bytes)

chunks = split_multiple_pdfs(pdf_texts)
print(len(chunks))
create_vector_store(chunks)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_data(req: QueryRequest):
    answer = rag_service.query_data(user_id, req.question)
    return {"answer": answer}
