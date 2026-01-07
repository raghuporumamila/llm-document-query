from fastapi import FastAPI
import app.service.rag_service as rag_service
from app.model.model import QueryRequest

app = FastAPI()

@app.post("/query")
def query_data(req: QueryRequest):
    answer = rag_service.query_data(req.user_id, req.question)
    return {"answer": answer}
