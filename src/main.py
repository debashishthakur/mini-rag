from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from vectorstore import upsert_documents
from qa import answer_question_with_pinecone_llama

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    mode: Literal["semantic", "lexical", "hybrid"] = "semantic"
    top_k: int = 3

@app.on_event("startup")
def startup_event():
    upsert_documents()

@app.post("/ask")
def ask_qonfido(request: QueryRequest):
    result = answer_question_with_pinecone_llama(
        request.question,
        top_k=request.top_k,
        mode=request.mode,
    )
    return result
