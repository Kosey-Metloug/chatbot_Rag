from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db
from .models import Document
from .schemas import AskIn, AskOut, DocumentOut
from .services.rag import answer_question

app = FastAPI(title="Docs Q&A")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(select(Document).order_by(Document.id.desc())).all()


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, db: Session = Depends(get_db)):
    try:
        answer, sources = answer_question(db, body.question)
    except Exception as e:
        raise HTTPException(503, f"Model unavailable: {e}")
    return AskOut(answer=answer, sources=sources)
