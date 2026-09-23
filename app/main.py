from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db, init_db
from .models import Conversation, Document, Message
from .schemas import AskIn, AskOut, ConversationOut, DocumentOut, MessageOut
from .services.rag import answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()   # creates any table in models.py that doesn't exist yet
    yield


app = FastAPI(title="Docs Q&A", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- documents (read-only; files are added via the sync script) ----------
@app.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(select(Document).order_by(Document.id.desc())).all()


# ---------- conversations ----------
@app.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(db: Session = Depends(get_db)):
    conv = Conversation()
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@app.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    return db.scalars(select(Conversation).order_by(Conversation.id.desc())).all()


@app.get("/conversations/{conv_id}/messages", response_model=list[MessageOut])
def get_messages(conv_id: int, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv.messages


@app.delete("/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: int, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    db.delete(conv)
    db.commit()


@app.post("/conversations/{conv_id}/ask", response_model=AskOut)
def ask_in_conversation(conv_id: int, body: AskIn, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")

    history = conv.messages[-settings.history_messages:]   # prior turns only — before this question is added

    try:
        answer, sources = answer_question(db, history, body.question)
    except Exception as e:
        raise HTTPException(503, f"Model unavailable: {e}")

    db.add_all([
        Message(conversation_id=conv.id, role="user", content=body.question),
        Message(conversation_id=conv.id, role="assistant", content=answer, sources=sources),
    ])
    if conv.title == "New conversation":
        conv.title = body.question[:60]
    db.commit()
    return AskOut(answer=answer, sources=sources)


# kept for direct testing / backward compatibility — no conversation, so no history
@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, db: Session = Depends(get_db)):
    try:
        answer, sources = answer_question(db, [], body.question)
    except Exception as e:
        raise HTTPException(503, f"Model unavailable: {e}")
    return AskOut(answer=answer, sources=sources)


app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
