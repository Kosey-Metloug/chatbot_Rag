from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from ..models import Chunk, Document
from . import vectorstore
from .chunking import chunk
from .extract import extract


def process_document(db: Session, doc: Document, path: Path) -> None:
    try:
        db.execute(delete(Chunk).where(Chunk.document_id == doc.id))  # safe to re-run
        sections = extract(path)
        items = [(loc, c) for loc, text in sections for c in chunk(text)]
        if not items:
            raise ValueError("No text found (scanned PDF?)")
        doc.chunk_count = vectorstore.add_chunks(db, doc.id, items)
        doc.status = "ready"
        doc.error = None
    except Exception as e:
        db.rollback()
        doc.status = "failed"
        doc.error = str(e)[:500]
    db.commit()
