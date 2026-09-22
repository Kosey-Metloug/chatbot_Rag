from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Chunk, Document
from .llm import embed


def add_chunks(db: Session, document_id: int, items: list[tuple[str, str]]) -> int:
    texts = [text for _, text in items]
    vectors = []
    for i in range(0, len(texts), 16):
        vectors += embed(texts[i:i + 16])

    db.add_all(
        Chunk(document_id=document_id, location=loc, content=text, embedding=vec)
        for (loc, text), vec in zip(items, vectors)
    )
    return len(items)


def search(db: Session, question: str, k: int) -> list[tuple[str, dict]]:
    q = embed([question])[0]
    stmt = (
        select(Chunk, Document.filename)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.status == "ready")
        .order_by(Chunk.embedding.cosine_distance(q))
        .limit(k)
    )
    return [(c.content, {"source": name, "location": c.location}) for c, name in db.execute(stmt)]
