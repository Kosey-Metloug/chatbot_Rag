from pathlib import Path

from sqlalchemy import select

from .config import settings
from .db import SessionLocal, init_db
from .models import Document
from .services.ingest import process_document

ALLOWED = {".pdf", ".md", ".markdown", ".txt"}


def sync():
    init_db()
    db = SessionLocal()
    try:
        folder = Path(settings.docs_dir)
        folder.mkdir(parents=True, exist_ok=True)
        files = [p for p in folder.iterdir() if p.suffix.lower() in ALLOWED]

        known = {d.filename: d for d in db.scalars(select(Document))}
        on_disk = {p.name for p in files}

        for path in files:
            if path.name in known:
                print(f"Skip (already ingested): {path.name}")
                continue
            print(f"New file: {path.name}")
            doc = Document(filename=path.name, status="processing")
            db.add(doc)
            db.commit()
            db.refresh(doc)
            process_document(db, doc, path)
            print(f"  -> {doc.status}, {doc.chunk_count} chunks" + (f" ({doc.error})" if doc.error else ""))

        for name, doc in known.items():
            if name not in on_disk:
                print(f"Removed from disk, deleting record: {name}")
                db.delete(doc)   # cascades to its chunks
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    sync()
