from sqlalchemy.orm import Session

from ..config import settings
from . import vectorstore
from .llm import chat

SYSTEM = (
    "You answer ONLY from the provided context. "
    "If the answer is not in the context, say: 'I couldn't find this in the documents.' "
    "Mention the source file and location for each claim, as plain text — "
    "never invent file paths or markdown links."
)


def answer_question(db: Session, question: str) -> tuple[str, list[dict]]:
    hits = vectorstore.search(db, question, settings.top_k)
    if not hits:
        return "No documents are ready yet. Add a file to the folder and run the sync script.", []

    context = "\n\n".join(f"[{m['source']} {m['location']}]\n{t}" for t, m in hits)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    answer = chat(messages)

    seen, sources = set(), []
    for _, m in hits:
        key = (m["source"], m["location"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": m["source"], "location": m["location"]})
    return answer, sources
