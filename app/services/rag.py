from ..config import settings
from . import vectorstore
from .llm import chat

SYSTEM = (
    "You answer ONLY from the provided context. "
    "If the answer is not in the context, say: 'I couldn't find this in the documents.' "
    "Mention the source file and location for each claim, as plain text — "
    "never invent file paths or markdown links."
)


def standalone_question(history: list, question: str) -> str:
    """Turn a follow-up like 'and the Basic plan?' into a question that makes
    sense searched on its own, using the recent conversation for context."""
    if not history:
        return question

    convo = "\n".join(f"{m.role}: {m.content}" for m in history)
    prompt = (
        "Given the conversation so far, rewrite the last question so it can be "
        "understood on its own, with no missing context. "
        "If the last question is already standalone, return it unchanged. "
        "Return ONLY the rewritten question, nothing else — no preamble, no quotes.\n\n"
        f"Conversation:\n{convo}\n\nLast question: {question}"
    )
    rewritten = chat([{"role": "user", "content": prompt}]).strip()
    return rewritten or question


def answer_question(db, history: list, question: str) -> tuple[str, list[dict]]:
    search_query = standalone_question(history, question)
    hits = vectorstore.search(db, search_query, settings.top_k)

    if not hits:
        return "No documents are ready yet. Add a file to the folder and run the sync script.", []

    context = "\n\n".join(f"[{m['source']} {m['location']}]\n{t}" for t, m in hits)
    messages = [{"role": "system", "content": SYSTEM}]
    messages += [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    answer = chat(messages)

    seen, sources = set(), []
    for _, m in hits:
        key = (m["source"], m["location"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": m["source"], "location": m["location"]})
    return answer, sources
