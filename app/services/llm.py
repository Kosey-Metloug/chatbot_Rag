import ollama

from ..config import settings


def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=settings.embed_model, input=texts)["embeddings"]


def chat(messages: list[dict]) -> str:
    res = ollama.chat(
        model=settings.chat_model,
        messages=messages,
        options={"temperature": 0.1, "num_ctx": 8192},
        think=False,
    )
    return res["message"]["content"]
