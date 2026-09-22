import sys
from pathlib import Path
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
import ollama

MODEL = "qwen3.5:2b"              # answers the questions
EMBED_MODEL = "nomic-embed-text"  # finds the right passages in the PDFs

db = chromadb.PersistentClient(path="./db", settings=Settings(anonymized_telemetry=False))
col = db.get_or_create_collection("pdfs", metadata={"hnsw:space": "cosine"})

SYSTEM = (
    "You answer ONLY from the provided context. "
    "If the answer is not in the context, say: 'I couldn't find this in the documents.' "
    "Mention the source file and page for each claim."
)

def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]

def chunk(text, size=600, overlap=100):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out

def ingest(pdf_path):
    name = Path(pdf_path).name
    ids, docs, metas = [], [], []
    for page_no, page in enumerate(fitz.open(pdf_path), start=1):
        text = page.get_text().strip()
        for j, c in enumerate(chunk(text)):
            ids.append(f"{name}-p{page_no}-c{j}")
            docs.append(c)
            metas.append({"source": name, "page": page_no})
    for i in range(0, len(docs), 16):
        b = slice(i, i + 16)
        col.upsert(ids=ids[b], documents=docs[b], metadatas=metas[b], embeddings=embed(docs[b]))
    print(f"Ingested {name}: {len(docs)} chunks")

def ask(question, k=3):
    res = col.query(query_embeddings=embed([question]), n_results=k)
    context = "\n\n".join(
        f"[{m['source']} p.{m['page']}]\n{d}"
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    )
    stream = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.1, "num_ctx": 8192},
        think=False,
        stream=True,
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        for pdf in Path("pdfs").glob("*.pdf"):
            ingest(pdf)
    else:
        while True:
            q = input("\nYou: ")
            if q.lower() in {"exit", "quit"}:
                break
            ask(q)
