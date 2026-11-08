import pathlib
import numpy as np, anthropic
from sentence_transformers import SentenceTransformer

client = anthropic.Anthropic()
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def chunk(text, size=1500, overlap=150):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

def embed(texts):
    return model.encode(texts, normalize_embeddings=True)

def top_k(q_vec, vecs, k=4):
    return np.argsort(-(vecs @ q_vec))[:k]

def answer(question, chunks):
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))
    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1000,
        system="Answer only from the provided context. Cite chunk numbers. "
               "If the answer isn't there, say so.",
        messages=[{"role": "user",
                   "content": f"<context>\n{ctx}\n</context>\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")

chunks, meta = [], []
for p in pathlib.Path("../docs").glob("**/*"):
    if p.suffix in {".md", ".txt"}:
        text = p.read_text(errors="ignore")
        for j, c in enumerate(chunk(text)):
            chunks.append(c)
            meta.append({"file": p.name, "chunk": j})

vecs = embed(chunks)

def ask(q, k=4):
    q_vec = embed(["Represent this sentence for searching relevant passages: " + q])[0]
    idx = top_k(q_vec, vecs, k)
    print(answer(q, [chunks[i] for i in idx]))

ask("Hello who is shuhail")