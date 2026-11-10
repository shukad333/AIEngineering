import re, pathlib
import numpy as np, anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
embedder = SentenceTransformer("BAAI/bge-m3")
DOCS_DIR = pathlib.Path(__file__).parent / "../docs"

# --- Arabic normalization (helps matching; applied to text we embed) ---
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]")  # tashkeel + tatweel
def normalize(t):
    t = DIACRITICS.sub("", t)
    t = re.sub("[إأآا]", "ا", t)
    t = t.replace("ى", "ي")
    return t

# --- Sentence-aware chunking (handles Arabic ؟ and English . ! ?) ---
def split_sentences(text):
    parts = re.split(r"(?<=[.!?؟۔])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]

def chunk(text, max_chars=1000, overlap_sents=1):
    sents, chunks, cur = split_sentences(text), [], []
    for s in sents:
        if sum(len(x) for x in cur) + len(s) > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def embed(texts):
    return embedder.encode([normalize(t) for t in texts], normalize_embeddings=True)

# --- Build index ---
chunks, meta = [], []
for p in DOCS_DIR.glob("**/*"):
    print("XXXX ",p)
    if p.suffix.lower() in {".md", ".txt"}:
        for j, c in enumerate(chunk(p.read_text(errors="ignore"))):
            chunks.append(c); meta.append({"file": p.name, "chunk": j})
assert chunks, f"No docs found in {DOCS_DIR}"
vecs = embed(chunks)
print(len(chunks), "chunks")

def retrieve(q, k=4):
    q_vec = embed([q])[0]                    # bge-m3 needs no query prefix
    sims = vecs @ q_vec
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def answer(q, idx):
    ctx = "\n\n".join(f"[{meta[i]['file']}#{meta[i]['chunk']}] {chunks[i]}" for i in idx)
    msg = client.messages.create(
        model=MODEL, max_tokens=2000,
        system="Answer only from the context, citing [file#chunk]. "
               "Reply in the same language as the question. "
               "If the answer is not in the context, say so in that language.",
        messages=[{"role": "user", "content": f"<context>\n{ctx}\n</context>\n\nQuestion: {q}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text"), msg.usage

def ask(q, k=4):
    idx, scores = retrieve(q, k)
    for i, s in zip(idx, scores):
        print(f"{s:.3f} {meta[i]['file']}#{meta[i]['chunk']} | {chunks[i][:80]!r}")
    text, _ = answer(q, idx)
    print("\n" + text)

def answer_stream(q, idx):
    ctx = "\n\n".join(f"[{meta[i]['file']}#{meta[i]['chunk']}] {chunks[i]}" for i in idx)
    with client.messages.stream(
        model=MODEL, max_tokens=2000,
        system="Answer only from the context, citing [file#chunk]. "
               "Reply in the same language as the question. "
               "If the answer is not in the context, say so in that language.",
        messages=[{"role": "user", "content": f"<context>\n{ctx}\n</context>\n\nQuestion: {q}"}],
    ) as stream:
        for text in stream.text_stream:
            yield ("text", text)
        yield ("usage", stream.get_final_message().usage)

# if __name__ == "__main__":
#     ask("ما هي عاصمة قطر؟")
#     ask("When did Doha Metro open?")   # English question, Arabic doc: cross-lingual