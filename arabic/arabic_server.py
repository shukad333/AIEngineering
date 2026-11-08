import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()
import arabic_rag as rag          # loads model + builds the index once, at startup

app = Flask(__name__)
app.json.ensure_ascii = False     # return Arabic as real characters, not \u escapes
API_KEY = os.environ.get("SERVICE_API_KEY")   # optional for now

@app.before_request
def check_key():
    if API_KEY and request.path != "/health":
        if request.headers.get("X-API-Key") != API_KEY:
            return jsonify(error="unauthorized"), 401

@app.get("/health")
def health():
    return jsonify(status="ok", chunks=len(rag.chunks))

@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify(error="question is required"), 400
    k = min(int(data.get("k", 4)), 10)
    idx, scores = rag.retrieve(q, k)
    return jsonify(
        answer=rag.answer(q, idx),
        sources=[{"file": rag.meta[i]["file"], "chunk": rag.meta[i]["chunk"],
                  "score": round(float(s), 3)} for i, s in zip(idx, scores)],
    )