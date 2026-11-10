import os, time, json, logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
from flask_limiter import Limiter

load_dotenv()
import arabic_rag as rag

app = Flask(__name__)
app.json.ensure_ascii = False
API_KEY = os.environ.get("SERVICE_API_KEY")

# One JSON object per line: easy to grep now, easy to ship to Kafka/ELK later
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("api")

def client_id():
    # Rate-limit per API key; fall back to the real client IP (behind cloudflared)
    return (request.headers.get("X-API-Key")
            or request.headers.get("CF-Connecting-IP")
            or request.remote_addr)

limiter = Limiter(key_func=client_id, app=app, storage_uri="memory://")

@app.before_request
def check_key():
    g.t0 = time.perf_counter()
    if API_KEY and request.path != "/health":
        if request.headers.get("X-API-Key") != API_KEY:
            return jsonify(error="unauthorized"), 401

@app.get("/health")
def health():
    return jsonify(status="ok!", chunks=len(rag.chunks))

@app.post("/ask")
@limiter.limit("10/minute")
def ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify(error="question is required"), 400
    k = min(int(data.get("k", 4)), 10)

    t = time.perf_counter()
    idx, scores = rag.retrieve(q, k)
    t_retrieve = time.perf_counter() - t

    t = time.perf_counter()
    text, usage = rag.answer(q, idx)
    t_llm = time.perf_counter() - t

    log.info(json.dumps({
        "event": "ask", "client": client_id()[:8], "q_chars": len(q), "k": k,
        "retrieve_ms": round(t_retrieve * 1000), "llm_ms": round(t_llm * 1000),
        "in_tokens": usage.input_tokens, "out_tokens": usage.output_tokens,
        "top_score": round(float(scores[0]), 3),
    }))
    return jsonify(
        answer=text,
        sources=[{"file": rag.meta[i]["file"], "chunk": rag.meta[i]["chunk"],
                  "score": round(float(s), 3)} for i, s in zip(idx, scores)],
    )


from flask import Response, stream_with_context

def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.post("/ask/stream")
@limiter.limit("10/minute")
def ask_stream():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify(error="question is required"), 400
    k = min(int(data.get("k", 4)), 10)
    idx, scores = rag.retrieve(q, k)

    def gen():
        t0 = time.perf_counter()
        yield sse("sources", [{"file": rag.meta[i]["file"], "chunk": rag.meta[i]["chunk"],
                               "score": round(float(s), 3)} for i, s in zip(idx, scores)])
        first = True
        for kind, val in rag.answer_stream(q, idx):
            if kind == "text":
                if first:
                    log.info(json.dumps({"event": "first_token_ms",
                                         "ms": round((time.perf_counter() - t0) * 1000)}))
                    first = False
                yield sse("token", {"t": val})
            else:
                log.info(json.dumps({"event": "ask_stream", "in_tokens": val.input_tokens,
                                     "out_tokens": val.output_tokens}))
        yield sse("done", {})

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    app.run(port=5001)