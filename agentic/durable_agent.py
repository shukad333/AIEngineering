import json, sqlite3, sys, time, uuid
import anthropic
from flask import Flask, request, jsonify
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()
MODEL, MAX_STEPS = "claude-sonnet-5-5", 10
db = sqlite3.connect("runs.db", isolation_level=None, check_same_thread=False)
db.execute("""CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, status TEXT,
              messages TEXT, pending TEXT, step INTEGER, tokens INTEGER, result TEXT)""")

# ---- tools (fake side effects, so we can focus on the runtime) ----
def lookup_customer(name):
    time.sleep(2)                                   # slow on purpose, so you can kill the worker
    return json.dumps({"name": name, "plan": "pro", "email": f"{name.lower()}@example.com"})
def send_email(to, body):
    return f"email sent to {to}"

def schema(name, desc, props):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": list(props)}}

def auth(name):
    pass
S = {"type": "string"}
TOOLS = {
    "lookup_customer": {"fn": lookup_customer, "risky": False,
                        "schema": schema("lookup_customer", "Look up a customer by name.", {"name": S})},
    "send_email": {"fn": send_email, "risky": True,
                   "schema": schema("send_email", "Send an email to a customer.", {"to": S, "body": S})},
}


def run_tool(block):
    try:
        out = TOOLS[block["name"]]["fn"](**block["input"])
    except Exception as e:
        out = f"Error: {e}"
    return {"type": "tool_result", "tool_use_id": block["id"], "content": str(out)}

# ---- persistence ----
def save(run_id, status, messages, pending, step, tokens, result=None):
    db.execute("UPDATE runs SET status=?, messages=?, pending=?, step=?, tokens=?, result=? WHERE id=?",
               (status, json.dumps(messages), json.dumps(pending) if pending else None,
                step, tokens, result, run_id))

# ---- one step of one run ----
def step(run_id):
    msgs, n, tokens = db.execute(
        "SELECT messages, step, tokens FROM runs WHERE id=?", (run_id,)).fetchone()
    messages = json.loads(msgs)
    resp = client.messages.create(model=MODEL, max_tokens=2000, messages=messages,
                                  tools=[t["schema"] for t in TOOLS.values()])
    tokens += resp.usage.input_tokens + resp.usage.output_tokens
    messages.append({"role": "assistant",
                     "content": [b.model_dump(exclude_none=True) for b in resp.content]})
    print(f"[{run_id[:6]}] step {n} -> {resp.stop_reason}")

    if resp.stop_reason != "tool_use":
        text = "".join(b.text for b in resp.content if b.type == "text")
        return save(run_id, "done", messages, None, n + 1, tokens, text)

    done, risky = [], []
    for b in resp.content:
        if b.type != "tool_use":
            continue
        blk = {"id": b.id, "name": b.name, "input": b.input}
        (risky if TOOLS[b.name]["risky"] else done).append(blk if b.name and TOOLS[b.name]["risky"] else run_tool(blk))
    if risky:                                        # pause: wait for a human decision
        return save(run_id, "waiting_approval", messages, {"done": done, "risky": risky}, n + 1, tokens)
    messages.append({"role": "user", "content": done})
    save(run_id, "running" if n + 1 < MAX_STEPS else "failed", messages, None, n + 1, tokens)

def worker():
    print("worker started")
    while True:
        row = db.execute("SELECT id FROM runs WHERE status='running' LIMIT 1").fetchone()
        if not row:
            time.sleep(0.5); continue
        try:
            step(row[0])
        except Exception as e:                       # e.g. API hiccup: leave as 'running', retry
            print("step failed, will retry:", e); time.sleep(3)

# ---- API ----
app = Flask(__name__)

@app.post("/runs")
def create():
    run_id = str(uuid.uuid4())
    task = (request.get_json() or {}).get("task", "")
    db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
               (run_id, "running", json.dumps([{"role": "user", "content": task}]), None, 0, 0, None))
    return jsonify(run_id=run_id), 202

@app.get("/runs/<run_id>")
def get(run_id):
    r = db.execute("SELECT status, step, tokens, result, pending FROM runs WHERE id=?", (run_id,)).fetchone()
    if not r: return jsonify(error="not found"), 404
    pend = json.loads(r[4])["risky"] if r[4] else None
    return jsonify(status=r[0], step=r[1], tokens=r[2], result=r[3], awaiting_approval=pend)

@app.post("/runs/<run_id>/decision")
def decide(run_id):
    approve = bool((request.get_json() or {}).get("approve"))
    msgs, pending, n, tokens = db.execute(
        "SELECT messages, pending, step, tokens FROM runs WHERE id=? AND status='waiting_approval'",
        (run_id,)).fetchone() or (None,) * 4
    if msgs is None: return jsonify(error="not waiting for approval"), 409
    pending, messages = json.loads(pending), json.loads(msgs)
    results = pending["done"]
    for blk in pending["risky"]:
        results.append(run_tool(blk) if approve else
                       {"type": "tool_result", "tool_use_id": blk["id"], "content": "DENIED by user"})
    messages.append({"role": "user", "content": results})
    save(run_id, "running", messages, None, n, tokens)
    return jsonify(ok=True)

if __name__ == "__main__":
    worker() if sys.argv[1:] == ["worker"] else app.run(port=5002)