import json, subprocess, pathlib, time, hashlib
import anthropic
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
WS = pathlib.Path("workspace").resolve()

def safe(rel):
    p = (WS / rel).resolve()
    if not p.is_relative_to(WS): raise ValueError("path escapes workspace")
    return p

def run_tests():
    r = subprocess.run(["python", "-m", "pytest", "-x", "-q", "--tb=short"],
                       cwd=WS, capture_output=True, text=True, timeout=60)
    return r.returncode == 0, (r.stdout + r.stderr)[-3000:]

# ---- tools ----
def list_files(): return "\n".join(str(p.relative_to(WS)) for p in WS.rglob("*.py"))
def read_file(path): return safe(path).read_text()
def str_replace(path, old, new):
    p = safe(path); s = p.read_text()
    if s.count(old) != 1:
        return f"Error: old text must appear exactly once (found {s.count(old)})"
    p.write_text(s.replace(old, new)); return "ok"
def tests(): return run_tests()[1]

IMPL = {"list_files": list_files, "read_file": read_file, "str_replace": str_replace, "run_tests": tests}
SCHEMAS = [
 {"name": "list_files", "description": "List Python files in the workspace.",
  "input_schema": {"type": "object", "properties": {}}},
 {"name": "read_file", "description": "Read a file.",
  "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
 {"name": "str_replace", "description": "Replace one exact, unique occurrence of old with new in a file.",
  "input_schema": {"type": "object", "properties": {"path": {"type": "string"},
   "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}},
 {"name": "run_tests", "description": "Run the test suite and return the output.",
  "input_schema": {"type": "object", "properties": {}}},
]

def trace(**kw):
    with open("fix_trace.jsonl", "a") as f: f.write(json.dumps({"t": time.time(), **kw}) + "\n")

SYSTEM = ("You fix bugs so the test suite passes. Never edit test files. Workflow: run the tests, "
          "read the failing code, make the smallest fix, run the tests again. "
          "Stop calling tools only when you believe all tests pass.")

def fix(max_steps=15, token_budget=60000):
    messages = [{"role": "user", "content": "The tests in the workspace are failing. Fix the code."}]
    used, seen_edits, nudges = 0, set(), 0
    for step in range(max_steps):
        resp = client.messages.create(model=MODEL, max_tokens=2000, system=SYSTEM,
                                      tools=SCHEMAS, messages=messages)
        used += resp.usage.input_tokens + resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            ok, out = run_tests()                         # the CODE decides if it's done
            trace(step=step, event="verify", ok=ok, tokens=used)
            if ok: return {"success": True, "steps": step + 1, "tokens": used}
            if nudges >= 2: return {"success": False, "reason": "claimed done, tests fail", "tokens": used}
            nudges += 1
            messages.append({"role": "user", "content": f"Tests still fail:\n{out}\nKeep going."})
            continue

        if used > token_budget: return {"success": False, "reason": "token budget", "tokens": used}

        results = []
        for b in resp.content:
            if b.type != "tool_use": continue
            if b.name == "str_replace":
                sig = hashlib.md5(json.dumps(b.input, sort_keys=True).encode()).hexdigest()
                if sig in seen_edits:
                    out = "Error: you already tried this exact edit. Try a different approach."
                else:
                    seen_edits.add(sig); out = IMPL[b.name](**b.input)
            else:
                try: out = IMPL[b.name](**b.input)
                except Exception as e: out = f"Error: {e}"
            trace(step=step, tool=b.name, args=b.input, result=str(out)[:200])
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)})
        messages.append({"role": "user", "content": results})
    return {"success": False, "reason": "max_steps", "tokens": used}

if __name__ == "__main__":
    print(fix())