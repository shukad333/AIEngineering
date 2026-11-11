import json, time, pathlib
import anthropic
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
SANDBOX = pathlib.Path("sandbox").resolve()
SANDBOX.mkdir(exist_ok=True)

# ---- Tool registry: schema + function + risk level in one place ----
TOOLS = {}
def tool(name, description, properties, required, risky=False):
    def deco(fn):
        TOOLS[name] = {
            "fn": fn, "risky": risky,
            "schema": {"name": name, "description": description,
                       "input_schema": {"type": "object", "properties": properties,
                                        "required": required}},
        }
        return fn
    return deco

def safe_path(rel):
    p = (SANDBOX / rel).resolve()
    if not p.is_relative_to(SANDBOX):          # blocks ../../ escapes
        raise ValueError("path escapes sandbox")
    return p

@tool("list_files", "List files in the sandbox folder.", {}, [])
def list_files():
    return "\n".join(str(p.relative_to(SANDBOX)) for p in SANDBOX.rglob("*") if p.is_file()) or "(empty)"

@tool("read_file", "Read a text file from the sandbox.",
      {"path": {"type": "string"}}, ["path"])
def read_file(path):
    return safe_path(path).read_text(errors="ignore")[:8000]    # cap output size

@tool("write_file", "Create or overwrite a text file in the sandbox.",
      {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"], risky=True)
def write_file(path, content):
    p = safe_path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"

# ---- Trace: one JSON line per event, so every run is replayable ----
def trace(event, **kw):
    with open("trace.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "event": event, **kw}, ensure_ascii=False) + "\n")

def approve(name, args):
    print(f"\n⚠️  Agent wants to run {name}({json.dumps(args, ensure_ascii=False)[:300]})")
    return input("Allow? [y/N] ").strip().lower() == "y"

# ---- The loop ----
def run(task, max_steps=10, max_tokens_total=40000):
    messages = [{"role": "user", "content": task}]
    used = 0
    for step in range(max_steps):
        resp = client.messages.create(
            model=MODEL, max_tokens=2000, messages=messages,
            tools=[t["schema"] for t in TOOLS.values()],
            system="You are a careful file assistant. Use tools to inspect files before "
                   "changing anything. If a tool call is denied, do not retry it; explain instead.")
        used += resp.usage.input_tokens + resp.usage.output_tokens
        trace("model", step=step, stop=resp.stop_reason, tokens=used)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        if used > max_tokens_total:
            return "Stopped: token budget exceeded"

        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            spec = TOOLS.get(b.name)
            if spec is None:
                out = f"Error: unknown tool {b.name}"
            elif spec["risky"] and not approve(b.name, b.input):
                out = "DENIED by user"
            else:
                try:
                    out = spec["fn"](**b.input)
                except Exception as e:
                    out = f"Error: {e}"
            trace("tool", step=step, name=b.name, args=b.input, result=str(out)[:300])
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)})
        messages.append({"role": "user", "content": results})
    return "Stopped: hit max_steps"

if __name__ == "__main__":
    print(run("Create notes.md in the sandbox summarizing what files exist here."))