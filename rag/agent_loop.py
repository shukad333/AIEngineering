import json
import anthropic
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"

# 1. Tool definitions: the model sees only these schemas
TOOLS = [
    {
        "name": "search_docs",
        "description": "Search the local documentation and return the most relevant passages. "
                       "Use for any question about the contents of the docs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "k": {"type": "integer", "description": "Number of passages", "default": 4},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression, e.g. '(1500 - 150) * 4'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

# 2. Tool implementations: plain Python
def search_docs(query, k=4):
    from doc_rag import embed, top_k, chunks, vecs, meta   # reuse your pipeline
    q_vec = embed(["Represent this sentence for searching relevant passages: " + query])[0]
    idx = top_k(q_vec, vecs, k)
    return "\n\n".join(f"[{meta[i]['file']}#{meta[i]['chunk']}] {chunks[i]}" for i in idx)

def calculate(expression):
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: unsupported characters"
    return str(eval(expression, {"__builtins__": {}}))

IMPLS = {"search_docs": search_docs, "calculate": calculate}

# 3. The loop
def run_agent(question, max_steps=8):
    messages = [{"role": "user", "content": question}]
    tools_used, tokens = [], 0
    for step in range(max_steps):
        resp = client.messages.create(model=MODEL, max_tokens=2000, tools=TOOLS,
                                      messages=messages, system="...same as before...")
        tokens += resp.usage.input_tokens + resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"answer": text, "tools": tools_used, "steps": step + 1, "tokens": tokens}
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                tools_used.append(b.name)
                try:    out = IMPLS[b.name](**b.input)
                except Exception as e: out = f"Error: {e}"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})
    return {"answer": "", "tools": tools_used, "steps": max_steps, "tokens": tokens}

if __name__ == "__main__":
    print(run_agent("XXX"))