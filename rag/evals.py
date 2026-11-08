import json
from agent_loop import run_agent, client, MODEL

CASES = [
    {"q": "What is (1500-150)*4?",
     "must_contain": ["5400"], "expect_tools": ["calculate"]},
    {"q": "How does leader election work?",
     "must_contain": ["leader"], "expect_tools": ["search_docs"]},
    {"q": "What is the capital of Mars?",          # should refuse / say not found
     "must_contain": [], "expect_tools": [], "judge": "Answer must not invent facts."},
    # add 10-15 more from your own docs
]

def judge(question, answer, rubric):
    resp = client.messages.create(
        model=MODEL, max_tokens=500,
        messages=[{"role": "user", "content":
            f"Question: {question}\nAnswer: {answer}\nRubric: {rubric}\n"
            'Reply with only JSON: {"pass": true|false, "reason": "..."}'}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text.strip().strip("`").removeprefix("json").strip())

def run_evals():
    passed = 0
    for c in CASES:
        r = run_agent(c["q"])
        ok = all(s.lower() in r["answer"].lower() for s in c["must_contain"])
        ok &= all(t in r["tools"] for t in c["expect_tools"])
        verdict = ""
        if "judge" in c:
            j = judge(c["q"], r["answer"], c["judge"])
            ok &= j["pass"]; verdict = j["reason"]
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} | steps={r['steps']} tokens={r['tokens']} "
              f"tools={r['tools']} | {c['q']} {verdict}")
    print(f"\n{passed}/{len(CASES)} passed")

if __name__ == "__main__":
    run_evals()