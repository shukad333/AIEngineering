import pathlib, shutil, subprocess, statistics, sys
import fix_agent

TRIALS = 3
ROOT = pathlib.Path("bench_ws")

TASKS = {
  "negative_slice": dict(
    code="def last_n(xs, n):\n    return xs[-n+1:]\n",
    tests="from m import last_n\ndef test_a(): assert last_n([1,2,3,4], 2) == [3,4]\n",
    hidden="from m import last_n\ndef test_zero(): assert last_n([1,2,3], 0) == []\n"
           "def test_big(): assert last_n([1,2], 5) == [1,2]\n"),
  "mutable_default": dict(
    code="def add(item, bucket=[]):\n    bucket.append(item)\n    return bucket\n",
    tests="from m import add\ndef test_a(): assert add(1) == [1]\ndef test_b(): assert add(2) == [2]\n",
    hidden="from m import add\ndef test_c():\n    b = [9]\n    assert add(1, b) == [9, 1] and b == [9, 1]\n"),
  "leap_year": dict(
    code="def is_leap(y):\n    return y % 4 == 0 and y % 100 != 0\n",
    tests="from m import is_leap\ndef test_a():\n    assert is_leap(2024) and is_leap(2000) and not is_leap(1900)\n",
    hidden="from m import is_leap\ndef test_b():\n    assert is_leap(1600) and not is_leap(2100) and not is_leap(2023)\n"),
  "word_count": dict(
    code="def word_count(s):\n    return len(s.split(' '))\n",
    tests="from m import word_count\ndef test_a():\n    assert word_count('a b c') == 3 and word_count('a  b') == 2\n",
    hidden="from m import word_count\ndef test_b():\n    assert word_count('') == 0 and word_count('  hi  ') == 1 and word_count('a\\tb\\nc') == 3\n"),
  "merge_mutates": dict(
    code="def merge(a, b):\n    a.update(b)\n    return a\n",
    tests="from m import merge\ndef test_a():\n    a = {'x': 1}\n    assert merge(a, {'y': 2}) == {'x': 1, 'y': 2} and a == {'x': 1}\n",
    hidden="from m import merge\ndef test_b(): assert merge({'x': 1}, {'x': 2}) == {'x': 2}\n"),
}

def run_pytest(ws):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ws,
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0

def run_one(task):
    ws = ROOT / "ws"
    shutil.rmtree(ws, ignore_errors=True); ws.mkdir(parents=True)
    (ws / "m.py").write_text(task["code"]); (ws / "test_m.py").write_text(task["tests"])
    fix_agent.WS = ws.resolve()                          # point the agent at this workspace
    res = fix_agent.fix()
    tests_intact = (ws / "test_m.py").read_text() == task["tests"]
    (ws / "test_hidden.py").write_text(task["hidden"])   # only now does the agent's fix face them
    return {"visible": bool(res.get("success")) and tests_intact,
            "hidden": run_pytest(ws) and tests_intact,
            "steps": res.get("steps", 15), "tokens": res.get("tokens", 0)}

VARIANTS = {
    "baseline": {},
    "terse_prompt": {"SYSTEM": "Fix the failing tests with minimal edits. Never edit test files."},
    # "haiku": {"MODEL": "claude-haiku-5-5"},
}

def run_variant(name, overrides):
    orig = {k: getattr(fix_agent, k) for k in overrides}
    for k, v in overrides.items(): setattr(fix_agent, k, v)
    rows = []
    try:
        for tname, task in TASKS.items():
            for _ in range(TRIALS):
                r = run_one(task); r["task"] = tname; rows.append(r)
    finally:
        for k, v in orig.items(): setattr(fix_agent, k, v)
    n = len(rows)
    print(f"\n== {name} ==")
    for tname in TASKS:
        t = [r for r in rows if r["task"] == tname]
        print(f"{tname:16} visible {sum(r['visible'] for r in t)}/{len(t)}  "
              f"hidden {sum(r['hidden'] for r in t)}/{len(t)}  "
              f"steps {statistics.mean(r['steps'] for r in t):.1f}  "
              f"tokens {statistics.mean(r['tokens'] for r in t):.0f}")
    print(f"TOTAL visible {sum(r['visible'] for r in rows)}/{n}  hidden {sum(r['hidden'] for r in rows)}/{n}  "
          f"avg tokens {statistics.mean(r['tokens'] for r in rows):.0f}")

if __name__ == "__main__":
    for name, ov in VARIANTS.items():
        run_variant(name, ov)