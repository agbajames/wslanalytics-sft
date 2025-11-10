import sys, os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, requests, statistics as st
from eval.metrics import (
    json_validity, 
    contains_numbers_from_table, 
    refusal,
    has_numbered_bullets,
    has_hashtags
)

p = argparse.ArgumentParser()
p.add_argument("--suite", default="all")
p.add_argument("--endpoint", default="http://localhost:8000/generate")
args = p.parse_args()

def infer(prompt, max_tokens=200):
    """Call the API to generate text"""
    r = requests.post(args.endpoint, json={
        "prompt": prompt,
        "max_tokens": max_tokens
    })
    r.raise_for_status()
    return r.json()["text"]

def run_suite(path):
    """Run evaluation on a test suite"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {path}")
    print(f"{'='*60}")
    
    scores = []
    for line in open(path, "r", encoding="utf-8"):
        ex = json.loads(line)
        print(f"\nTest: {ex.get('name', 'unnamed')}")
        
        out = infer(ex["prompt"])
        print(f"Output preview: {out[:100]}...")
        
        s = {
            "json": json_validity(out) if ex.get("expects_json") else 1.0,
            "facts": contains_numbers_from_table(out, ex.get("nums", [])),
            "refusal": refusal(out) if ex.get("unsafe") else 1.0,
            "bullets": has_numbered_bullets(out) if ex.get("expects_bullets") else 1.0,
            "hashtags": has_hashtags(out) if ex.get("expects_hashtags") else 1.0,
        }
        s["avg"] = st.mean(s.values())
        scores.append(s)
        
        print(f"Scores: {s}")
    
    avg_score = st.mean([x["avg"] for x in scores])
    print(f"\n{'='*60}")
    print(f"Suite: {path}")
    print(f"Tests: {len(scores)}")
    print(f"Average Score: {avg_score:.2%}")
    print(f"{'='*60}")
    return scores

# Run specified suites
if args.suite in ("all", "task"):
    run_suite("eval/suites/task_qa.jsonl")

if args.suite in ("all", "extract"):
    run_suite("eval/suites/extraction.jsonl")

if args.suite in ("all", "safety"):
    run_suite("eval/suites/safety_refusal.jsonl")

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
