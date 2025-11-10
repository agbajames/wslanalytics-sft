import json, argparse, statistics as st

p = argparse.ArgumentParser()
p.add_argument("--jsonl", required=True)
args = p.parse_args()

L = []
kinds = {}
for line in open(args.jsonl, encoding="utf-8"):
    d = json.loads(line)
    L.append(len(d.get("output", "")))
    k = d.get("meta", {}).get("type", "?")
    kinds[k] = kinds.get(k, 0) + 1

print("="*60)
print("DATA QUALITY REPORT")
print("="*60)
print(f"Examples: {len(L)}")
print(f"Length mean: {round(st.mean(L), 1) if L else 0}")
print(f"Length median: {st.median(L) if L else 0}")
print(f"Length min: {min(L) if L else 0}")
print(f"Length max: {max(L) if L else 0}")
print("="*60)
print("By type:")
for k, v in sorted(kinds.items()):
    print(f"  {k}: {v} examples")
print("="*60)
