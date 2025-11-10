import json, argparse, pathlib

def clean(text: str) -> str:
    """Remove carriage returns and normalize whitespace"""
    text = text.replace("\r", "")
    text = " ".join(text.split())
    return text.strip()

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", required=True)
p.add_argument("--out_jsonl", default="data/interim/all_posts.jsonl")
args = p.parse_args()

pathlib.Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

seen = set()
with open(args.out_jsonl, "w", encoding="utf-8") as out:
    for fp in args.inputs:
        for line in open(fp, "r", encoding="utf-8"):
            d = json.loads(line)
            key = (d.get("source"), d.get("id"))
            if key in seen:
                continue
            seen.add(key)
            body = clean(d.get("body", ""))
            title = clean(d.get("title", ""))
            if not body or len(body) < 60:
                continue
            d["body"] = body
            d["title"] = title
            out.write(json.dumps(d, ensure_ascii=False) + "\n")

print("Wrote", args.out_jsonl)
