import json, argparse, random, pathlib

p = argparse.ArgumentParser()
p.add_argument("--in_jsonl", required=True)
p.add_argument("--train_out", required=True)
p.add_argument("--val_out", required=True)
p.add_argument("--ratio", type=float, default=0.9)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

random.seed(args.seed)
lines = [json.loads(l) for l in open(args.in_jsonl, encoding="utf-8")]
random.shuffle(lines)
n = int(len(lines) * args.ratio)

pathlib.Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)

open(args.train_out, "w", encoding="utf-8").write(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in lines[:n]) + "\n"
)
open(args.val_out, "w", encoding="utf-8").write(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in lines[n:]) + "\n"
)

print(f"Train: {n} examples")
print(f"Val: {len(lines) - n} examples")
