# scripts/build_sft_pairs.py
# WSLAnalytics-styled instruction builder (British English, numbered bullets, light emoji, hashtags)
import json, argparse, pathlib, re

# --- Style anchors (tune freely) -------------------------------------------------
EMOJI_LIGHT = ["⚙️","🧱","🎯","📊","🔥","🧠","🧮","��","🔁","🔧","⏱️","🧭"]
HASH_CORE   = ["#WSL", "#WSLAnalytics"]
SYSTEM_CUE = (
    "You are WSLAnalytics: a British-English football analyst. "
    "Write concise, data-led analysis with light emoji and appropriate hashtags. "
    "Prefer numbered bullets. Use British English spellings. Avoid hype and clichés. "
    "Never invent statistics—only reference numbers explicitly present in the context."
)

# Formatting rules applied to all tasks
COMMON_RULES = [
    "Do not hallucinate stats; only cite figures present in <CONTEXT>.",
    "Keep sentences tight (12–22 words). Avoid filler.",
    "Use British English (defence, centre, organisation, pressing 'intensity', etc.).",
    "Use at most one emoji per bullet and not every bullet needs one.",
    "No more than 2 hashtags total unless explicitly told.",
    "Never include links.",
]

# Task-specific templates
PREVIEW_TEMPLATE = {
    "title": "PREVIEW THREAD (5–7 bullets + verdict)",
    "bullets": [
        "Attacking process: chance creation route(s), xG trend, shot quality/volume.",
        "Defensive stability: GA/xGA trend, set-piece record, pressing/PPDA notes.",
        "Transitions/press: where the match tilts (press traps, rest defence).",
        "Key player(s): 1–2 roles tied to numbers in context.",
        "Game-state risks: early goal effects, late-game subs pattern.",
        "Set-pieces: corners, wide free-kicks, delivery zones.",
        "Intangibles: travel, fixture congestion, injuries (if present in context)."
    ],
    "verdict": "Short 'Data Verdict' (one sentence) with a grounded lean (not a guarantee)."
}

RECAP_TEMPLATE = {
    "title": "POST-MATCH RECAP (5–7 bullets + Data Verdict)",
    "bullets": [
        "Attacking process summary: shot quality vs. volume; key actions.",
        "Defensive phases: structure, last-line actions, GA vs. xGA notes.",
        "Momentum pivots: pressing waves, substitutions, injuries (if in context).",
        "Key player(s): on-ball value or duel wins, tied to context stats.",
        "Boxes & set-pieces: entries, second balls, corners/FKs.",
        "Game management: lead protection or chase behaviour.",
        "Comparative note vs. prior matchweek (only if in context)."
    ],
    "verdict": "One-line 'Data Verdict' that reconciles result vs. underlying numbers."
}

THREAD_TEMPLATE = {
    "title": "WEEKLY/GENERAL THREAD (6 bullets + closer)",
    "bullets": [
        "Team #1 micro-trend (form/xG/GA).",
        "Team #2 micro-trend.",
        "League-wide or positional theme.",
        "Standout player note grounded in numbers.",
        "Risk or regression flag.",
        "Actionable takeaway.",
    ],
    "verdict": "Short closer (1 sentence)."
}

CAPTION_TEMPLATE = {
    "title": "SINGLE-PARAGRAPH SOCIAL CAPTION",
    "bullets": [],
    "verdict": "One paragraph, 2–3 sentences, 1–2 hashtags."
}

# --- Helpers ---------------------------------------------------------------------
def classify(title: str, body: str) -> str:
    t = (title + " " + body[:200]).lower()
    if "preview" in t or "pre-match" in t or "pre-match" in t: return "preview"
    if "post" in t or "recap" in t or "full-time" in t or "full time" in t: return "recap"
    if "thread" in t or "1️⃣" in t or "weekly" in t: return "thread"
    return "caption"

def extract_hashtags(text: str):
    tags = set(re.findall(r"(#\w+)", text))
    # Keep common WSL tags near the front later
    ordered = [t for t in HASH_CORE if t in tags] + [t for t in tags if t not in HASH_CORE]
    # cap to 2 by default; can be changed per template if needed
    return ordered[:2] if ordered else HASH_CORE[:1]

def bullet_style_guidance(kind: str) -> str:
    if kind == "preview":
        count = "5–7"
        tpl = PREVIEW_TEMPLATE
    elif kind == "recap":
        count = "5–7"
        tpl = RECAP_TEMPLATE
    elif kind == "thread":
        count = "6"
        tpl = THREAD_TEMPLATE
    else:
        count = "0"
        tpl = CAPTION_TEMPLATE
    bullets = "\n".join([f"- {b}" for b in tpl["bullets"]]) if tpl["bullets"] else "- (No bullets for captions)"
    rules = "\n".join([f"- {r}" for r in COMMON_RULES])
    return (
        f"### STYLE & STRUCTURE\n"
        f"- Numbered bullets: {count} (use 1️⃣ 2️⃣ 3️⃣ ... or 1. 2. 3.)\n"
        f"- Light emoji allowed {EMOJI_LIGHT[:6]} (max one per bullet).\n"
        f"- Finish with: {tpl['verdict']}\n"
        f"\n### CONTENT HINTS\n{bullets}\n"
        f"\n### RULES\n{rules}\n"
    )

def make_instruction(kind: str, title: str, body: str):
    tags = " ".join(extract_hashtags(title + " " + body))
    style = bullet_style_guidance(kind)
    task = "Draft a pre-match preview" if kind=="preview" else \
           "Write a post-match recap" if kind=="recap" else \
           "Turn the context into a weekly thread" if kind=="thread" else \
           "Write a concise social caption"
    return (
        f"<s>{SYSTEM_CUE}</s>\n"
        f"<TITLE>{title}</TITLE>\n"
        f"<CONTEXT>\n{body}\n</CONTEXT>\n\n"
        f"### TASK\n{task} in WSLAnalytics house style.\n"
        f"{style}\n"
        f"### HASHTAGS\nInclude subtly at the end: {tags}\n"
        f"### OUTPUT FORMAT\n"
        f"- Use numbered bullets for previews/recaps/threads.\n"
        f"- Keep each bullet to one or two sentences.\n"
        f"- End with the required verdict/closer line.\n"
    )

# --- Main ------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--in_jsonl", default="data/interim/all_posts.jsonl")
p.add_argument("--out_jsonl", default="data/processed/sft_all.jsonl")
args = p.parse_args()

pathlib.Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

w = open(args.out_jsonl, "w", encoding="utf-8")
for line in open(args.in_jsonl, "r", encoding="utf-8"):
    d = json.loads(line)
    title, body = d["title"], d["body"]
    kind = classify(title, body)
    instruction = make_instruction(kind, title, body)
    # Supervision target: your original text (style learning)
    ex = {
        "instruction": instruction,
        "output": body,
        "meta": {"type": kind, "title": title, "hashtags": extract_hashtags(title + " " + body)}
    }
    w.write(json.dumps(ex, ensure_ascii=False) + "\n")

w.close()
print("Wrote", args.out_jsonl)
