"""
Interactive relabeling tool: walks through outputs/calibration_drafts.jsonl
(the CURRENT, real LLM-generated drafts) and lets you assign a fresh
human_overall score (1-5) to each one, overwriting
data/golden/human_calibration_labels.csv.

Why this exists: the original human_calibration_labels.csv was hand-scored
against old mock-mode placeholder drafts, before a real API key was wired
up. Now that outputs/calibration_drafts.jsonl contains real, per-message
LLM drafts, the old human scores are scoring different text than what the
judge sees -- this regenerates them against the SAME text the judge scores,
which also fixes the golden-set/calibration tweet_id mismatch as a side
effect (every row here comes from the current calibration_drafts.jsonl,
so every tweet_id will match).

Usage:
    python -m eval.label_calibration_interactive

For each example you'll see the customer message and the drafted reply,
then type a score 1-5 (or 's' to skip, 'q' to quit and save what you have
so far).
"""
import csv
import json
import os

DRAFTS = "outputs/calibration_drafts.jsonl"
OUT = "data/golden/human_calibration_labels.csv"


def main():
    if not os.path.exists(DRAFTS):
        print(f"{DRAFTS} not found -- run `python -m eval.make_calibration_drafts` first.")
        return

    rows = []
    with open(DRAFTS, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Labelling {len(rows)} examples. For each: read the customer "
          f"message and the drafted reply, then score the REPLY overall, "
          f"1 (poor) to 5 (excellent) -- would you be comfortable sending "
          f"this reply to a real customer as-is?\n")

    results = []
    for i, r in enumerate(rows, 1):
        print(f"\n--- {i}/{len(rows)} (tweet_id={r['tweet_id']}) ---")
        print(f"Customer: {r['customer_message']}")
        print(f"Drafted reply: {r['draft_reply']}")
        while True:
            ans = input("Score 1-5 (or 's' skip, 'q' quit+save): ").strip().lower()
            if ans == "q":
                print("Stopping early, saving what's labelled so far.")
                _write(results)
                return
            if ans == "s":
                break
            if ans in {"1", "2", "3", "4", "5"}:
                note = input("Optional one-line note (why this score): ").strip()
                results.append({
                    "tweet_id": r["tweet_id"],
                    "human_overall": ans,
                    "human_notes": note,
                })
                break
            print("Please enter 1-5, 's', or 'q'.")

    _write(results)


def _write(results):
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tweet_id", "human_overall", "human_notes"])
        for r in results:
            w.writerow([r["tweet_id"], r["human_overall"], r["human_notes"]])
    print(f"\nWrote {len(results)} fresh human labels to {OUT}")


if __name__ == "__main__":
    main()