"""Generates outputs/calibration_drafts.jsonl -- the FIXED set of 30 drafts
used by eval/human_agreement.py, so the human-labelling and the judge-scoring
are both done against the exact same text (see human_agreement.py docstring)."""
import csv
import json
from src.agent import handle_message
from src.config import GOLDEN_SET

N = 30


def main():
    with open(GOLDEN_SET, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:N]

    with open("outputs/calibration_drafts.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            out = handle_message(r["customer_message"], thread_id=r["tweet_id"])
            f.write(json.dumps({
                "tweet_id": r["tweet_id"],
                "customer_message": r["customer_message"],
                "draft_reply": out["draft_reply"],
                "grounding": out["grounding_used"],
            }) + "\n")
    print(f"Wrote {N} calibration drafts to outputs/calibration_drafts.jsonl")


if __name__ == "__main__":
    main()
