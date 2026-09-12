"""
Loads the raw twcs.csv (real Kaggle file OR our bundled sample -- same
schema, this code does not care which) and reconstructs
(customer_message -> brand_reply) pairs for a single brand, writing a
clean JSONL that the rest of the pipeline reads.

Real dataset column reference (thoughtvector/customer-support-on-twitter):
tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id
"""
import csv
import json
import re
import sys
from src.config import RAW_CSV, PROCESSED_THREADS, BRAND_AUTHOR_ID


def clean_text(text: str) -> str:
    """Light cleanup: strip @mentions used only for routing, collapse
    whitespace. We deliberately KEEP hashtags/emoji/typos -- see
    decision_log.md decision #1 (don't over-clean noisy real-world text,
    the model needs to handle it as-is in production)."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_threads(rows, brand_author_id=BRAND_AUTHOR_ID):
    by_id = {r["tweet_id"]: r for r in rows}
    threads = []
    for r in rows:
        if r["inbound"] != "True":
            continue  # only start from customer-initiated tweets
        resp_id = r.get("response_tweet_id", "")
        if not resp_id:
            continue
        brand_reply = by_id.get(resp_id)
        if not brand_reply or brand_reply["author_id"] != brand_author_id:
            continue
        threads.append({
            "thread_id": r["tweet_id"],
            "customer_author_id": r["author_id"],
            "customer_message": clean_text(r["text"]),
            "customer_created_at": r["created_at"],
            "brand_reply": clean_text(brand_reply["text"]),
            "brand_reply_created_at": brand_reply["created_at"],
        })
    return threads


def main():
    src_path = sys.argv[1] if len(sys.argv) > 1 else RAW_CSV
    rows = load_rows(src_path)
    threads = build_threads(rows)
    with open(PROCESSED_THREADS, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")
    print(f"Loaded {len(rows)} raw tweets -> {len(threads)} resolved "
          f"customer->brand threads for {BRAND_AUTHOR_ID}")
    print(f"Wrote {PROCESSED_THREADS}")


if __name__ == "__main__":
    main()
