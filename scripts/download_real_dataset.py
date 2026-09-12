"""
Optional: pull the REAL Kaggle "Customer Support on Twitter" dataset
(thoughtvector/customer-support-on-twitter) to replace the bundled sample
at data/raw/sample_twcs.csv.

Requires a Kaggle account + API token (~/.kaggle/kaggle.json) and the
`kaggle` package. Not run automatically by run_pipeline.py because it
needs your own credentials -- see README "Using the real dataset".

Usage:
    pip install kaggle
    kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip
    # then filter to one brand's rows, e.g.:
    python scripts/download_real_dataset.py --brand AmazonHelp --out data/raw/real_twcs_filtered.csv

Everything downstream (src/data_prep.py, src/retrieval.py, eval/*) reads
whatever CSV path you point it at (same twcs.csv column schema), so
swapping in the real file is a one-flag change, not a code change.
"""
import argparse
import csv


def filter_to_brand(in_path, out_path, brand_author_id):
    with open(in_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    by_id = {r["tweet_id"]: r for r in rows}
    keep_ids = set()
    for r in rows:
        if r["author_id"] == brand_author_id:
            keep_ids.add(r["tweet_id"])
            if r.get("in_response_to_tweet_id"):
                keep_ids.add(r["in_response_to_tweet_id"])
        if r["inbound"] == "True" and r.get("response_tweet_id"):
            resp = by_id.get(r["response_tweet_id"])
            if resp and resp["author_id"] == brand_author_id:
                keep_ids.add(r["tweet_id"])
                keep_ids.add(resp["tweet_id"])

    filtered = [r for r in rows if r["tweet_id"] in keep_ids]
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        w.writeheader()
        w.writerows(filtered)
    print(f"Filtered {len(rows)} -> {len(filtered)} rows for brand={brand_author_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/raw/twcs.csv")
    ap.add_argument("--out", default="data/raw/real_twcs_filtered.csv")
    ap.add_argument("--brand", default="AmazonHelp")
    args = ap.parse_args()
    filter_to_brand(args.in_path, args.out, args.brand)
