"""
Calibrates the LLM-judge against human ratings on a fixed subset.

Process (see reports/REPORT.md "how well does your judge agree with a
human" section for the write-up):
1. We took the FIRST 30 rows of the golden set (fixed, not random --
   reproducible) and ran the live agent to produce (draft_reply, grounding).
2. One of us hand-scored each (customer_message, draft_reply) pair on the
   same 1-5 "overall" rubric the judge uses, blind to the judge's score.
   Those scores are in data/golden/human_calibration_labels.csv.
3. This script re-runs the judge on the SAME saved drafts (not fresh drafts
   -- otherwise a non-deterministic LLM draft would make the comparison
   apples-to-oranges) and reports: Pearson correlation, exact-match rate,
   and within-1-point agreement rate.

Honesty check we make readers do: with only 30 points and one human rater,
this is a calibration SIGNAL, not a validated judge -- see REPORT.md
"what's misleading about my headline number".
"""
import csv
import json
import statistics
from eval.judge import judge_reply

CALIBRATION_DRAFTS = "outputs/calibration_drafts.jsonl"
HUMAN_LABELS = "data/golden/human_calibration_labels.csv"
OUT = "outputs/judge_human_agreement.json"


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def main():
    drafts = {}
    with open(CALIBRATION_DRAFTS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            drafts[row["tweet_id"]] = row

    human = {}
    with open(HUMAN_LABELS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human[row["tweet_id"]] = int(row["human_overall"])

    judge_scores, human_scores, records = [], [], []
    for tid, d in drafts.items():
        if tid not in human:
            continue
        j = judge_reply(d["customer_message"], d["draft_reply"], d["grounding"])
        judge_overall = int(j.get("overall", 0) or 0)
        h = human[tid]
        judge_scores.append(judge_overall)
        human_scores.append(h)
        records.append({"tweet_id": tid, "judge_overall": judge_overall, "human_overall": h})

    n = len(records)
    if n == 0:
        raise RuntimeError(
            "No overlapping calibration records found. Run "
            "`py -m eval.make_calibration_drafts` successfully first."
        )

    exact = sum(1 for r in records if r["judge_overall"] == r["human_overall"]) / n
    within1 = sum(1 for r in records if abs(r["judge_overall"] - r["human_overall"]) <= 1) / n
    r = pearson(judge_scores, human_scores)

    result = {
        "n": n,
        "exact_match_rate": round(exact, 3),
        "within_1_point_rate": round(within1, 3),
        "pearson_r": round(r, 3) if r is not None else None,
        "records": records,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"n={n} exact_match={exact:.2f} within_1={within1:.2f} pearson_r={r}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
