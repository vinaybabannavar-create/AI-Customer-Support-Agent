"""
Full evaluation harness. Run: python -m eval.run_eval

Produces outputs/eval_results.json + outputs/eval_summary.md covering:
  - intent accuracy (agent vs trivial baseline vs simple baseline)
  - per-intent precision/recall/F1 for the agent
  - escalation decision precision/recall/F1 (positive class = "should escalate")
  - reply-quality judge scores (mean per rubric dimension) on a subsample
"""
import csv
import json
from collections import defaultdict

from src.agent import handle_message
from src.config import GOLDEN_SET, INTENTS
from eval.baselines import trivial_baseline, simple_baseline
from eval.judge import judge_reply

JUDGE_SAMPLE_SIZE = 10  # judging every example costs $ + time; a subsample is standard practice


def load_golden():
    with open(GOLDEN_SET, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 3), round(r, 3), round(f1, 3)


def intent_accuracy(golden_rows, preds_by_id, pred_key="pred_intent"):
    correct = 0
    per_intent = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in golden_rows:
        gold = r["golden_intent"]
        pred = preds_by_id[r["tweet_id"]][pred_key]
        if pred == gold:
            correct += 1
            per_intent[gold]["tp"] += 1
        else:
            per_intent[gold]["fn"] += 1
            per_intent[pred]["fp"] += 1
    acc = correct / len(golden_rows)
    breakdown = {}
    for intent in INTENTS:
        c = per_intent[intent]
        p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
        breakdown[intent] = {"precision": p, "recall": r, "f1": f1, "support": c["tp"] + c["fn"]}
    return round(acc, 3), breakdown


def escalation_metrics(golden_rows, preds_by_id, pred_key="pred_should_escalate"):
    tp = fp = fn = tn = 0
    for r in golden_rows:
        gold = r["golden_should_escalate"] == "True"
        pred = bool(preds_by_id[r["tweet_id"]][pred_key])
        if gold and pred:
            tp += 1
        elif not gold and pred:
            fp += 1
        elif gold and not pred:
            fn += 1
        else:
            tn += 1
    p, r, f1 = prf1(tp, fp, fn)
    acc = (tp + tn) / len(golden_rows)
    return {"precision": p, "recall": r, "f1": f1, "accuracy": round(acc, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_agent_on_golden(golden_rows):
    preds = {}
    for r in golden_rows:
        out = handle_message(r["customer_message"], thread_id=r["tweet_id"])
        preds[r["tweet_id"]] = {
            "pred_intent": out["intent"],
            "pred_should_escalate": out["should_escalate"],
            "escalation_reason": out["escalation_reason"],
            "draft_reply": out["draft_reply"],
            "grounding_used": out["grounding_used"],
            "confidence": out["confidence"],
        }
    return preds


def run_judge_subsample(golden_rows, agent_preds, n=JUDGE_SAMPLE_SIZE):
    sample = golden_rows[:n]
    scores = defaultdict(list)
    for r in sample:
        pred = agent_preds[r["tweet_id"]]
        j = judge_reply(r["customer_message"], pred["draft_reply"], pred["grounding_used"])
        for k in ("grounded", "correct_intent_handling", "tone", "actionable", "overall"):
            v = j.get(k)
            if isinstance(v, (int, float)):
                scores[k].append(v)
    means = {k: round(sum(v) / len(v), 2) for k, v in scores.items() if v}
    return means, len(sample)


def main():
    golden_rows = load_golden()

    agent_preds = run_agent_on_golden(golden_rows)
    trivial_preds = {p["tweet_id"]: p for p in trivial_baseline(golden_rows)}
    simple_preds = {p["tweet_id"]: p for p in simple_baseline(golden_rows)}

    agent_acc, agent_breakdown = intent_accuracy(golden_rows, agent_preds)
    trivial_acc, _ = intent_accuracy(golden_rows, trivial_preds)
    simple_acc, _ = intent_accuracy(golden_rows, simple_preds)

    agent_esc = escalation_metrics(golden_rows, agent_preds)
    trivial_esc = escalation_metrics(golden_rows, trivial_preds)
    simple_esc = escalation_metrics(golden_rows, simple_preds)

    judge_means, judge_n = run_judge_subsample(golden_rows, agent_preds)

    results = {
        "n_golden": len(golden_rows),
        "intent_accuracy": {"agent": agent_acc, "trivial_baseline": trivial_acc, "simple_baseline": simple_acc},
        "agent_intent_breakdown": agent_breakdown,
        "escalation_metrics": {"agent": agent_esc, "trivial_baseline": trivial_esc, "simple_baseline": simple_esc},
        "reply_quality_judge_means": judge_means,
        "reply_quality_judge_n": judge_n,
    }

    with open("outputs/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("outputs/eval_summary.md", "w", encoding="utf-8") as f:
        f.write("# Eval Summary\n\n")
        f.write(f"Golden set size: {results['n_golden']}\n\n")
        f.write("## Intent accuracy\n\n")
        f.write("| Model | Accuracy |\n|---|---|\n")
        for k, v in results["intent_accuracy"].items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## Per-intent breakdown (agent)\n\n")
        f.write("| Intent | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n")
        for intent, m in agent_breakdown.items():
            f.write(f"| {intent} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |\n")
        f.write("\n## Escalation decision metrics\n\n")
        f.write("| Model | Precision | Recall | F1 | Accuracy |\n|---|---|---|---|---|\n")
        for k, m in results["escalation_metrics"].items():
            f.write(f"| {k} | {m['precision']} | {m['recall']} | {m['f1']} | {m['accuracy']} |\n")
        f.write(f"\n## Reply quality (LLM judge, n={judge_n})\n\n")
        f.write("| Dimension | Mean (1-5) |\n|---|---|\n")
        for k, v in judge_means.items():
            f.write(f"| {k} | {v} |\n")

    print(json.dumps(results["intent_accuracy"], indent=2))
    print(json.dumps(results["escalation_metrics"], indent=2))
    print("Judge means:", judge_means)
    print("Wrote outputs/eval_results.json and outputs/eval_summary.md")


if __name__ == "__main__":
    main()
