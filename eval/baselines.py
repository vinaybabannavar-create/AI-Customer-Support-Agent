"""
Two baselines the report compares the agent against:

1. TRIVIAL: always predict the single most common intent in the golden set,
   and NEVER escalate. This is the "what if we did nothing smart" floor.
2. SIMPLE: a hand-written keyword/regex rule-based classifier (no LLM),
   with a fixed escalation rule (only the hard-trigger-keyword check).
   This is "what a junior engineer could ship in an afternoon without ML".

Both exist so the agent's headline numbers have something to beat --
see decision_log.md decision #8 and report section "Results vs baselines".
"""
from collections import Counter
from src.escalation import HARD_ESCALATION_KEYWORDS


def trivial_baseline(golden_rows):
    majority_intent = Counter(r["golden_intent"] for r in golden_rows).most_common(1)[0][0]
    preds = []
    for r in golden_rows:
        preds.append({
            "tweet_id": r["tweet_id"],
            "pred_intent": majority_intent,
            "pred_should_escalate": False,
        })
    return preds


_SIMPLE_KEYWORD_MAP = [
    ("delivery_delay", ["still not here", "hasn't arrived", "not delivered", "days now", "package for"]),
    ("order_status_inquiry", ["status of order", "shipped yet", "on track"]),
    ("refund_or_return", ["refund", "return label", "returned item"]),
    ("damaged_or_wrong_item", ["smashed", "broken", "wrong item", "damaged", "empty when"]),
    ("account_or_login_issue", ["log into", "password", "locked out", "2fa", "hacked"]),
    ("billing_or_charge_dispute", ["charged twice", "duplicate", "don't recognize", "cancelled last month", "charged"]),
    ("general_feedback", ["thank you", "excellent", "helpful", "love the new", "not happy with how support"]),
]


def simple_baseline(golden_rows):
    preds = []
    for r in golden_rows:
        text = r["customer_message"].lower()
        pred_intent = "order_status_inquiry"  # fallback default
        for intent, kws in _SIMPLE_KEYWORD_MAP:
            if any(kw in text for kw in kws):
                pred_intent = intent
                break
        should_escalate = any(kw in text for kw in HARD_ESCALATION_KEYWORDS)
        preds.append({
            "tweet_id": r["tweet_id"],
            "pred_intent": pred_intent,
            "pred_should_escalate": should_escalate,
        })
    return preds
