"""End-to-end agent: one customer message in, full decision bundle out."""
from src.intents import classify
from src.drafting import draft_reply
from src.escalation import decide
from src.retrieval import ThreadIndex

_index = None


def get_index():
    global _index
    if _index is None:
        _index = ThreadIndex()
    return _index


def handle_message(customer_message: str, thread_id: str = None, k_grounding: int = 3):
    intent, confidence = classify(customer_message)

    grounding = get_index().top_k(customer_message, k=k_grounding, exclude_thread_id=thread_id)

    should_escalate, reason = decide(customer_message, intent, confidence)

    if should_escalate:
        # We still draft a suggested reply for the human agent to review/
        # edit -- escalation means "needs a human in the loop", not "the
        # bot goes silent". See decision_log.md decision #7.
        reply = draft_reply(customer_message, grounding)
        mode = "escalate_with_suggested_reply"
    else:
        reply = draft_reply(customer_message, grounding)
        mode = "auto_handle"

    return {
        "customer_message": customer_message,
        "intent": intent,
        "confidence": confidence,
        "should_escalate": should_escalate,
        "escalation_reason": reason,
        "mode": mode,
        "draft_reply": reply,
        "grounding_used": [
            {"customer_message": g["customer_message"], "similarity": round(g["similarity"], 3)}
            for g in grounding
        ],
    }
