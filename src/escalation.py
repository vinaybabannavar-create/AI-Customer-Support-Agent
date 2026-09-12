"""
Escalation decision: deliberately a deterministic policy function, NOT an
LLM call. See decision_log.md decision #4/#5 -- for a support agent, the
auto-handle/escalate boundary is a business/trust decision, and we want
it auditable and testable, not subject to LLM prompt drift.
"""
from src.config import (
    AUTO_HANDLE_ELIGIBLE_INTENTS,
    CONFIDENCE_ESCALATION_THRESHOLD,
    HARD_ESCALATION_KEYWORDS,
)


def decide(customer_message: str, intent: str, confidence: float):
    """Returns (should_escalate: bool, reason: str)."""
    low = customer_message.lower()

    for kw in HARD_ESCALATION_KEYWORDS:
        if kw in low:
            return True, f"hard escalation trigger phrase matched: '{kw}'"

    if intent == "UNKNOWN":
        return True, "intent classifier could not confidently assign a taxonomy label"

    if intent not in AUTO_HANDLE_ELIGIBLE_INTENTS:
        return True, f"intent '{intent}' is policy-excluded from auto-handling (needs judgment/refund authority/account access a bot shouldn't have)"

    if confidence < CONFIDENCE_ESCALATION_THRESHOLD:
        return True, f"classifier confidence {confidence:.2f} below threshold {CONFIDENCE_ESCALATION_THRESHOLD}"

    return False, f"intent '{intent}' is auto-handle eligible and confidence {confidence:.2f} clears threshold"
