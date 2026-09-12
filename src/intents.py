"""Intent classification: fixed taxonomy, single LLM call, JSON out."""
import json
from src.config import INTENTS
from src.json_utils import parse_json_object
from src.llm_client import call_llm

SYSTEM_PROMPT = f"""You are an intent classifier for a customer support system.
Classify the customer's message into EXACTLY ONE of these intents:
{json.dumps(INTENTS)}

Rules:
- Pick the single best-fitting intent, even if the message could fit two.
- confidence is your calibrated belief (0.0-1.0) that this is the correct
  label, not just how confident you sound. If the message is ambiguous or
  off-taxonomy, give a LOW confidence score rather than forcing a
  high-confidence guess.
- Return ONLY a JSON object like {{"intent": "...", "confidence": 0.0}}.
  No other text.
"""


def classify(customer_message: str):
    user = f"Customer message:\n{customer_message}"
    raw = call_llm(SYSTEM_PROMPT, user, max_tokens=100)
    try:
        obj = parse_json_object(raw)
        intent = obj["intent"]
        conf = float(obj["confidence"])
        if intent not in INTENTS:
            return "UNKNOWN", 0.0
        return intent, conf
    except Exception:
        return "UNKNOWN", 0.0
