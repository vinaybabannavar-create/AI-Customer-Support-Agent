"""
Thin wrapper around Anthropic, Gemini, xAI/Grok, or Groq APIs (pick via
LLM_PROVIDER in .env). If the matching API key isn't set, falls back to a
deterministic mock mode so `README reproduce in 15 min` still works with
zero setup -- see decision_log.md decision #9 for why we think shipping a
mock mode is more honest than making the whole repo un-runnable without a
key. Gemini is included as a free-tier-friendly alternative to a paid
Anthropic key -- see README "Using Gemini instead of Anthropic". xAI/Grok
uses its OpenAI-compatible API endpoint.

RATE LIMITING: Gemini's free tier is strict (as low as 5 requests/minute
for some models). call_llm() below auto-retries on a 429 with the delay
the API itself asks for, so a full run will PAUSE repeatedly rather than
crash -- expect a full golden-set run on the free tier to take much
longer than 15 minutes (see README "Gemini free tier is slow").
"""
import json
import re
import time
from src.config import (
    ANTHROPIC_MODEL,
    GEMINI_MODEL,
    GROQ_MODEL,
    LLM_PROVIDER,
    USE_LLM,
    XAI_MODEL,
)

_client = None
MAX_RETRIES = 8
DEFAULT_BACKOFF_SECONDS = 20


def _get_anthropic_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _get_gemini_client():
    global _client
    if _client is None:
        from google import genai
        import os
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _get_xai_client():
    global _client
    if _client is None:
        from openai import OpenAI
        import os
        _client = OpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
            timeout=60.0,
        )
    return _client


def _get_groq_client():
    global _client
    if _client is None:
        from openai import OpenAI
        import os
        _client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            timeout=60.0,
        )
    return _client


def _extract_retry_seconds(exc) -> float:
    """Gemini's 429 error message contains 'Please retry in 20.2s' --
    pull that number out so we wait exactly as long as asked instead of
    guessing. Falls back to a fixed backoff if it can't find one."""
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    if match:
        return float(match.group(1)) + 1  # +1s safety margin
    return DEFAULT_BACKOFF_SECONDS


def _wants_json_response(system: str) -> bool:
    return "Return ONLY a JSON object" in system


def call_llm(system: str, user: str, max_tokens: int = 500) -> str:
    """Returns raw text response. Caller is responsible for parsing."""
    if not USE_LLM:
        return _mock_response(system, user)

    if LLM_PROVIDER == "gemini":
        return _call_gemini_with_retry(system, user, max_tokens)

    if LLM_PROVIDER == "xai":
        return _call_xai_with_retry(system, user, max_tokens)

    if LLM_PROVIDER == "groq":
        return _call_groq_with_retry(system, user, max_tokens)

    return _call_anthropic(system, user, max_tokens)


def _call_anthropic(system: str, user: str, max_tokens: int) -> str:
    client = _get_anthropic_client()
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _call_gemini_with_retry(system: str, user: str, max_tokens: int) -> str:
    from google.genai import errors as genai_errors
    import httpx

    client = _get_gemini_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user,
                config={
                    "system_instruction": system,
                    "max_output_tokens": max_tokens,
                },
            )
            return resp.text
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait = _extract_retry_seconds(e)
                print(f"  [rate limited, attempt {attempt}/{MAX_RETRIES}] "
                      f"waiting {wait:.0f}s before retrying...")
                time.sleep(wait)
                continue
            raise
        except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
            wait = DEFAULT_BACKOFF_SECONDS
            print(f"  [transient Gemini connection error, attempt "
                  f"{attempt}/{MAX_RETRIES}] waiting {wait:.0f}s before "
                  f"retrying... ({type(e).__name__})")
            time.sleep(wait)
            continue
    raise RuntimeError(
        f"Gave up after {MAX_RETRIES} rate-limit retries. "
        f"Your Gemini free-tier quota is likely too low for this run size -- "
        f"see README 'Gemini free tier is slow' for options."
    )


def _call_xai_with_retry(system: str, user: str, max_tokens: int) -> str:
    import openai

    client = _get_xai_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {}
            request_max_tokens = max_tokens
            if _wants_json_response(system):
                kwargs["response_format"] = {"type": "json_object"}
                kwargs["temperature"] = 0
                request_max_tokens = max(max_tokens, 500)
            resp = client.chat.completions.create(
                model=XAI_MODEL,
                max_tokens=request_max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except openai.RateLimitError as e:
            wait = _extract_retry_seconds(e)
            print(f"  [xAI rate limited, attempt {attempt}/{MAX_RETRIES}] "
                  f"waiting {wait:.0f}s before retrying...")
            time.sleep(wait)
            continue
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            wait = DEFAULT_BACKOFF_SECONDS
            print(f"  [transient xAI connection error, attempt "
                  f"{attempt}/{MAX_RETRIES}] waiting {wait:.0f}s before "
                  f"retrying... ({type(e).__name__})")
            time.sleep(wait)
            continue
    raise RuntimeError(
        f"Gave up after {MAX_RETRIES} xAI/Grok retries. "
        f"Your xAI quota or rate limit may be too low for this run size."
    )


def _call_groq_with_retry(system: str, user: str, max_tokens: int) -> str:
    import openai

    client = _get_groq_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {}
            request_max_tokens = max_tokens
            if _wants_json_response(system):
                kwargs["response_format"] = {"type": "json_object"}
                kwargs["temperature"] = 0
                request_max_tokens = max(max_tokens, 500)
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=request_max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except openai.RateLimitError as e:
            wait = _extract_retry_seconds(e)
            print(f"  [Groq rate limited, attempt {attempt}/{MAX_RETRIES}] "
                  f"waiting {wait:.0f}s before retrying...")
            time.sleep(wait)
            continue
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            wait = DEFAULT_BACKOFF_SECONDS
            print(f"  [transient Groq connection error, attempt "
                  f"{attempt}/{MAX_RETRIES}] waiting {wait:.0f}s before "
                  f"retrying... ({type(e).__name__})")
            time.sleep(wait)
            continue
    raise RuntimeError(
        f"Gave up after {MAX_RETRIES} Groq retries. "
        f"Your Groq quota or rate limit may be too low for this run size."
    )


# ---------------------------------------------------------------------------
# Mock mode: cheap keyword heuristics standing in for the LLM so the pipeline
# is fully runnable offline. This is intentionally simple -- it is NOT the
# thing being evaluated for "reply quality"; it exists purely so
# `python run_pipeline.py` produces *some* end-to-end output without a key.
# The report's headline numbers assume ANTHROPIC_API_KEY is set.
# ---------------------------------------------------------------------------
_KEYWORD_INTENT_MAP = [
    ("delivery_delay", ["still not here", "hasn't arrived", "late", "delivered but", "not delivered", "days now"]),
    ("order_status_inquiry", ["status of order", "has it shipped", "shipped yet", "on track"]),
    ("refund_or_return", ["refund", "return label", "returned item", "as described"]),
    ("damaged_or_wrong_item", ["smashed", "broken", "wrong item", "damaged", "empty when i opened"]),
    ("account_or_login_issue", ["log into my account", "password incorrect", "locked out", "2fa", "hacked"]),
    ("billing_or_charge_dispute", ["charged twice", "duplicate", "charge i don't recognize", "cancelled last month", "charged"]),
    ("general_feedback", ["thank you", "excellent", "helpful", "not happy with how support", "love the new"]),
]


def _mock_classify(text: str):
    low = text.lower()
    for intent, kws in _KEYWORD_INTENT_MAP:
        if any(kw in low for kw in kws):
            return intent, 0.72
    return "order_status_inquiry", 0.4


def _mock_response(system: str, user: str) -> str:
    """Very rough router: looks at the system prompt to decide which
    'skill' is being invoked (classify / draft / judge) and returns a
    plausible-shaped mock output (JSON where the real prompts ask for JSON)."""
    if "Return ONLY a JSON object" in system and "intent" in system and "confidence" in system:
        # classification call
        text = user.split("Customer message:")[-1].strip()
        intent, conf = _mock_classify(text)
        return json.dumps({"intent": intent, "confidence": conf})

    if "customer support reply" in system.lower():
        # drafting call
        return ("Thanks for reaching out -- sorry for the trouble! Could you "
                "please DM us your order number so we can look into this "
                "right away and get it fixed for you?")

    if "judge" in system.lower():
        return json.dumps({
            "grounded": 3, "correct_intent_handling": 3,
            "tone": 3, "actionable": 3, "overall": 3,
            "rationale": "mock-mode judge: heuristic mid-score, not a real quality signal"
        })

    return "{}"
