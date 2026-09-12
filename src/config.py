"""Central config: paths, brand, intent taxonomy, escalation policy knobs."""
import os


def _load_dotenv(path: str = ".env"):
    """Load simple KEY=VALUE lines without requiring an extra dependency."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

BRAND = "AmazonHelp"
BRAND_AUTHOR_ID = "AmazonHelp"

RAW_CSV = "data/raw/sample_twcs.csv"
PROCESSED_THREADS = "data/processed/threads.jsonl"
GOLDEN_SET = "data/golden/golden_eval.csv"

INTENTS = [
    "delivery_delay",
    "order_status_inquiry",
    "refund_or_return",
    "damaged_or_wrong_item",
    "account_or_login_issue",
    "billing_or_charge_dispute",
    "general_feedback",
]

# Keywords that, regardless of predicted intent, force escalation.
# See reports/decision_log.md decision #5 for why this is a hard override
# rather than something the LLM decides on its own.
HARD_ESCALATION_KEYWORDS = [
    "lawyer", "lawsuit", "sue", "legal action",
    "fraud", "dispute with my bank", "chargeback",
    "real human", "speak to a manager", "get me a manager",
    "cancel everything", "never order from you again",
]

# Intents the agent is allowed to auto-handle at all (others always route
# to a human regardless of confidence -- see decision_log.md decision #4).
AUTO_HANDLE_ELIGIBLE_INTENTS = {
    "delivery_delay",
    "order_status_inquiry",
    "general_feedback",
}

# Below this confidence, escalate even for an eligible intent.
CONFIDENCE_ESCALATION_THRESHOLD = 0.65

# LLM settings
# LLM_PROVIDER picks which real API to call: "anthropic", "gemini", "xai",
# or "groq".
# If unset (or the matching API key is missing), falls back to mock mode.
# Gemini is a reasonable free-tier alternative to a paid Anthropic key --
# see src/llm_client.py.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL = os.environ.get("HIVER_MODEL", "claude-sonnet-4-6")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4.6")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
# NOTE: Gemini model names get retired/renamed over time. If this default
# 404s for you, check https://aistudio.google.com/ for the current list of
# available model names and override with the GEMINI_MODEL env var.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

USE_LLM = (
    (LLM_PROVIDER == "anthropic" and bool(os.environ.get("ANTHROPIC_API_KEY")))
    or (LLM_PROVIDER == "gemini" and bool(os.environ.get("GEMINI_API_KEY")))
    or (LLM_PROVIDER == "xai" and bool(os.environ.get("XAI_API_KEY")))
    or (LLM_PROVIDER == "groq" and bool(os.environ.get("GROQ_API_KEY")))
)
