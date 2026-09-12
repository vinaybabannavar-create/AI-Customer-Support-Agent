"""
LLM-as-judge: scores a drafted reply against the customer message + the
grounding evidence it was supposed to use, on a fixed rubric (1-5 each).

Deliberately NOT reusing the drafting model's own reasoning -- the judge
prompt is asked to be adversarial/skeptical about grounding in particular,
since that's the dimension most likely to be rubber-stamped by a lenient
judge. See report "how well does your judge agree with a human" section
and eval/human_agreement.py for calibration evidence.
"""
import json
from src.json_utils import parse_json_object
from src.llm_client import call_llm

JUDGE_SYSTEM_PROMPT = """You are a strict judge evaluating a customer
support reply. Score each dimension 1 (poor) to 5 (excellent):

- grounded: Does the reply avoid inventing policies/promises/timelines not
  supported by the provided historical examples or the customer's message?
  Score 1 if it invents anything (a refund amount, a specific date, a
  policy) that isn't backed by the grounding.
- correct_intent_handling: Does the reply actually address the customer's
  stated issue (not a generic non-answer)?
- tone: Is the tone appropriate for a brand support account (empathetic,
  not robotic, not overly apologetic)?
- actionable: Does it give the customer a clear, concrete next step?
- overall: Your holistic 1-5 score.

Return ONLY a JSON object:
{"grounded": 1-5, "correct_intent_handling": 1-5, "tone": 1-5,
 "actionable": 1-5, "overall": 1-5, "rationale": "one sentence"}
"""


def judge_reply(customer_message: str, draft_reply: str, grounding_examples: list):
    grounding_block = "\n".join(
        f"- \"{g['customer_message']}\" -> \"{g.get('brand_reply', '')}\""
        for g in grounding_examples
    ) or "(none provided)"

    user = f"""Customer message: "{customer_message}"

Grounding examples given to the drafting model:
{grounding_block}

Drafted reply to judge: "{draft_reply}"
"""
    # NOTE: max_tokens raised from 250 -> 500. At 250, Gemini's response
    # (which ends with the free-text "rationale" field) was sometimes
    # getting cut off mid-sentence, producing truncated/unparseable JSON --
    # that's the leading suspect for the near-zero judge scores seen in
    # outputs/eval_results.json. See README "Known issues" #2.
    raw = call_llm(JUDGE_SYSTEM_PROMPT, user, max_tokens=500)
    try:
        return parse_json_object(raw)
    except Exception:
        # Print the raw response so a parse failure is debuggable instead
        # of silently collapsing to a 0 score -- set HIVER_DEBUG=0 to quiet
        # this down once you've diagnosed the issue.
        import os
        if os.environ.get("HIVER_DEBUG", "1") != "0":
            print(f"  [judge PARSE_ERROR] raw response was:\n  {raw!r}\n")
        return {"grounded": 0, "correct_intent_handling": 0, "tone": 0,
                "actionable": 0, "overall": 0, "rationale": "PARSE_ERROR"}