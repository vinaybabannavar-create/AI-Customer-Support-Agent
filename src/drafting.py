"""Drafts a reply grounded on retrieved similar historically-resolved threads."""
from src.llm_client import call_llm

SYSTEM_PROMPT = """You are drafting a customer support reply for a brand's
Twitter support account. Write a short (1-3 sentence) customer support reply.

Grounding rules:
- You will be given 1-3 examples of how this brand has historically resolved
  similar customer issues. Match their tone, structure, and the concrete next
  action they ask for (usually: "DM us your order number").
- Do NOT invent policies, refund amounts, timelines, or promises that are not
  supported by the grounding examples or the customer's own message.
- If nothing in the grounding examples supports a confident resolution,
  acknowledge the issue and ask for the specific info needed (order number,
  etc.) rather than guessing.
- Do not use a customer's real name if you don't have it.
"""


def draft_reply(customer_message: str, grounding_examples: list):
    grounding_block = "\n".join(
        f"- Similar past issue: \"{g['customer_message']}\"\n"
        f"  Brand's resolution: \"{g['brand_reply']}\""
        for g in grounding_examples
    ) or "(no similar historical example found -- respond generically and ask for order details)"

    user = f"""Customer message:
"{customer_message}"

Historically similar resolved cases:
{grounding_block}

Write the reply now."""
    return call_llm(SYSTEM_PROMPT, user, max_tokens=200).strip()
