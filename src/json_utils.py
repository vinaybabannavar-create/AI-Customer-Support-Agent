"""Utilities for parsing LLM JSON responses that may include extra text."""
import json


def parse_json_object(text: str):
    """Parse a JSON object, tolerating Markdown fences or surrounding text."""
    if not text:
        raise ValueError("empty LLM response")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in LLM response")
    return json.loads(cleaned[start:end + 1])
