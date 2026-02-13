# src/llm.py
from __future__ import annotations
import os
from typing import Dict, Any, List
from openai import OpenAI

from .schemas import INSIGHTS_JSON_SCHEMA


def get_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_insights(facts_pack: Dict[str, Any], user_question: str | None = None) -> Dict[str, Any]:
    """
    Uses GPT-5.2 to turn computed facts into: insights + recommended actions + chart specs.
    Grounded: The model is instructed to ONLY use facts_pack.
    """
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    client = get_client()

    prompt = _build_prompt(facts_pack=facts_pack, user_question=user_question)

    resp = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": INSIGHTS_JSON_SCHEMA["name"],
                "schema": INSIGHTS_JSON_SCHEMA["schema"],
                "strict": True,
            }
        },
        temperature=0.2,
    )

    # The SDK returns structured text output; easiest is resp.output_text
    # but with structured outputs you can parse as JSON from output_text safely.
    import json
    return json.loads(resp.output_text)


def chat_answer(
    facts_pack: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    user_message: str,
) -> str:
    """
    Lightweight chat: answers using only facts_pack and prior messages.
    (V1: we keep it text-only in chat; charts come from the main insights call.)
    """
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    client = get_client()

    system_rules = (
        "You are a fleet safety insights assistant. "
        "You MUST only use the provided facts_pack. "
        "If the answer is not supported by facts_pack, say what is missing and how to provide it. "
        "Do not invent numbers."
    )

    # Build conversation input
    convo = [{"role": "system", "content": system_rules}]
    convo.append({"role": "user", "content": "facts_pack:\n" + _safe_truncate_json(facts_pack, max_chars=14000)})

    for m in chat_history[-8:]:
        convo.append({"role": m["role"], "content": m["content"]})

    convo.append({"role": "user", "content": user_message})

    resp = client.responses.create(
        model=model,
        input=convo,
        temperature=0.2,
    )
    return resp.output_text


def _build_prompt(facts_pack: Dict[str, Any], user_question: str | None) -> str:
    base = (
        "You are a fleet safety analytics expert.\n"
        "You will receive a facts_pack computed deterministically from an uploaded report.\n"
        "Rules:\n"
        "1) Use ONLY facts_pack for all numbers and claims.\n"
        "2) Be concise and executive-friendly.\n"
        "3) Provide 3–5 charts using only available fields.\n"
        "4) If data quality issues exist, include them in data_quality_notes.\n"
        "5) Prioritize coaching and operational actions (P0/P1/P2).\n\n"
    )
    if user_question:
        base += f"User focus question: {user_question}\n\n"

    base += "facts_pack:\n"
    base += _safe_truncate_json(facts_pack, max_chars=25000)
    return base


def _safe_truncate_json(obj: Dict[str, Any], max_chars: int) -> str:
    import json
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "...(truncated)"
