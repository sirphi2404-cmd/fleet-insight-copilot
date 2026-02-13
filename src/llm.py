from __future__ import annotations
import os
import json
from typing import Dict, Any, List

from openai import OpenAI

from .schemas import INSIGHTS_JSON_SCHEMA


def get_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_insights(facts_pack: Dict[str, Any], user_question: str | None = None) -> Dict[str, Any]:
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    client = get_client()
    prompt = _build_prompt(facts_pack=facts_pack, user_question=user_question)

    # 1) Try Responses API (newer SDK)
    try:
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
        return json.loads(resp.output_text)
    except Exception:
        # 2) Fallback: Chat Completions (more widely supported)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid JSON that matches the required schema. "
                        "No markdown. No extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        return json.loads(content)


def chat_answer(
    facts_pack: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    user_message: str,
) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    client = get_client()

    system_rules = (
        "You are a fleet safety insights assistant. "
        "You MUST only use the provided facts_pack. "
        "If the answer is not supported by facts_pack, say what is missing and how to provide it. "
        "Do not invent numbers."
    )

    convo = [{"role": "system", "content": system_rules}]
    convo.append({"role": "user", "content": "facts_pack:\n" + _safe_truncate_json(facts_pack, max_chars=14000)})

    for m in chat_history[-8:]:
        convo.append({"role": m["role"], "content": m["content"]})

    convo.append({"role": "user", "content": user_message})

    # Try responses, fallback to chat completions
    try:
        resp = client.responses.create(model=model, input=convo, temperature=0.2)
        return resp.output_text
    except Exception:
        resp = client.chat.completions.create(model=model, messages=convo, temperature=0.2)
        return resp.choices[0].message.content


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
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= max_chars else s[:max_chars] + "...(truncated)"
