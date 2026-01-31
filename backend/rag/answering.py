# backend/rag/answering.py
# Phase 4: LLM answering with citations (Gemini via google-genai)
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv(".env.local")

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _build_sources(results: List[Dict[str, Any]]) -> str:
    """Format retrieval results into a sources block with chunk_id + page."""
    lines: List[str] = []
    for r in results:
        chunk_id = r.get("chunk_uid")
        page = r.get("page_number")
        text = (r.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{chunk_id}] (page {page}) {text}")
    return "\n".join(lines)


def _get_text(resp: Any) -> str:
    """
    Extract text from google-genai response across versions.
    Prefer resp.text; fallback to candidates/parts if needed.
    """
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t

    try:
        # Common structure: resp.candidates[0].content.parts -> each part may have .text
        parts = resp.candidates[0].content.parts
        out = "".join(p.text for p in parts if getattr(p, "text", None))
        return out or ""
    except Exception:
        return ""


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Gemini may return:
      - raw JSON
      - JSON wrapped in ```json fences
      - extra prose before/after JSON
    This extracts the first JSON object and parses it.
    """
    if not text or not text.strip():
        raise ValueError("Gemini returned empty text (nothing to parse).")

    t = text.strip()

    # Remove fenced code blocks if present
    t = re.sub(r"^```json\s*", "", t)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()

    # Extract the first JSON object substring
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Gemini did not return JSON. Raw output:\n{t[:500]}")

    candidate = t[start : end + 1]
    return json.loads(candidate)


def answer_with_citations(question: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Inputs:
      question: user question
      results: list of retrieved chunks from retrieval.py, each with:
        - chunk_uid
        - page_number
        - text

    Output (dict):
      {
        "answer": "...",
        "citations": [{"chunk_id": "...", "page": 1, "snippet": "..."}, ...]
      }

    Hard guardrail: citations must reference only retrieved chunk_uids.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not results:
        return {"answer": "I don't know based on the provided sources.", "citations": []}

    allowed = {r["chunk_uid"] for r in results if "chunk_uid" in r}

    prompt = f"""
You are answering a question using ONLY the provided sources.

Rules:
- Use ONLY the sources below. Do not use outside knowledge.
- If the answer is not contained in the sources, return exactly:
  "I don't know based on the provided sources."
- Cite sources using chunk IDs like [docid_00012] where the ID matches the bracketed IDs in Sources.
- Return ONLY JSON. No markdown. No backticks. No extra text.

JSON format:
{{
  "answer": "string",
  "citations": [
    {{"chunk_id": "string", "page": 1, "snippet": "string"}}
  ]
}}

Question:
{question}

Sources:
{_build_sources(results)}
""".strip()

    resp = client.models.generate_content(model=MODEL, contents=prompt)

    text = _get_text(resp)



    data = _extract_json(text)

    # Basic shape checks
    if "answer" not in data or "citations" not in data:
        raise ValueError(f"Gemini JSON missing keys. Raw output:\n{(text or '')[:500]}")

    if not isinstance(data["citations"], list):
        raise ValueError("citations must be a list")

    # Validate citations: chunk_id must be from retrieved set
    for c in data["citations"]:
        if not isinstance(c, dict):
            raise ValueError("Each citation must be an object")
        if "chunk_id" not in c or "page" not in c or "snippet" not in c:
            raise ValueError("Each citation must contain chunk_id, page, snippet")
        if c["chunk_id"] not in allowed:
            raise ValueError(f"Invalid citation chunk_id: {c['chunk_id']}")

    return data
