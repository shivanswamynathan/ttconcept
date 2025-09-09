from typing import List, Dict, Any
from .llm import GeminiLLMWrapper
from backend.prompts import revision_prompts
import asyncio

llm_wrapper = GeminiLLMWrapper()

class RevisionAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_wrapper

    async def generate_explanation_steps(self, title: str, content: str, conversation_history: str = "", steps: int = 3) -> List[str]:
        prompt = revision_prompts.EXPLANATION_TEMPLATE.format(
            title=title, steps=steps, content=content, conversation_history=conversation_history
        )
        # call LLM
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        # naive parse: split into lines, take the top `steps` non-empty lines
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
        if len(lines) == 0:
            # fallback: split sentences from response
            parts = [s.strip() for s in resp.replace("\n", " ").split(".") if s.strip()]
            lines = [f"{i+1}. {parts[i]}." for i in range(min(steps, len(parts)))]
        return lines[:steps]

    async def make_check_question(self, title: str, content: str, conversation_history: str = "") -> str:
        prompt = revision_prompts.CHECK_QUESTION_TEMPLATE.format(
            title=title, content=content, conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        return resp.strip()

    async def evaluate_answer(self, user_answer: str, expected_keywords: List[str], conversation_history: str = "") -> Dict[str, Any]:
        # quick keyword match first
        ua = user_answer.lower()
        matched = [k for k in expected_keywords if k.lower() in ua]
        if len(matched) == len(expected_keywords) and len(expected_keywords) > 0:
            return {"verdict":"CORRECT", "justification": "All keywords present.", "correction": ""}
        if len(matched) > 0:
            return {"verdict":"PARTIAL", "justification": f"Matched keywords: {matched}", "correction": "Add missing key points."}
        # else ask LLM to evaluate
        prompt = revision_prompts.EVAL_PROMPT_TEMPLATE.format(
            keywords=expected_keywords, user_answer=user_answer, conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        # expected LLM reply in the specified format; we'll return it as 'llm_eval'
        return {"verdict": "WRONG", "justification": resp, "correction": resp}
