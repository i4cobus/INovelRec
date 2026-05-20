"""Transformers-only local LLM feature extraction for recommendation ranking."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT_VERSION = "stage4_llm_rerank_v3_transformers_only"


@dataclass(frozen=True)
class LLMMatchResult:
    """Structured local-LLM scoring result for one candidate."""

    llm_match_score: float
    confidence: str
    matched_preferences: list[str] = field(default_factory=list)
    violated_preferences: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def confidence_score(self) -> float:
        return confidence_to_score(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMMatchResult":
        return cls(
            llm_match_score=clamp_score(data.get("llm_match_score", data.get("match_score", 0.0))),
            confidence=normalize_confidence(str(data.get("confidence", "low"))),
            matched_preferences=[str(item) for item in data.get("matched_preferences", data.get("evidence", []))],
            violated_preferences=[str(item) for item in data.get("violated_preferences", [])],
            risk_flags=[str(item) for item in data.get("risk_flags", [])],
            reason=str(data.get("reason", data.get("rationale", ""))),
        )


def clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(number, 1.0))


def normalize_confidence(confidence: str) -> str:
    value = confidence.strip().lower()
    return value if value in {"high", "medium", "low"} else "low"


def confidence_to_score(confidence: str) -> float:
    return {"high": 1.0, "medium": 0.6, "low": 0.3}.get(normalize_confidence(confidence), 0.3)


def build_match_prompt(
    query: str,
    candidate: dict[str, Any],
    profile_text: str,
    max_profile_chars: int = 1200,
) -> str:
    """Build a compact JSON-only scoring prompt."""

    title = str(candidate.get("title_guess", ""))
    semantic_score = float(candidate.get("score", 0.0))
    profile = profile_text[:max_profile_chars]
    return (
        "You are a local Chinese web novel recommendation feature extractor.\n"
        "Score how well the candidate matches the user preference using only the provided profile text.\n"
        "Return valid JSON only. Do not include markdown. Do not add explanations outside JSON.\n"
        "Do not invent plot details. Do not assume popularity, author reputation, ratings, or completion status.\n"
        "If evidence is limited, lower confidence.\n\n"
        f"User preference: {query}\n"
        f"Candidate title: {title}\n"
        f"Semantic score: {semantic_score:.6f}\n"
        f"Profile text:\n{profile}\n\n"
        "Expected JSON:\n"
        '{"llm_match_score":0.0,"confidence":"high|medium|low",'
        '"matched_preferences":["..."],"violated_preferences":["..."],'
        '"risk_flags":["..."],"reason":"one concise sentence"}'
    )


def build_query_expansion_prompt(raw_query: str, max_queries: int = 4) -> str:
    """Build a JSON-only prompt for retrieval query expansion."""

    return (
        "You are improving retrieval recall for a Chinese web novel search system.\n"
        "Rewrite the user preference into retrieval-friendly Chinese search queries.\n"
        "Use related genre, trope, setting, protagonist, pacing, and exclusion terms.\n"
        "Do not recommend titles. Do not invent preferences not implied by the query.\n"
        "Return valid JSON only. No markdown. No explanation outside JSON.\n\n"
        f"User preference: {raw_query}\n"
        f"Return at most {max_queries} expanded queries.\n"
        "Expected JSON:\n"
        '{"expanded_queries":[{"text":"普通资质 主角 修仙 宗门 炼气 筑基 谨慎 低调 慢热成长","source":"llm","weight":0.9}]}'
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from generated text."""

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group(0))


def parse_llm_match_result(text: str) -> LLMMatchResult:
    """Parse a generated JSON match result."""

    return LLMMatchResult.from_dict(extract_json_object(text))


class TransformersMatcher:
    """Local Hugging Face/Qwen Instruct matcher loaded once and reused."""

    provider = "transformers"

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device or "auto",
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens

    def score(self, query: str, candidate: dict[str, Any], profile_text: str, max_profile_chars: int = 1200) -> LLMMatchResult:
        prompt = build_match_prompt(query=query, candidate=candidate, profile_text=profile_text, max_profile_chars=max_profile_chars)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = outputs[0][inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        try:
            return parse_llm_match_result(response)
        except (ValueError, json.JSONDecodeError, TypeError):
            return LLMMatchResult(
                llm_match_score=0.0,
                confidence="low",
                risk_flags=["llm_parse_failed"],
                reason=response[:180],
            )

    def expand_queries(self, raw_query: str, max_queries: int) -> str:
        """Generate retrieval-friendly expanded queries as raw JSON text."""

        prompt = build_query_expansion_prompt(raw_query=raw_query, max_queries=max_queries)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=min(self.max_new_tokens, 256),
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def create_transformers_matcher(
    model_name: str,
    device: str | None = None,
    max_new_tokens: int = 256,
) -> TransformersMatcher:
    """Create the only supported local matcher: Hugging Face transformers."""

    return TransformersMatcher(model_name=model_name, device=device, max_new_tokens=max_new_tokens)
