"""LLM-as-judge relevance labelling with a hard spend cap.

The corpus has no relevance labels, so system comparisons rest on judged
verdicts. Two properties make those verdicts worth trusting:

1. The judge reads evidence the *system never saw* (see ``src/evidence.py``),
   so it grades the book rather than the system's own summary of it.
2. Spend is bounded by the endpoint-reported ``usage``, not by an estimate.
   ``BudgetGuard`` refuses to start a run it cannot afford and stops mid-run the
   moment the cap is reached, flushing whatever completed.

Verdicts use the same schema as the human annotation sheet
(``relevance_label`` 0/1/2 plus ``constraint_violation``) so judge and human
columns can be compared directly with Cohen's kappa.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from src.config import DATA_DIR
from src.http_matcher import TokenUsage

JUDGE_PROMPT_VERSION = "judge_v1"
JUDGE_CACHE_PATH = DATA_DIR / "cache" / "judge_cache.jsonl"
DEFAULT_JUDGE_MAX_TOKENS = 300
DEFAULT_JUDGE_WORKERS = 8

# Rough per-item request shape, used only for the pre-flight estimate.
ESTIMATED_INPUT_TOKENS = 2500
ESTIMATED_OUTPUT_TOKENS = 150


class JudgeTransport(Protocol):
    """Chat transport that reports token usage, so spend is measured not guessed."""

    def complete_with_usage(self, prompt: str, max_tokens: int) -> tuple[str, TokenUsage]:
        """Return the response text and its token usage."""


@dataclass(frozen=True)
class PricePerMillion:
    """Endpoint pricing in USD per million tokens."""

    input_usd: float
    output_usd: float

    def cost(self, usage: TokenUsage) -> float:
        return (usage.prompt_tokens * self.input_usd + usage.completion_tokens * self.output_usd) / 1_000_000


@dataclass(frozen=True)
class JudgeTask:
    """One (query, novel) pair to be judged."""

    query_id: str
    query: str
    novel_id: str
    title: str
    evidence: str
    wanted: list[str] = field(default_factory=list)
    unwanted: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JudgeVerdict:
    """A judged label, mirroring the human annotation columns."""

    relevance_label: int
    constraint_violation: bool
    reason: str = ""
    judge_confidence: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeVerdict":
        return cls(
            relevance_label=clamp_label(data.get("relevance_label", 0)),
            constraint_violation=coerce_bool(data.get("constraint_violation", False)),
            reason=str(data.get("reason", ""))[:400],
            judge_confidence=normalize_confidence(str(data.get("judge_confidence", "low"))),
        )


def clamp_label(value: Any) -> int:
    """Coerce a relevance label into {0, 1, 2}."""

    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(number, 2))


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_confidence(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in {"high", "medium", "low"} else "low"


def build_judge_prompt(task: JudgeTask) -> str:
    """Build a JSON-only grading prompt over independently sampled evidence."""

    wanted = "、".join(task.wanted) if task.wanted else "（未显式列出）"
    unwanted = "、".join(task.unwanted) if task.unwanted else "（无）"
    return (
        "你是中文网络小说推荐结果的评审员。\n"
        "请只依据下方提供的正文摘录判断这本小说是否符合用户偏好。\n"
        "摘录是从原文不同位置随机截取的片段，不是完整作品，信息有限时请降低置信度。\n"
        "不要臆测剧情、人气、作者、评分或完结状态。只输出合法 JSON，不要 markdown。\n\n"
        f"用户偏好：{task.query}\n"
        f"正向要求：{wanted}\n"
        f"负向排除：{unwanted}\n"
        f"候选标题：{task.title}\n"
        f"正文摘录：\n{task.evidence}\n\n"
        "评分标准：\n"
        "relevance_label = 2 高度相关；1 部分相关；0 不相关\n"
        "constraint_violation = true 当且仅当摘录中有证据表明它违反了负向排除项\n\n"
        "输出 JSON：\n"
        '{"relevance_label":0,"constraint_violation":false,'
        '"judge_confidence":"high|medium|low","reason":"一句话依据"}'
    )


def parse_judge_verdict(text: str) -> JudgeVerdict:
    """Parse a judge response, falling back to a neutral abstention."""

    try:
        return JudgeVerdict.from_dict(json.loads(text))
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return JudgeVerdict(relevance_label=0, constraint_violation=False, reason="judge_parse_failed")
    try:
        return JudgeVerdict.from_dict(json.loads(match.group(0)))
    except (json.JSONDecodeError, TypeError):
        return JudgeVerdict(relevance_label=0, constraint_violation=False, reason="judge_parse_failed")


def judge_cache_key(task: JudgeTask, judge_model: str) -> str:
    """Stable key over the pair, the evidence, the model, and the prompt version.

    Evidence is hashed so re-sampling it invalidates the entry: a verdict is only
    valid for the text the judge actually read.
    """

    payload = {
        "query_id": task.query_id,
        "query": task.query,
        "novel_id": task.novel_id,
        "evidence": hashlib.sha256(task.evidence.encode("utf-8")).hexdigest(),
        "judge_model": judge_model,
        "prompt_version": JUDGE_PROMPT_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_judge_cache(cache_path: Path = JUDGE_CACHE_PATH) -> dict[str, JudgeVerdict]:
    """Load cached verdicts keyed by cache key."""

    if not cache_path.exists():
        return {}
    cache: dict[str, JudgeVerdict] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = str(item.get("cache_key", ""))
        if key:
            cache[key] = JudgeVerdict.from_dict(item.get("verdict", {}))
    return cache


def append_judge_cache(cache_path: Path, cache_key: str, verdict: JudgeVerdict) -> None:
    """Append one verdict to the JSONL cache."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"cache_key": cache_key, "verdict": verdict.to_dict()}, ensure_ascii=False) + "\n")


class BudgetExceeded(RuntimeError):
    """Raised before a run that cannot be afforded."""


@dataclass
class BudgetGuard:
    """Hard USD ceiling enforced against endpoint-reported usage."""

    limit_usd: float
    prices: PricePerMillion
    spent_usd: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def estimate_usd(
        self,
        items: int,
        input_tokens: int = ESTIMATED_INPUT_TOKENS,
        output_tokens: int = ESTIMATED_OUTPUT_TOKENS,
    ) -> float:
        """Pre-flight cost estimate for a number of uncached items."""

        return self.prices.cost(
            TokenUsage(prompt_tokens=items * input_tokens, completion_tokens=items * output_tokens)
        )

    def remaining_usd(self) -> float:
        with self._lock:
            return max(self.limit_usd - self.spent_usd, 0.0)

    def exhausted(self) -> bool:
        return self.remaining_usd() <= 0.0

    def record(self, usage: TokenUsage) -> float:
        """Accumulate real usage and return the new total spend."""

        with self._lock:
            self.usage = self.usage + usage
            self.spent_usd += self.prices.cost(usage)
            return self.spent_usd

    def ensure_affordable(self, items: int) -> float:
        """Raise unless the projected cost fits inside the remaining budget."""

        projected = self.estimate_usd(items)
        if projected > self.remaining_usd():
            raise BudgetExceeded(
                f"Estimated ${projected:.2f} for {items} items exceeds the remaining "
                f"${self.remaining_usd():.2f} budget. Lower the item count, switch to a "
                f"cheaper judge model, or raise --budget-usd."
            )
        return projected


@dataclass
class JudgeRunSummary:
    """Outcome of a judging run, including why it stopped."""

    requested: int = 0
    cache_hits: int = 0
    judged: int = 0
    failed: int = 0
    skipped_over_budget: int = 0
    spent_usd: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def stopped_early(self) -> bool:
        return self.skipped_over_budget > 0


ResultCallback = Callable[[JudgeTask, JudgeVerdict | None], None]


def run_judgements(
    tasks: list[JudgeTask],
    transport: JudgeTransport,
    judge_model: str,
    budget: BudgetGuard,
    cache_path: Path = JUDGE_CACHE_PATH,
    use_cache: bool = True,
    max_workers: int = DEFAULT_JUDGE_WORKERS,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    on_result: ResultCallback | None = None,
) -> tuple[dict[str, JudgeVerdict], JudgeRunSummary]:
    """Judge tasks concurrently, reusing cache and stopping at the budget cap.

    Returns verdicts keyed by cache key plus a summary. A task whose request
    failed is absent from the mapping rather than recorded as a zero, so a dead
    request is never mistaken for "the judge said not relevant".
    """

    summary = JudgeRunSummary(requested=len(tasks))
    verdicts: dict[str, JudgeVerdict] = {}
    if not tasks:
        return verdicts, summary

    cache = load_judge_cache(cache_path) if use_cache else {}
    pending: list[tuple[str, JudgeTask]] = []

    for task in tasks:
        key = judge_cache_key(task, judge_model)
        cached = cache.get(key)
        if cached is not None:
            verdicts[key] = cached
            summary.cache_hits += 1
            if on_result:
                on_result(task, cached)
        else:
            pending.append((key, task))

    if not pending:
        return verdicts, summary

    budget.ensure_affordable(len(pending))
    write_lock = threading.Lock()

    def run(entry: tuple[str, JudgeTask]) -> None:
        key, task = entry
        if budget.exhausted():
            with write_lock:
                summary.skipped_over_budget += 1
            return
        try:
            response, usage = transport.complete_with_usage(build_judge_prompt(task), max_tokens)
        except Exception:  # noqa: BLE001 - one dead request must not kill the run
            with write_lock:
                summary.failed += 1
            if on_result:
                on_result(task, None)
            return

        budget.record(usage)
        verdict = parse_judge_verdict(response)
        with write_lock:
            verdicts[key] = verdict
            summary.judged += 1
            summary.usage = summary.usage + usage
            if use_cache:
                append_judge_cache(cache_path, key, verdict)
        if on_result:
            on_result(task, verdict)

    workers = min(max_workers, len(pending))
    if workers <= 1:
        for entry in pending:
            run(entry)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(run, pending))

    summary.spent_usd = budget.spent_usd
    return verdicts, summary
