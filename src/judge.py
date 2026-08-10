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

JUDGE_PROMPT_VERSION = "judge_v2"
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
    """Build a JSON-only grading prompt over independently sampled evidence.

    v2 fixes three failure modes measured against 200 human labels with v1
    (relevance weighted kappa 0.244):

    * The judge scored 0 on 40% of items against the human's 12.5%, marking 37 of
      the human's 118 "highly relevant" as "not relevant". "Do not invent details"
      was being read as "score low when unsure", so confidence and label are now
      separated explicitly: thin evidence lowers *confidence*, never the label.
    * It barely used the middle label (14 items against the human's 57), so each
      label carries an operational test rather than a one-word gloss.
    * It missed 21 of the human's 34 constraint violations. "Only when there is
      evidence it violates" set the burden of proof too high. Presence of the
      excluded element now suffices, and the asymmetry is stated outright: the
      excerpts sample a long novel, so presence is evidence while absence is not.
    """

    wanted = "、".join(task.wanted) if task.wanted else "（未显式列出，按 query 字面理解）"
    unwanted = "、".join(task.unwanted) if task.unwanted else "（无）"
    return (
        "你是中文网络小说推荐结果的评审员。下面给出一本小说的作者简介和若干段正文摘录，"
        "请判断它是否符合用户偏好。\n\n"
        "【最重要的两条原则】\n"
        "1. 摘录只是全书的极小一部分。出现即证据，未出现不算证据——摘录里没提到某个要素，"
        "不能据此推断全书没有它。\n"
        "2. 证据不足时请降低 judge_confidence，不要因此降低 relevance_label。"
        "按摘录所能支持的最合理判断打分，而不是因为看不全就打低分。\n\n"
        "【relevance_label 判据】\n"
        "2 = 高度相关：摘录体现了正向要求中的主要部分（题材、设定、主角类型、氛围等），"
        "一个抱着这条偏好来找书的读者会认为「就是它」。\n"
        "1 = 部分相关：题材大方向对但侧重不同，或只满足部分要求，读者会觉得「沾边但不完全是」。\n"
        "0 = 不相关：题材或核心设定与要求明显不符。\n"
        "注意：检索结果多数应落在 1 或 2，只有明显跑题才给 0。\n\n"
        "【constraint_violation 判据】\n"
        "true = 摘录中出现了负向排除的要素即可判定，无需判断它是否为全书主线。\n"
        "  例：排除「系统」而摘录出现系统面板／任务奖励；排除「穿越」而主角来自现代；"
        "排除「后宫」而出现多位女性伴侣。\n"
        "false = 摘录中没有出现该要素。负向排除为「（无）」时一律 false。\n\n"
        f"用户偏好：{task.query}\n"
        f"正向要求：{wanted}\n"
        f"负向排除：{unwanted}\n"
        f"候选标题：{task.title}\n"
        f"正文摘录：\n{task.evidence}\n\n"
        "只输出 JSON，不要 markdown，不要 JSON 之外的解释：\n"
        '{"relevance_label":0,"constraint_violation":false,'
        '"judge_confidence":"high|medium|low","reason":"一句话依据"}'
    )


def parse_judge_verdict(text: str) -> JudgeVerdict | None:
    """Parse a judge response, or return ``None`` when it cannot be parsed.

    ``None``, not label 0. An unparseable response means *the judge did not answer*,
    which is the same situation as a dead request and must be excluded from the
    results — recording it as "not relevant, no violation" invents a verdict and
    biases both metrics downward at once. ``run_judgements`` therefore neither
    records nor caches it.

    Parsing reuses ``extract_json_object`` rather than a greedy ``{.*}``. That regex
    spans from the first brace anywhere in the output to the last, so a single brace
    inside a reasoning model's ``<think>`` block swallows the real answer. Measured
    on a trace containing ``{系统}`` followed by a valid verdict of
    ``label 2 / violation true``, the old parser returned ``label 0 / violation
    false`` — inverted on both fields — and cached it.
    """

    from src.llm_matcher import extract_json_object

    try:
        return JudgeVerdict.from_dict(extract_json_object(text))
    except (ValueError, json.JSONDecodeError, TypeError, AttributeError):
        return None


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
    unparsed: int = 0
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
        if verdict is None:
            # The tokens are spent either way, so the usage above still counts. But an
            # unparseable answer is not a verdict: it is neither recorded nor cached,
            # exactly as a failed request is not. Caching it would freeze a
            # non-answer into the results for every future run.
            with write_lock:
                summary.unparsed += 1
                summary.usage = summary.usage + usage
            if on_result:
                on_result(task, None)
            return
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
