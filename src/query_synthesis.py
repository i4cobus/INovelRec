"""Synthesise training queries from train-fold novels.

Evaluation queries are hand written and frozen; these are the thousands the
reranker trains and does rollouts on. They must never be the same set, so the
whole pipeline is built around three separations:

* **Fold.** Seeds come only from ``fold=train``. A query generated from a book in
  the eval fold would put that book's text into training.
* **Direction.** Queries are generated *from* a book rather than matched *to*
  one, which gives every query a seed positive that is guaranteed to have an
  answer — without anyone labelling anything.
* **Distribution.** A synthetic query is derived from a profile and is therefore
  answerable from it, making these systematically easier than real user needs.
  That is exactly why the evaluation set stays hand written: measuring on
  synthetic queries would only show whether the model can invert the generator.

The seed is a *weak* label. Other novels in the corpus will also fit, so nothing
downstream may assume "the seed must rank first" — that trains recall of a
specific book, not understanding of a preference. The seed's only job is to
guarantee the query is answerable.

Negative constraints are synthesised in pairs: for one book, a query whose
exclusion it satisfies and a query whose exclusion it violates. That manufactures
supervision for the constraint hypothesis instead of waiting for enough of it to
appear naturally.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from src.preferences import IN_TEXT_NEGATIVES, constraint_violation_by_rule, is_rule_checkable

SYNTHESIS_PROMPT_VERSION = "querysynth_v1"

# Same three shapes the evaluation set uses. Training on one shape only would let
# the model overfit to "4-5 space separated keywords" and make any domain-adaptation
# result meaningless.
SHAPES = ("kw", "sent", "cmp")
SHAPE_INSTRUCTIONS = {
    "kw": "4-6 个空格分隔的关键词，例如「玄幻 升级流 热血 不圣母」",
    "sent": "一句完整的口语化需求，例如「想看主角天赋普通、靠脑子翻盘的玄幻」",
    "cmp": "对比式表述，例如「类似某类作品那种世界观严密的，但不要克苏鲁」（不要写出真实书名）",
}

DEFAULT_QUERIES_PER_BOOK = 2
MIN_QUERY_CHARS = 6
MAX_QUERY_CHARS = 60
# Two queries overlapping this much in tokens are the same need twice.
DUPLICATE_TOKEN_OVERLAP = 0.8
# Below this, a "positive side" is one long phrase rather than a keyword list, and
# comparing it to an evaluation query measures tokenisation rather than leakage.
MIN_POSITIVE_TOKENS_FOR_LEAK_CHECK = 2

TOKEN_SPLIT = re.compile(r"[\s,，、;；/|]+")


@dataclass(frozen=True)
class SynthesisTask:
    """One train-fold novel to generate queries from."""

    novel_id: str
    title: str
    profile: str
    fold: str = "train"


@dataclass(frozen=True)
class SynthesizedQuery:
    """One generated training query plus the supervision it carries."""

    query: str
    shape: str
    wanted: list[str] = field(default_factory=list)
    unwanted: list[str] = field(default_factory=list)
    seed_novel_id: str = ""
    seed_title: str = ""
    # True when the seed book satisfies the exclusion, False when it violates it.
    # The violating half is generated on purpose: it is the only way to get
    # labelled negatives for the constraint arm without hand annotation.
    seed_satisfies_constraint: bool = True
    constraint_checkable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_tokens(query: str) -> set[str]:
    """Split a query into comparable tokens for duplicate detection."""

    return {token for token in TOKEN_SPLIT.split(query.strip()) if len(token) >= 2}


def token_overlap(left: str, right: str) -> float:
    """Jaccard-style overlap used to catch the same need phrased twice."""

    a, b = normalize_tokens(left), normalize_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def is_duplicate(query: str, existing: list[str], threshold: float = DUPLICATE_TOKEN_OVERLAP) -> bool:
    """Whether a query repeats one already generated, or an evaluation query."""

    return any(token_overlap(query, other) >= threshold for other in existing)


NEGATIVE_MARKER_RE = "(?:不要|不想|不含|不|别|无|非)"


def is_self_contradictory(query: str, wanted: list[str], unwanted: list[str]) -> bool:
    """Whether the query asks for the very thing it excludes.

    「类似都市异能那种主角实力强大的，但不要异能」 is 2% of a smoke run — rare, but it
    is not a preference, and a reranker trained on it learns that the exclusion
    clause can be ignored. Cheap and unambiguous to detect, so it is detected.
    """

    for term in unwanted:
        if term in wanted:
            return True
        # Remove the exclusion clause itself; anything left is a positive mention.
        remainder = re.sub(f"{NEGATIVE_MARKER_RE}\\s*{re.escape(term)}", "", query)
        if term in remainder:
            return True
    return False


def build_synthesis_prompt(task: SynthesisTask, shapes: tuple[str, ...] = SHAPES, count: int = DEFAULT_QUERIES_PER_BOOK) -> str:
    """Ask the teacher what a reader would search for to reach this novel.

    The prompt requests one query the book satisfies and one it violates, because
    a corpus-natural sample would leave the violating half far too sparse to train
    a constraint-aware reranker on.
    """

    shape_lines = "\n".join(f"  - {shape}: {SHAPE_INSTRUCTIONS[shape]}" for shape in shapes)
    checkable = "、".join(sorted(IN_TEXT_NEGATIVES))
    return (
        "你在为中文网络小说推荐系统构造训练数据。下面给出一本小说的画像，"
        "请反推：什么样的读者需求会指向这本书。\n\n"
        f"【小说画像】\n{task.profile}\n\n"
        "【要求】\n"
        f"1. 生成 {count} 条读者需求 query，形状各不相同，可选形状：\n{shape_lines}\n"
        "2. 每条 query 都要带一个负向排除项（「不XX」），并标明这本书是否**违反**它：\n"
        "   - satisfies: 这本书**不含**该排除项 —— 它是这条 query 的正例\n"
        "   - violates:  这本书**明确含有**该排除项 —— 它是这条 query 的负例\n"
        f"   **必须生成至少一条 violates**，且负向词优先从这个表里选：{checkable}\n"
        "3. query 只描述读者需求，**不要出现任何书名**——写出书名会让检索退化成字符串匹配。\n"
        "4. 只依据画像判断，不要臆测画像里没有的内容。\n\n"
        "只输出 JSON，不要 markdown：\n"
        '{"queries":[{"query":"玄幻 升级流 热血 不圣母","shape":"kw",'
        '"wanted":["玄幻","升级流"],"unwanted":["圣母"],"seed_satisfies_constraint":true}]}'
    )


def parse_synthesis_response(text: str, task: SynthesisTask) -> list[SynthesizedQuery]:
    """Parse generated queries, dropping anything malformed or unusable."""

    from src.llm_matcher import extract_json_object

    try:
        data = extract_json_object(text)
    except (ValueError, json.JSONDecodeError):
        return []

    results: list[SynthesizedQuery] = []
    for item in data.get("queries", []):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not MIN_QUERY_CHARS <= len(query) <= MAX_QUERY_CHARS:
            continue
        # A book title in the query turns retrieval into string matching against
        # the profile header, which is not the task being trained.
        if "《" in query or "》" in query:
            continue
        unwanted = [str(term).strip() for term in item.get("unwanted", []) if str(term).strip()]
        wanted = [str(term).strip() for term in item.get("wanted", []) if str(term).strip()]
        if is_self_contradictory(query, wanted, unwanted):
            continue
        shape = str(item.get("shape", "kw")).strip().lower()
        results.append(
            SynthesizedQuery(
                query=query,
                shape=shape if shape in SHAPES else "kw",
                wanted=wanted,
                unwanted=unwanted,
                seed_novel_id=task.novel_id,
                seed_title=task.title,
                seed_satisfies_constraint=bool(item.get("seed_satisfies_constraint", True)),
                constraint_checkable=bool(unwanted) and all(is_rule_checkable(term) for term in unwanted),
            )
        )
    return results


def verify_constraint_claim(query: SynthesizedQuery, novel_text: str) -> bool | None:
    """Whether the teacher's satisfies/violates claim matches the text.

    Kept as a *diagnostic*, not a filter. The teacher judges from an 8000-character
    profile while the rule reads all three million characters, so disagreement is
    mostly the teacher extrapolating from one mention in the sample. Measuring that
    rate is worth doing — it is the same extrapolation the student will be asked to
    make at inference — but acting on it deletes data for no gain.

    Returns None when the claim cannot be checked or the rule declines to decide.
    """

    if not query.constraint_checkable or not query.unwanted:
        return None
    violates = constraint_violation_by_rule(novel_text, query.unwanted)
    if violates is None:
        return None
    return query.seed_satisfies_constraint is not violates


def label_constraint_by_rule(query: SynthesizedQuery, novel_text: str) -> SynthesizedQuery | None:
    """Relabel a query's seed from the text, discarding the teacher's claim.

    Asking the teacher whether the seed violates its own exclusion was a design
    error: it invites a wrong answer to a question whose ground truth is free, and
    then the correction throws the query away. Filtering on the claim collapsed the
    violating half from ~35% to under 3% of a smoke run — precisely the half that
    exists to supply labelled negatives for the constraint arm.

    So the teacher writes the query and picks the exclusion; the rule decides which
    side of it the book falls on. Returns None when the rule abstains, because a
    query with no trustworthy label is not training data.
    """

    if not query.constraint_checkable or not query.unwanted:
        return query
    violates = constraint_violation_by_rule(novel_text, query.unwanted)
    if violates is None:
        return None
    return replace(query, seed_satisfies_constraint=not violates)


def looks_truncated(text: str) -> bool:
    """Whether generation stopped mid-JSON.

    A truncated response parses to zero queries, which is indistinguishable from
    "this book yielded nothing" unless it is counted separately. At 600 output
    tokens a third of responses were cut off mid-object and the loss was invisible
    in the summary; the run reported `Failed requests 0` while silently discarding
    half its seeds.
    """

    body = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return body.count("{") != body.count("}")


def positive_tokens(query: str, unwanted: list[str]) -> set[str]:
    """Query tokens with the exclusion clause removed.

    Only meaningful for space-separated keyword queries; a sentence-shaped query
    yields one or two long tokens, which is why callers require a minimum size
    before comparing.
    """

    return {token for token in normalize_tokens(query) if not any(term in token for term in unwanted)}


def leaks_positive_side(query: SynthesizedQuery, reserved: list[SynthesizedQuery | str], threshold: float = DUPLICATE_TOKEN_OVERLAP) -> bool:
    """Whether a query reuses an evaluation query's positive half verbatim.

    「美食 日常 温情 不重生」 against evaluation's 「美食 日常 温情 不系统」 falls under the
    whole-query overlap threshold because the exclusion differs, yet training on it
    still shows the model exactly which books satisfy that positive phrasing. Rare
    (5 in 9,415) but free to remove, and evaluation validity is not worth 0.1%.
    """

    mine = positive_tokens(query.query, query.unwanted)
    if len(mine) < MIN_POSITIVE_TOKENS_FOR_LEAK_CHECK:
        return False
    for other in reserved:
        text, unwanted = (other, []) if isinstance(other, str) else (other.query, other.unwanted)
        theirs = positive_tokens(text, unwanted)
        if len(theirs) < MIN_POSITIVE_TOKENS_FOR_LEAK_CHECK:
            continue
        if len(mine & theirs) / min(len(mine), len(theirs)) >= threshold:
            return True
    return False


def deduplicate(
    queries: list[SynthesizedQuery],
    reserved: list[str],
    reserved_queries: list[SynthesizedQuery] | None = None,
) -> tuple[list[SynthesizedQuery], int, int]:
    """Drop queries repeating each other, or leaking an evaluation query.

    Returns the kept queries, the duplicate count, and the positive-side leak count.
    """

    kept: list[SynthesizedQuery] = []
    seen = list(reserved)
    dropped = leaked = 0
    guard = reserved_queries if reserved_queries is not None else list(reserved)
    for item in queries:
        if is_duplicate(item.query, seen):
            dropped += 1
            continue
        if leaks_positive_side(item, guard):
            leaked += 1
            continue
        seen.append(item.query)
        kept.append(item)
    return kept, dropped, leaked
