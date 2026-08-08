"""Browser-based relevance annotation for the human-vs-judge calibration set.

Two properties matter more than convenience:

* **The annotator must not see what the system thought.** Rank and system variant
  are deliberately withheld. Knowing that a result was ranked first biases a
  human toward agreeing with it, and the whole point of these labels is to be an
  independent yardstick — both for judge agreement (Cohen's kappa) and, later,
  for the constraint metric that GRPO optimises against a keyword rule.
* **Progress survives a crash or a closed tab.** Every label is appended to a
  JSONL log the moment it is saved, and the sheet is rebuilt from that log. A
  three-hour task will not be done in one sitting.

Run with:
    uv run streamlit run src/annotate_app.py --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

SHEET_PATH = Path("eval/manual_judgements_sheet.csv")
LOG_PATH = Path("eval/annotations.jsonl")
AGREEMENT_PATH = Path("eval/results/agreement_terra.csv")
JUDGE_CACHE_PATH = Path("data/cache/judge_cache.jsonl")
REVIEW_LOG_PATH = Path("eval/disagreement_reviews.jsonl")

MODE_ANNOTATE = "标注"
MODE_REVIEW = "分歧复核"

REVIEW_VERDICTS = {
    "judge_right": "judge 对 — 我当时标松/标严了",
    "human_right": "我对 — judge 判错了",
    "neither": "都不对 / 说不清",
}

# Withheld from the annotator on purpose; see the module docstring.
HIDDEN_COLUMNS = ("rank", "system_variant")

RELEVANCE_OPTIONS = {
    2: "2 · 高度相关 — 明确符合这条偏好",
    1: "1 · 部分相关 — 沾边但不完全符合",
    0: "0 · 不相关 — 不符合",
}


def load_sheet(path: Path) -> pd.DataFrame:
    """Load the annotation sheet produced by 10_annotation_sheet.py."""

    frame = pd.read_csv(path)
    required = {"query_id", "query", "novel_id", "title_guess", "evidence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Sheet is missing columns: {sorted(missing)}")
    frame["query_id"] = frame["query_id"].astype(str)
    frame["novel_id"] = frame["novel_id"].astype(str)
    for column in ("wanted", "unwanted"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("")
    return frame


def load_labels(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Replay the append-only log; later entries win."""

    labels: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        labels[(str(record["query_id"]), str(record["novel_id"]))] = record
    return labels


def append_label(path: Path, record: dict[str, Any]) -> None:
    """Append one label, flushing immediately so a crash cannot lose it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_filled_sheet(sheet: pd.DataFrame, labels: dict[tuple[str, str], dict[str, Any]], path: Path) -> int:
    """Rebuild the sheet with labels applied, for 11_agreement.py to read."""

    filled = sheet.copy()
    filled["relevance_label"] = [
        labels.get((row.query_id, row.novel_id), {}).get("relevance_label", "") for row in filled.itertuples()
    ]
    filled["constraint_violation"] = [
        labels.get((row.query_id, row.novel_id), {}).get("constraint_violation", "") for row in filled.itertuples()
    ]
    filled["notes"] = [labels.get((row.query_id, row.novel_id), {}).get("notes", "") for row in filled.itertuples()]
    filled.to_csv(path, index=False, encoding="utf-8-sig")
    return int((filled["relevance_label"] != "").sum())


def next_unlabeled(sheet: pd.DataFrame, labels: dict[tuple[str, str], dict[str, Any]], start: int) -> int:
    """Return the next index without a label, wrapping around.

    When everything is labelled there is nowhere to go, so stay put — clamped into
    range, because the caller seeds this with -1 on first load and a sheet that is
    already complete would otherwise hand back -1 as a row number.
    """

    total = len(sheet)
    if total == 0:
        return 0
    for offset in range(1, total + 1):
        index = (start + offset) % total
        row = sheet.iloc[index]
        if (row["query_id"], row["novel_id"]) not in labels:
            return index
    return start % total if start >= 0 else 0


def load_disagreements(path: Path, min_gap: int = 2) -> pd.DataFrame:
    """Rows where human and judge differ by at least ``min_gap`` labels."""

    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame[frame["relevance_label"].notna() & frame["judge_relevance_label"].notna()].copy()
    frame["human"] = frame["relevance_label"].astype(int)
    frame["judge"] = frame["judge_relevance_label"].astype(int)
    frame["gap"] = (frame["human"] - frame["judge"]).abs()
    return frame[frame["gap"] >= min_gap].sort_values(["query_id", "novel_id"]).reset_index(drop=True)


def load_judge_reasons(path: Path) -> dict[str, dict[str, Any]]:
    """Verdict bodies keyed by cache key, for showing the judge's stated reason."""

    reasons: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return reasons
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        reasons[str(record.get("cache_key", ""))] = record.get("verdict", {})
    return reasons


def judge_reason_for(row: Any, sheet: pd.DataFrame, reasons: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    """Recompute the cache key for one pair so its reason can be displayed."""

    from src.evaluation import load_eval_queries
    from src.judge import JudgeTask, judge_cache_key

    match = sheet[(sheet["query_id"] == row.query_id) & (sheet["novel_id"] == str(row.novel_id))]
    if match.empty:
        return {}
    queries = {q.query_id: q for q in load_eval_queries(Path("eval/eval_queries.jsonl"))}
    query = queries.get(row.query_id)
    if query is None:
        return {}
    task = JudgeTask(
        query_id=row.query_id,
        query=query.query,
        novel_id=str(row.novel_id),
        title=str(row.title_guess),
        evidence=str(match.iloc[0]["evidence"]),
        wanted=query.wanted,
        unwanted=query.unwanted,
    )
    return reasons.get(judge_cache_key(task, model), {})


def render_review(sheet: pd.DataFrame) -> None:
    """Adjudicate cases where the judge and the annotator disagree sharply.

    Both read the same evidence, so a two-label gap is either the judge failing or
    the annotator's own standard drifting. Which one decides whether kappa can be
    trusted — and kappa alone cannot tell them apart, because a low value is also
    what a skewed label distribution produces when agreement is genuinely high.
    """

    disagreements = load_disagreements(AGREEMENT_PATH)
    if disagreements.empty:
        st.info(f"未找到分歧数据。先运行 09_judge_eval.py 与 11_agreement.py 生成 {AGREEMENT_PATH}。")
        return

    if "reviews" not in st.session_state:
        st.session_state.reviews = load_labels(REVIEW_LOG_PATH)
    if "review_index" not in st.session_state:
        st.session_state.review_index = 0

    reviews = st.session_state.reviews
    reasons = load_judge_reasons(JUDGE_CACHE_PATH)
    total = len(disagreements)
    done = sum(1 for row in disagreements.itertuples() if (row.query_id, str(row.novel_id)) in reviews)

    with st.sidebar:
        st.header("复核进度")
        st.progress(done / total if total else 0.0)
        st.metric("已复核", f"{done} / {total}")
        st.divider()
        tally: dict[str, int] = {}
        for record in reviews.values():
            tally[record.get("verdict", "?")] = tally.get(record.get("verdict", "?"), 0) + 1
        for key, label in REVIEW_VERDICTS.items():
            st.write(f"{label.split(' — ')[0]}: **{tally.get(key, 0)}**")
        st.divider()
        st.caption(
            "两边看的是同一份摘录。若多数判 judge 对，说明可以信任 judge 去跑全量；"
            "若多数判你对，说明 judge 不能替代人工。"
        )

    index = min(st.session_state.review_index, total - 1)
    row = disagreements.iloc[index]
    key = (row["query_id"], str(row["novel_id"]))
    existing = reviews.get(key, {})
    verdict = judge_reason_for(row, sheet, reasons, "gpt-5.6-terra")

    st.subheader(f"分歧 {index + 1} / {total} · {row['query_id']} · 《{row['title_guess']}》")

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("#### 用户偏好")
        st.info(str(row.get("query", "")))
        st.markdown(f"**正向**: {str(row.get('wanted','')).replace('|', '、') or '—'}")
        st.markdown(f"**负向**: :red[{str(row.get('unwanted','')).replace('|', '、') or '—'}]")

        st.markdown("#### 两边的判断")
        a, b = st.columns(2)
        a.metric("你", int(row["human"]))
        b.metric("judge", int(row["judge"]))
        if verdict.get("reason") == "judge_parse_failed":
            st.error("judge 输出未能解析，这条被兜底成 0 分 —— 属技术故障，不是判断分歧。")
        elif verdict.get("reason"):
            st.markdown(f"**judge 理由**（置信度 {verdict.get('judge_confidence','?')}）")
            st.write(verdict["reason"])

        st.markdown("#### 你的裁定")
        choice = st.radio(
            "谁更合理",
            options=list(REVIEW_VERDICTS),
            format_func=lambda value: REVIEW_VERDICTS[value],
            index=list(REVIEW_VERDICTS).index(existing["verdict"]) if existing.get("verdict") in REVIEW_VERDICTS else 0,
            key=f"rv-{index}",
        )
        note = st.text_input("备注（可选）", value=str(existing.get("notes", "")), key=f"rn-{index}")

        save, skip = st.columns(2)
        if save.button("保存并下一条", type="primary", use_container_width=True):
            append_label(
                REVIEW_LOG_PATH,
                {
                    "query_id": row["query_id"],
                    "novel_id": str(row["novel_id"]),
                    "title_guess": str(row["title_guess"]),
                    "human_label": int(row["human"]),
                    "judge_label": int(row["judge"]),
                    "verdict": choice,
                    "notes": note,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            reviews[key] = {"verdict": choice, "notes": note}
            st.session_state.review_index = min(index + 1, total - 1)
            st.rerun()
        if skip.button("跳过", use_container_width=True):
            st.session_state.review_index = min(index + 1, total - 1)
            st.rerun()

    with right:
        st.markdown("#### 正文摘录（双方所见完全相同）")
        match = sheet[(sheet["query_id"] == row["query_id"]) & (sheet["novel_id"] == str(row["novel_id"]))]
        st.text_area("evidence", value=str(match.iloc[0]["evidence"]) if not match.empty else "", height=620, label_visibility="collapsed")


def main() -> None:
    st.set_page_config(page_title="偏好相关性标注", layout="wide")

    if not SHEET_PATH.exists():
        st.error(f"未找到标注表 {SHEET_PATH}。先运行 `uv run python scripts/10_annotation_sheet.py`。")
        return

    sheet = load_sheet(SHEET_PATH)
    mode = st.sidebar.radio("模式", [MODE_ANNOTATE, MODE_REVIEW], key="mode")
    st.sidebar.divider()
    if mode == MODE_REVIEW:
        render_review(sheet)
        return

    if "labels" not in st.session_state:
        st.session_state.labels = load_labels(LOG_PATH)
    if "index" not in st.session_state:
        st.session_state.index = next_unlabeled(sheet, st.session_state.labels, -1)

    labels = st.session_state.labels
    total = len(sheet)
    done = len(labels)

    with st.sidebar:
        st.header("进度")
        st.progress(done / total if total else 0.0)
        st.metric("已标注", f"{done} / {total}")
        st.caption(f"剩余约 {max(total - done, 0) * 2} 分钟")

        st.divider()
        st.header("跳转")
        current = min(max(st.session_state.index, 0), max(total - 1, 0))
        target = st.number_input("条目", min_value=1, max_value=max(total, 1), value=current + 1, step=1)
        if st.button("跳到该条", use_container_width=True):
            st.session_state.index = int(target) - 1
            st.rerun()
        if st.button("跳到下一条未标注", use_container_width=True):
            st.session_state.index = next_unlabeled(sheet, labels, st.session_state.index)
            st.rerun()

        st.divider()
        if st.button("导出到标注表", type="primary", use_container_width=True):
            count = write_filled_sheet(sheet, labels, SHEET_PATH)
            st.success(f"已写入 {SHEET_PATH}（{count} 条有标签）")
        st.caption("每次保存都会自动追加到 " + str(LOG_PATH) + "，导出只是把它合并回 CSV。")

        st.divider()
        st.caption(
            "评分只依据下方摘录，不要依据你对这本书的既有印象——"
            "judge 看到的也是同一份摘录，两边必须可比。"
        )

    index = min(max(st.session_state.index, 0), max(len(sheet) - 1, 0))
    st.session_state.index = index
    row = sheet.iloc[index]
    key = (row["query_id"], row["novel_id"])
    existing = labels.get(key, {})

    st.subheader(f"第 {index + 1} / {total} 条 · {row['query_id']}")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("#### 用户偏好")
        st.info(row["query"])
        wanted = str(row.get("wanted", "")).replace("|", "、")
        unwanted = str(row.get("unwanted", "")).replace("|", "、")
        st.markdown(f"**正向要求：** {wanted or '（未列出）'}")
        st.markdown(f"**负向排除：** :red[{unwanted or '（无）'}]")

        st.markdown("#### 候选")
        st.markdown(f"**{row['title_guess']}**")

        st.markdown("#### 评分")
        current = existing.get("relevance_label")
        choice = st.radio(
            "相关性",
            options=list(RELEVANCE_OPTIONS),
            format_func=lambda value: RELEVANCE_OPTIONS[value],
            index=list(RELEVANCE_OPTIONS).index(current) if current in RELEVANCE_OPTIONS else 0,
            key=f"rel-{index}",
        )
        violation = st.checkbox(
            "违反负向排除项",
            value=bool(existing.get("constraint_violation", False)),
            key=f"vio-{index}",
            help="仅当摘录里有证据表明它触犯了上面的负向排除项时才勾选。",
        )
        notes = st.text_input("备注（可选）", value=str(existing.get("notes", "")), key=f"note-{index}")

        save_col, skip_col = st.columns(2)
        if save_col.button("保存并下一条", type="primary", use_container_width=True):
            record = {
                "query_id": row["query_id"],
                "novel_id": row["novel_id"],
                "title_guess": row["title_guess"],
                "relevance_label": int(choice),
                "constraint_violation": bool(violation),
                "notes": notes,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
            }
            append_label(LOG_PATH, record)
            labels[key] = record
            st.session_state.index = next_unlabeled(sheet, labels, index)
            st.rerun()
        if skip_col.button("跳过", use_container_width=True):
            st.session_state.index = (index + 1) % total
            st.rerun()

        if existing:
            st.caption(f"已标注：相关性 {existing['relevance_label']}，违反约束 {existing['constraint_violation']}")

    with right:
        st.markdown("#### 正文摘录")
        st.caption(
            "从原文不同位置随机截取，不是完整作品。信息不足时按摘录能支持的判断给分，"
            "不要脑补。"
        )
        st.text_area("evidence", value=str(row["evidence"]), height=620, label_visibility="collapsed")


if __name__ == "__main__":
    main()
