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
    """Return the next index without a label, wrapping around."""

    total = len(sheet)
    for offset in range(1, total + 1):
        index = (start + offset) % total
        row = sheet.iloc[index]
        if (row["query_id"], row["novel_id"]) not in labels:
            return index
    return start


def main() -> None:
    st.set_page_config(page_title="偏好相关性标注", layout="wide")

    if not SHEET_PATH.exists():
        st.error(f"未找到标注表 {SHEET_PATH}。先运行 `uv run python scripts/10_annotation_sheet.py`。")
        return

    sheet = load_sheet(SHEET_PATH)
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
        target = st.number_input("条目", min_value=1, max_value=total, value=st.session_state.index + 1, step=1)
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

    index = st.session_state.index
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
