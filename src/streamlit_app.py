"""Streamlit demo application for explainable Chinese web novel discovery."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Streamlit's file watcher can inspect torch/transformers lazy modules and
# trigger misleading import errors during model-heavy app runs.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# `streamlit run src/streamlit_app.py` places `src/` on sys.path. That makes
# `src/profile.py` shadow Python's stdlib `profile` module, which breaks
# torch -> cProfile imports during Transformers/SentenceTransformer loading.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != APP_DIR]

import streamlit as st

from src.app_pipeline import PipelineResult, PipelineSettings, load_profiles_for_app, run_discovery_pipeline, validate_runtime_files
from src.embed import DEFAULT_EMBEDDING_MODEL, load_embedding_model
from src.llm_matcher import DEFAULT_LLM_MODEL, create_transformers_matcher
from src.search import load_faiss_index, load_id_map
from src.vector_index import DEFAULT_ID_MAP_PATH, DEFAULT_INDEX_PATH, DEFAULT_PROFILES_PATH


EXAMPLE_QUERIES = [
    "凡人流 仙侠 慢热 理性主角 不系统",
    "赛博朋克 科幻 群像 智能体 社会冲突",
    "都市异能 节奏快 主角冷静 不无脑爽文",
    "西幻 冒险 小队成长 世界观完整",
    "克苏鲁 悬疑 慢热 氛围感强",
]


@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embedding_model(model_name: str, device: str):
    return load_embedding_model(model_name, device=None if device == "auto" else device)


@st.cache_resource(show_spinner="Loading local Qwen LLM...")
def cached_llm_matcher(model_name: str, device: str):
    return create_transformers_matcher(model_name=model_name, device=None if device == "auto" else device, max_new_tokens=256)


@st.cache_resource(show_spinner="Loading FAISS index...")
def cached_faiss_index(index_path: str):
    return load_faiss_index(Path(index_path))


@st.cache_data(show_spinner="Loading id map...")
def cached_id_map(id_map_path: str):
    return load_id_map(Path(id_map_path))


@st.cache_data(show_spinner="Loading novel profiles...")
def cached_profiles(profiles_path: str):
    return load_profiles_for_app(Path(profiles_path))


def initialize_query_state() -> None:
    if "query" not in st.session_state:
        st.session_state.query = EXAMPLE_QUERIES[0]


def set_example_query(query: str) -> None:
    st.session_state.query = query


def score_text(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def list_text(values: Any) -> str:
    if not values:
        return "None"
    if isinstance(values, list):
        return "、".join(str(item) for item in values) if values else "None"
    return str(values)


def render_pipeline_summary(result: PipelineResult) -> None:
    settings = result.settings
    st.subheader("Pipeline Summary")
    cols = st.columns(3)
    cols[0].metric("candidate-k", settings.candidate_k)
    cols[1].metric("llm-candidate-k", result.resolved_llm_candidate_k)
    cols[2].metric("top-k", settings.top_k)
    st.write(
        {
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "device": result.resolved_device or "cpu/auto",
            "total_runtime_seconds": round(result.timing.get("total_runtime", 0.0), 2),
            "rerank_cache_hits": int(result.timing.get("rerank_cache_hits", 0)),
            "rerank_cache_misses": int(result.timing.get("rerank_cache_misses", 0)),
        }
    )


def render_expanded_queries(result: PipelineResult) -> None:
    st.subheader("Expanded Retrieval Queries")
    for idx, expanded in enumerate(result.expanded_queries, start=1):
        st.markdown(f"{idx}. `[{expanded.source}, weight={expanded.weight:.2f}]` {expanded.text}")
    with st.expander("Retrieval summary"):
        st.json(result.retrieval_summary)


def render_recommendation_card(candidate: dict[str, Any], explanation: Any) -> None:
    title = explanation.title_guess or candidate.get("title_guess", "Unknown")
    with st.container(border=True):
        st.markdown(f"### {explanation.final_rank}. 《{title.strip('《》')}》")
        cols = st.columns(4)
        cols[0].metric("Final", score_text(explanation.source_scores.get("final_score")))
        cols[1].metric("Semantic", score_text(explanation.source_scores.get("semantic_score")))
        cols[2].metric("LLM Match", score_text(explanation.source_scores.get("llm_match_score")))
        cols[3].metric("Confidence", explanation.confidence)

        st.markdown("**Why recommended**")
        st.write(explanation.why_recommended)

        st.markdown("**Matched preferences**")
        st.write(list_text(explanation.matched_preferences))

        risks = explanation.possible_risks or candidate.get("violated_preferences") or candidate.get("risk_flags")
        st.markdown("**Possible risks / violated preferences**")
        st.write(list_text(risks))

        meta_cols = st.columns(2)
        meta_cols[0].write(f"Retrieval sources: {list_text(candidate.get('retrieval_sources'))}")
        meta_cols[1].write(f"Matched query count: {candidate.get('matched_query_count', '-')}")

        st.markdown("**Evidence**")
        for item in explanation.evidence or ["Evidence is limited to sampled profile text and Stage 4 fields."]:
            st.write(f"- {item}")

        st.markdown("**User takeaway**")
        st.write(explanation.user_takeaway or "Use this result as a candidate to inspect further.")


def main() -> None:
    st.set_page_config(page_title="AI-Powered Chinese Web Novel Discovery System", layout="wide")
    initialize_query_state()

    st.title("AI-Powered Chinese Web Novel Discovery System")
    st.caption("Semantic retrieval + local Qwen LLM reranking + grounded explanation reports.")

    with st.sidebar:
        st.header("Example Queries")
        for query in EXAMPLE_QUERIES:
            st.button(query, key=f"example-{query}", on_click=set_example_query, args=(query,))

        st.header("Advanced Settings")
        embedding_model = st.text_input("Embedding model", DEFAULT_EMBEDDING_MODEL)
        llm_model = st.text_input("LLM model", DEFAULT_LLM_MODEL)
        device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
        explanation_profile_max_chars = st.number_input("Explanation profile max chars", min_value=300, max_value=4000, value=1200, step=100)
        llm_profile_max_chars = st.number_input("Rerank profile max chars", min_value=300, max_value=4000, value=1200, step=100)
        use_query_expansion = st.checkbox("Use query expansion", value=True)
        use_domain_hints = st.checkbox("Use domain hints", value=True)
        use_cache = st.checkbox("Use LLM rerank cache", value=True)

    st.subheader("Input")
    query = st.text_area(
        "Preference query",
        key="query",
        height=110,
        placeholder="凡人流 仙侠 慢热 理性主角 不系统",
    )

    col1, col2, col3 = st.columns(3)
    candidate_k = col1.number_input("candidate-k", min_value=10, max_value=1000, value=50, step=10)
    llm_candidate_k = col2.number_input("llm-candidate-k", min_value=1, max_value=50, value=3, step=1)
    top_k = col3.number_input("top-k", min_value=1, max_value=20, value=3, step=1)

    with st.expander("Index and profile paths"):
        index_path = st.text_input("FAISS index", str(DEFAULT_INDEX_PATH))
        id_map_path = st.text_input("ID map", str(DEFAULT_ID_MAP_PATH))
        profiles_path = st.text_input("Profiles parquet", str(DEFAULT_PROFILES_PATH))

    if not st.button("Generate Recommendations", type="primary"):
        st.info("Use fast settings first: candidate-k 50, llm-candidate-k 3, top-k 3.")
        return

    settings = PipelineSettings(
        candidate_k=int(candidate_k),
        llm_candidate_k=int(llm_candidate_k),
        top_k=int(top_k),
        embedding_model=embedding_model,
        llm_model=llm_model,
        device=device,
        explanation_profile_max_chars=int(explanation_profile_max_chars),
        llm_profile_max_chars=int(llm_profile_max_chars),
        use_query_expansion=use_query_expansion,
        use_domain_hints=use_domain_hints,
        use_cache=use_cache,
        index_path=Path(index_path),
        id_map_path=Path(id_map_path),
        profiles_path=Path(profiles_path),
    )

    missing = validate_runtime_files(settings)
    if missing:
        for message in missing:
            st.error(message)
        return
    if not query.strip():
        st.warning("Please enter a recommendation query.")
        return

    progress = st.progress(0.0)
    status = st.status("Starting recommendation pipeline...", expanded=True)

    def progress_callback(stage: str, message: str, value: float | None) -> None:
        status.write(f"[{stage}] {message}")
        if value is not None:
            progress.progress(min(max(value, 0.0), 1.0))

    try:
        with status:
            progress_callback("load", "Loading cached models, index, id map, and profiles.", 0.02)
            embedder = cached_embedding_model(settings.embedding_model, settings.device)
            matcher = cached_llm_matcher(settings.llm_model, settings.device)
            index = cached_faiss_index(str(settings.index_path))
            id_map = cached_id_map(str(settings.id_map_path))
            profile_lookup = cached_profiles(str(settings.profiles_path))
            result = run_discovery_pipeline(
                query=query,
                settings=settings,
                embedder=embedder,
                matcher=matcher,
                index=index,
                id_map=id_map,
                profile_lookup=profile_lookup,
                progress_callback=progress_callback,
            )
            status.update(label="Recommendation pipeline complete.", state="complete", expanded=False)
    except RuntimeError as exc:
        text = str(exc)
        if "out of memory" in text.lower():
            st.error("CUDA out of memory. Try CPU, lower llm-candidate-k/top-k, or restart the app to clear GPU memory.")
        else:
            st.error(f"Runtime error: {text}")
            with st.expander("Show traceback"):
                st.code(traceback.format_exc())
        return
    except Exception as exc:
        st.error(f"Recommendation failed: {exc}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())
        return

    render_pipeline_summary(result)
    render_expanded_queries(result)

    st.subheader("Recommendations")
    for candidate, explanation in zip(result.ranked_candidates, result.explanations, strict=False):
        render_recommendation_card(candidate, explanation)

    st.subheader("Export")
    st.download_button("Download Markdown Report", result.report_markdown, file_name="recommendation_report.md", mime="text/markdown")
    st.download_button("Download Raw JSON", result.report_json, file_name="recommendation_report.json", mime="application/json")

    with st.expander("Show raw candidate data"):
        st.json(result.raw_dict())


if __name__ == "__main__":
    main()
