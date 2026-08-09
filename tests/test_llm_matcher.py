

def test_the_prompt_does_not_reveal_the_retrieval_score() -> None:
    """The teacher echoes any score it is shown, so it is not shown one.

    Holding candidates fixed and varying only the displayed retrieval score moved
    the median output to exactly the injected value (0.05 -> 0.05, 0.50 -> 0.50)
    and the mean from 0.123 to 0.655. rank.py blends 0.40*semantic with
    0.50*llm_match; an llm_match that restates semantic makes that blend count one
    signal twice.
    """

    from src.llm_matcher import build_match_prompt

    prompt = build_match_prompt(
        query="凡人流 仙侠 慢热 不系统",
        candidate={"title_guess": "《测试》", "score": 0.123456},
        profile_text="标题：《测试》\n一个修仙故事。",
    )

    assert "Semantic score" not in prompt
    assert "0.123456" not in prompt
    # The profile and query must still be there — this is a removal, not a rewrite.
    assert "凡人流 仙侠 慢热 不系统" in prompt
    assert "一个修仙故事" in prompt
