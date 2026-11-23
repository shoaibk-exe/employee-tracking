"""VLM appearance token matching (no GPU / model download required)."""

from core.vlm.vlm_describer import appearance_similarity, best_description_match


def test_conflicting_top_colors_do_not_match():
    blue = "person wearing a blue shirt and black pants"
    red = "person wearing a red jacket and black pants"
    assert appearance_similarity(blue, red) < 1.0
    pid, score, _ = best_description_match(red, {"P01": blue})
    assert pid is None and score < 2.5


def test_matching_top_colors_reuse_id():
    a = "person wearing a navy hoodie and gray jeans with a black backpack"
    b = "person in a navy hoodie, grey jeans, black backpack"
    pid, score, second = best_description_match(b, {"P01": a, "P02": "red dress white shoes"})
    assert pid == "P01" and score >= 2.5 and score > second


def test_live_exclusive_candidates_only_free():
    # Caller passes only free IDs; ensure second person description enrolls as no-match.
    blue = "blue top black bottoms"
    red = "red top black bottoms"
    pid, _, _ = best_description_match(red, {"P01": blue})
    assert pid is None
