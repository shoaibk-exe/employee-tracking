"""Structured VLM feature parse / majority vote / compatibility (no GPU)."""

from core.vlm.vlm_features import (
    FEATURE_KEYS,
    HEAD_ATTR_KEYS,
    SINGLE_ATTR_PROMPTS,
    UNKNOWN,
    MoondreamFeatureExtractor,
    QwenVlFeatureExtractor,
    VlmFeatureConfig,
    consensus_embedding,
    create_feature_extractor,
    expand_person_bbox,
    features_compatible,
    head_crop_from_person,
    head_crop_usable,
    majority_vote,
    parse_binary,
    parse_color,
    parse_feature_text,
    parse_gender,
    parse_hair,
    parse_single_attr,
)
import numpy as np


def test_create_feature_extractor_routes_qwen_and_moondream():
    moon = create_feature_extractor(VlmFeatureConfig(model_id="vikhyatk/moondream2"))
    qwen = create_feature_extractor(
        VlmFeatureConfig(model_id="Qwen/Qwen2.5-VL-3B-Instruct", revision="2025-06-21")
    )
    assert isinstance(moon, MoondreamFeatureExtractor)
    assert not isinstance(moon, QwenVlFeatureExtractor)
    assert isinstance(qwen, QwenVlFeatureExtractor)
    # Moondream-dated revision must not stick on Qwen.
    assert qwen.config.revision == ""


def test_parse_feature_text_fixed_keys():
    text = """
    gender: Male
    glasses: yes
    top_color: navy
    bottom_color: black
    hair: short
    hat: no
    """
    f = parse_feature_text(text)
    assert set(f) == set(FEATURE_KEYS)
    assert f["gender"] == "male"
    assert f["glasses"] == "yes"
    assert f["top_color"] == "blue"  # navy → blue
    assert f["bottom_color"] == "black"
    assert f["hair"] == "short"
    assert f["hat"] == "no"


def test_format_example_clothing_copy_scrubbed():
    """Multi-option / menu echoes from old multi-attr prompts → unknown clothing."""
    f = parse_feature_text(
        "gender: male\n"
        "glasses: yes\n"
        "top_color: pink or yellow\n"
        "bottom_color: yellow / orange\n"
        "hair: long or short\n"
        "hat: yes only for a real hat/cap\n"
    )
    assert f["gender"] == "male"
    assert f["glasses"] == "yes"
    assert f["top_color"] == UNKNOWN
    assert f["bottom_color"] == UNKNOWN
    assert f["hair"] == UNKNOWN
    assert f["hat"] == UNKNOWN


def test_parse_binary_never_yes_on_no_not_wearing():
    """Classic bug: keyword 'wearing' used to force yes; must be unknown now."""
    assert parse_binary("no, not wearing") == UNKNOWN
    assert parse_binary("no") == "no"
    assert parse_binary("yes") == "yes"
    assert parse_binary("true") == "yes"
    assert parse_binary("false") == "no"
    assert parse_binary("wearing a hat") == UNKNOWN
    assert parse_binary("has glasses") == UNKNOWN


def test_hat_no_not_wearing_line():
    f = parse_feature_text("hat: no, not wearing")
    assert f["hat"] == UNKNOWN  # strict: not a bare yes/no


def test_prompt_echo_hat_and_glasses_become_unknown():
    # Old prompt line that started with "yes only for a real hat..."
    f = parse_feature_text(
        "hat: yes only for a real hat/cap/hijab; headset/earphones = no\n"
        "glasses: yes OR no OR unknown\n"
        "gender: male OR female OR unknown\n"
    )
    assert f["hat"] == UNKNOWN
    assert f["glasses"] == UNKNOWN
    assert f["gender"] == UNKNOWN


def test_prompt_echo_equals_list_form():
    """Moondream often echoes 'gender = male, female, unknown' — must stay unknown."""
    f = parse_feature_text(
        "gender = male, female, unknown\n"
        "glasses = yes, no, unknown\n"
        "top_color = one color name or unknown\n"
        "bottom_color = black\n"
        "hair = short\n"
        "hat = no\n"
    )
    assert f["gender"] == UNKNOWN
    assert f["glasses"] == UNKNOWN
    assert f["top_color"] == UNKNOWN
    assert f["bottom_color"] == "black"
    assert f["hair"] == "short"
    assert f["hat"] == "no"


def test_prompt_echo_allowed_values_block():
    f = parse_feature_text(
        "top_color: one color name or unknown\n"
        "bottom_color: black\n"
        "hair: short\n"
    )
    assert f["top_color"] == UNKNOWN
    assert f["bottom_color"] == "black"
    assert f["hair"] == "short"


def test_conflicting_colors_unknown():
    # Neutral + one chromatic → keep chromatic (CCTV "black or brown" / "blue gray").
    assert parse_color("black or brown") == "brown"
    assert parse_color("navy blue") == "blue"  # both alias to blue
    assert parse_color("navy blue shirt") == "blue"
    assert parse_color("red") == "red"
    assert parse_color("grey") == "gray"
    assert parse_color("teal green") == "green"  # teal+green → same canon
    assert parse_color("blue and red") == UNKNOWN  # two chromatics still conflict
    assert parse_color("unknown") == UNKNOWN
    assert parse_color("") == UNKNOWN


def test_invalid_and_partial_values():
    assert parse_gender("maybe") == UNKNOWN
    assert parse_gender("man") == UNKNOWN  # strict; only male/female
    assert parse_gender("male") == "male"
    assert parse_hair("medium") == UNKNOWN
    assert parse_hair("short") == "short"
    f = parse_feature_text(
        "gender:\n"
        "glasses: \n"
        "top_color: ???\n"
        "bottom_color: black\n"
        "hair: buzz cut\n"
        "hat: no\n"
    )
    assert f["gender"] == UNKNOWN
    assert f["glasses"] == UNKNOWN
    assert f["top_color"] == UNKNOWN
    assert f["bottom_color"] == "black"
    assert f["hair"] == UNKNOWN  # "buzz cut" not exact short/long/bald
    assert f["hat"] == "no"


def test_majority_vote_and_unknown_on_tie():
    samples = [
        {"gender": "male", "glasses": "yes", "top_color": "blue",
         "bottom_color": "black", "hair": "short", "hat": "no"},
        {"gender": "male", "glasses": "yes", "top_color": "blue",
         "bottom_color": "black", "hair": UNKNOWN, "hat": "no"},
        {"gender": "male", "glasses": "no", "top_color": "blue",
         "bottom_color": "black", "hair": "long", "hat": "no"},
        {"gender": "male", "glasses": "yes", "top_color": "blue",
         "bottom_color": "black", "hair": "short", "hat": "no"},
        {"gender": "male", "glasses": "yes", "top_color": "blue",
         "bottom_color": "black", "hair": "short", "hat": "no"},
    ]
    voted = majority_vote(samples, majority_min=2)
    assert voted["gender"] == "male"
    assert voted["glasses"] == "yes"
    assert voted["top_color"] == "blue"
    assert voted["hair"] == "short"
    # All unknown → stays unknown
    unk = majority_vote([{k: UNKNOWN for k in FEATURE_KEYS}] * 5, majority_min=2)
    assert all(v == UNKNOWN for v in unk.values())


def test_features_compatible_skips_unknown():
    a = {k: UNKNOWN for k in FEATURE_KEYS}
    a.update({"gender": "male", "top_color": "blue", "glasses": "yes"})
    b = {k: UNKNOWN for k in FEATURE_KEYS}
    b.update({"gender": "male", "top_color": "blue", "hair": "short"})  # glasses unknown on b
    ok, n_comp, n_agree = features_compatible(a, b, min_comparable=2)
    assert ok and n_comp == 2 and n_agree == 2  # gender + top_color; glasses skipped

    b["top_color"] = "red"
    ok, _, _ = features_compatible(a, b, min_comparable=2)
    assert not ok  # conflict on comparable key


def test_consensus_embedding_picks_cluster():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    c = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    outlier = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    pick = consensus_embedding([a, b, c, outlier])
    assert pick is not None
    # Should be one of the cluster, not the outlier.
    assert float(pick @ outlier) < float(pick @ a)


def test_single_attr_prompts_have_no_format_menu():
    """Prompts stay short (no pipe menus / forced colour palettes)."""
    for key, prompt in SINGLE_ATTR_PROMPTS.items():
        low = prompt.lower()
        assert "|" not in prompt
        assert "allowed" not in low
        assert "example" not in low
        assert "exactly one word from:" not in low
        assert key in FEATURE_KEYS
        if key in {"top_color", "bottom_color"}:
            assert "one color word only" in low
            assert "ignore other people" in low
        else:
            assert "one word only" in low
    assert "top_pattern" in SINGLE_ATTR_PROMPTS
    assert "plaid" in SINGLE_ATTR_PROMPTS["top_pattern"].lower()


def test_parse_single_attr_accepts_short_answers():
    assert parse_single_attr("gender", "male") == "male"
    assert parse_single_attr("glasses", "yes") == "yes"
    assert parse_single_attr("top_color", "Gray") == "gray"
    assert parse_single_attr("bottom_color", "black") == "black"
    assert parse_single_attr("top_pattern", "plaid") == "plaid"
    assert parse_single_attr("top_pattern", "solid") == "solid"
    assert parse_single_attr("hair", "short") == "short"
    assert parse_single_attr("hat", "no") == "no"


def test_parse_pattern_aliases():
    from core.vlm.vlm_features import parse_pattern
    assert parse_pattern("checkered") == "plaid"
    assert parse_pattern("camo") == "other"
    assert parse_pattern("plain") == "solid"
    assert parse_pattern("maybe") == UNKNOWN


def test_sanitize_top_color_rejects_false_blue_on_bright_shirt():
    from core.vlm.vlm_features import sanitize_top_color
    bright = np.full((80, 60, 3), 200, dtype=np.uint8)  # bright low-sat → gray/white
    assert sanitize_top_color("blue", bright) in {"gray", "white", "beige"}
    dark = np.full((80, 60, 3), 40, dtype=np.uint8)
    assert sanitize_top_color("beige", dark) == UNKNOWN


def test_clothing_crop_quality_prefers_mid_bright_large():
    from core.vlm.vlm_features import clothing_crop_quality
    tiny = np.zeros((20, 20, 3), dtype=np.uint8)
    good = np.full((90, 70, 3), 120, dtype=np.uint8)
    assert clothing_crop_quality(tiny) < clothing_crop_quality(good)
    assert clothing_crop_quality(good) >= 0.35


def test_stable_color_unique_plurality_two_votes():
    """Two agreeing unique colour votes commit (clear-frame path)."""
    from core.vlm.vlm_features import add_feature_votes, stable_features
    votes: dict = {}
    add_feature_votes(votes, {"top_color": "blue", "top_pattern": "plaid"})
    add_feature_votes(votes, {"top_color": "blue", "top_pattern": "plaid"})
    out = stable_features(votes, min_votes=2, min_share=0.55, color_min_votes=2,
                           color_min_share=0.55)
    assert out["top_color"] == "blue"
    assert out["top_pattern"] == "plaid"


def test_parse_single_attr_rejects_prompt_echo():
    assert parse_single_attr("gender", "male, female, unknown") == UNKNOWN
    assert parse_single_attr("glasses", "yes | no | unknown") == UNKNOWN
    assert parse_single_attr(
        "top_color", "black|white|gray|blue|red|green|yellow|orange|brown|beige|pink|purple|unknown",
    ) == UNKNOWN
    assert parse_single_attr("hat", "yes only for a real hat/cap") == UNKNOWN


def test_charcoal_normalizes_to_gray():
    assert parse_color("charcoal") == "gray"
    assert parse_color("dark gray") == "gray"


def test_head_crop_from_person_preserves_aspect_no_stretch():
    person = np.zeros((420, 97, 3), dtype=np.uint8)
    person[:50, :] = 200
    head = head_crop_from_person(person, height_fraction=0.38)
    assert head is not None
    assert head.shape[0] == int(round(420 * 0.38))
    assert head.shape[1] == 97  # width unchanged — no horizontal stretch
    assert head_crop_usable(head, min_side_px=24)
    tiny = np.zeros((20, 10, 3), dtype=np.uint8)
    assert not head_crop_usable(head_crop_from_person(tiny), min_side_px=24)


def test_expand_person_bbox_clips_to_frame():
    x1, y1, x2, y2 = expand_person_bbox(
        (10, 10, 50, 200), (100, 240),
        pad_x_fraction=0.5, pad_top_fraction=0.2,
    )
    assert x1 == 0 and y1 == 0
    assert x2 <= 100 and y2 <= 240
    assert x2 > x1 and y2 > y1


def test_accessory_keys_use_head_prompts():
    assert set(HEAD_ATTR_KEYS) == {"glasses", "hat", "hair"}
    assert "clearly visible" in SINGLE_ATTR_PROMPTS["hat"].lower()
    assert "eyeglass" in SINGLE_ATTR_PROMPTS["glasses"].lower()
    assert "looking down" in SINGLE_ATTR_PROMPTS["glasses"].lower()


def test_majority_vote_accepts_single_unique_sample():
    """Static employees may only ever yield one distinct crop."""
    sample = {
        "gender": "male", "glasses": "no", "top_color": "gray",
        "bottom_color": "black", "hair": "short", "hat": "no",
    }
    voted = majority_vote([sample], majority_min=2)
    assert voted["glasses"] == "no"
    assert voted["hat"] == "no"


def test_majority_vote_single_sample_not_trusted_for_identity():
    sample = {
        "gender": "male", "glasses": "no", "top_color": "brown",
        "bottom_color": "black", "hair": "short", "hat": "no",
    }
    voted = majority_vote([sample], majority_min=2, allow_single=False)
    assert all(v == UNKNOWN for v in voted.values())


def test_stable_features_later_majority_corrects_early_wrong_value():
    from collections import Counter

    from core.vlm.vlm_features import add_feature_votes, stable_features

    votes: dict[str, Counter] = {}
    add_feature_votes(votes, {"top_color": "brown", "bottom_color": "black"})
    first = stable_features(votes, min_votes=2, min_share=0.6)
    assert first["top_color"] == UNKNOWN  # one occluded crop is not committed
    add_feature_votes(votes, {"top_color": "red", "bottom_color": "black"})
    tied = stable_features(votes, min_votes=2, min_share=0.6)
    assert tied["top_color"] == UNKNOWN  # 1 brown vs 1 red
    assert tied["bottom_color"] == "black"
    add_feature_votes(votes, {"top_color": "red"})
    add_feature_votes(votes, {"top_color": "red"})
    assert stable_features(votes, min_votes=2, min_share=0.6)["top_color"] == "red"


def test_stable_features_keeps_current_for_keys_without_votes():
    from core.vlm.vlm_features import stable_features

    out = stable_features({}, {"gender": "female"}, min_votes=2)
    assert out["gender"] == "female"
    assert out["top_color"] == UNKNOWN


def test_feature_agreement_counts_all_conflicts():
    from core.vlm.vlm_features import feature_agreement

    a = {"gender": "male", "top_color": "red", "bottom_color": "black", "hat": UNKNOWN}
    b = {"gender": "male", "top_color": "teal", "bottom_color": "blue", "hat": "no"}
    assert feature_agreement(a, b) == (3, 1)


def test_parse_color_dark_green_and_blue_gray():
    from core.vlm.vlm_features import parse_color

    assert parse_color("dark") == UNKNOWN
    assert parse_color("dark green") == "green"
    assert parse_color("blue gray") == "blue"
    assert parse_color("grey blue") == "blue"
    assert parse_color("Tan") == "beige"
    assert parse_color("olive") == "green"
    assert parse_color("red blue") == UNKNOWN  # two chromatics


def test_torso_and_lower_crops():
    from core.vlm.vlm_features import (
        clothing_crop_usable,
        lower_crop_from_person,
        torso_crop_from_person,
    )

    person = np.full((200, 80, 3), 120, dtype=np.uint8)
    torso = torso_crop_from_person(person, y0_fraction=0.18, y1_fraction=0.58)
    lower = lower_crop_from_person(person, y0_fraction=0.50, y1_fraction=0.95)
    assert torso is not None and torso.shape[0] < person.shape[0]
    assert lower is not None and lower.shape[0] < person.shape[0]
    assert clothing_crop_usable(torso, min_side_px=40)
    assert not clothing_crop_usable(np.zeros((80, 80, 3), dtype=np.uint8), min_mean=22.0)


def test_stable_features_color_share_softer_than_other_keys():
    from collections import Counter

    from core.vlm.vlm_features import add_feature_votes, stable_features

    votes: dict[str, Counter] = {}
    # 3 green + 2 yellow + 1 brown = green at 50%
    for c in ("green", "green", "green", "yellow", "yellow", "brown"):
        add_feature_votes(votes, {"top_color": c, "gender": "male"})
    out = stable_features(votes, min_votes=2, min_share=0.6, color_min_share=0.5)
    assert out["top_color"] == "green"
    assert out["gender"] == "male"  # 6/6 male, share 1.0
    # gender would fail at 0.6 if split; colour commits at 0.5


def test_cv_color_assist_upgrades_black_to_green():
    green = np.zeros((80, 60, 3), dtype=np.uint8)
    green[:, :, 1] = 150
    green[:, :, 0] = 40
    green[:, :, 2] = 40
    from core.vlm.vlm_features import estimate_clothing_color_cv, refine_color_with_cv

    assert estimate_clothing_color_cv(green) == "green"
    assert refine_color_with_cv("black", "green") == "green"
    assert refine_color_with_cv("black", "blue") == "blue"
    assert refine_color_with_cv("red", "green") == "red"  # never override concrete VLM
    assert refine_color_with_cv("black", "red") == "black"  # ignore warm CCTV cast
    assert refine_color_with_cv("brown", "beige") == "brown"


def test_stable_features_rejects_warm_top_when_cool_outvotes():
    """Tied red vs green: cool coalition wins (neighbour contamination)."""
    from collections import Counter

    from core.vlm.vlm_features import add_feature_votes, stable_features

    votes: dict[str, Counter] = {}
    for c in ("red", "red", "red", "red", "red",
              "green", "green", "green", "green", "green",
              "beige", "beige", "beige"):
        add_feature_votes(votes, {"top_color": c})
    out = stable_features(
        votes, min_votes=2, min_share=0.6, color_min_share=0.55, color_min_votes=2,
    )
    assert out["top_color"] == "green"

    # Clear red plurality (true red tee) must stay red even with cool noise.
    votes2: dict[str, Counter] = {}
    for c in ("red", "red", "red", "red", "red",
              "green", "green", "green", "green",
              "beige", "beige", "beige", "beige"):
        add_feature_votes(votes2, {"top_color": c})
    out2 = stable_features(
        votes2, min_votes=2, min_share=0.6, color_min_share=0.55, color_min_votes=2,
    )
    assert out2["top_color"] == "red"


def test_sanitize_top_color_drops_beige_on_dark_torso():
    from core.vlm.vlm_features import sanitize_top_color

    dark = np.full((60, 40, 3), 40, dtype=np.uint8)
    light = np.full((60, 40, 3), 160, dtype=np.uint8)
    assert sanitize_top_color("beige", dark) == UNKNOWN
    assert sanitize_top_color("beige", light) == "beige"
    assert sanitize_top_color("red", dark) == "red"
