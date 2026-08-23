"""Tests for the pure helpers in main.py — the tuning readout's logic.

main.py is mostly the capture loop and untestable without a camera, but the
metrics HUD is what the gesture thresholds get tuned against, so the part that
decides what it says is worth pinning down.
"""

import main
from gestures import FINGER_CURLED_RATIO, FINGER_EXTENDED_RATIO


def test_finger_state_names_the_two_sides_of_the_thresholds():
    assert main.finger_state(FINGER_EXTENDED_RATIO + 0.2) == "extended"
    assert main.finger_state(FINGER_CURLED_RATIO - 0.2) == "curled"


def test_finger_state_names_the_dead_band_between_them():
    """A ratio between the two thresholds is neither extended nor curled, so
    every gesture using that finger silently fails. It has no boolean of its
    own, which is the whole reason the readout gives it a name."""
    midway = (FINGER_EXTENDED_RATIO + FINGER_CURLED_RATIO) / 2

    assert main.finger_state(midway) == "dead"
    assert main.finger_state(FINGER_EXTENDED_RATIO) == "dead"
    assert main.finger_state(FINGER_CURLED_RATIO) == "dead"


def test_every_finger_state_has_a_mark_and_a_colour():
    for ratio in (FINGER_EXTENDED_RATIO + 0.2, FINGER_CURLED_RATIO - 0.2, 1.05):
        state = main.finger_state(ratio)
        assert state in main.FINGER_MARKS
        assert state in main.FINGER_STATE_COLORS


def test_metric_segments_labels_one_run_per_finger():
    import sys
    sys.path.insert(0, "tests")
    from test_gestures import _hand, CURLED

    ratios, pinch = main.metric_segments("hand", _hand(pinky=CURLED))

    # The label, then one coloured run per finger.
    assert len(ratios) == 1 + len(main.FINGER_NAMES)
    assert ratios[0][0].strip() == "hand"
    assert [run[0].split()[0] for run in ratios[1:]] == list(main.FINGER_NAMES)
    # The curled pinky is coloured differently from the extended index.
    assert ratios[1][1] == main.FINGER_STATE_COLORS["extended"]
    assert ratios[4][1] == main.FINGER_STATE_COLORS["curled"]
    assert "scale" in "".join(text for text, _ in pinch)


def test_metrics_readout_only_reports_a_draw_length_while_drawing():
    import sys
    sys.path.insert(0, "tests")
    from test_gestures import _hand

    measured = [("hand", _hand())]
    assert not any(
        "draw" in text
        for line in main.metrics_readout(measured, None)
        for text, _ in line
    )

    nocked = ((300, 200), (100, 200), 50.0)
    assert any(
        "draw" in text
        for line in main.metrics_readout(measured, nocked)
        for text, _ in line
    )
