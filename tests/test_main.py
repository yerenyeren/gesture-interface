"""Tests for the pure helpers in main.py.

main.py is mostly the capture loop and untestable without a camera, but two
parts of it are worth pinning down: the metrics HUD, which is what the gesture
thresholds get tuned against, and the guard that keeps a gesture click from
landing on the app's own window.
"""

from unittest.mock import patch

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


def test_a_click_landing_on_the_app_window_is_recognised():
    """The cursor follows the palm across the whole screen, so it crosses this
    app's own window constantly. A click there raises or minimises the view the
    gestures are being read from, which is why those clicks are dropped."""
    rect = (100, 100, 640, 480)

    assert main.is_over_window((400, 300), rect) is True
    assert main.is_over_window((100, 100), rect) is True


def test_a_click_away_from_the_window_passes():
    rect = (100, 100, 640, 480)

    assert main.is_over_window((99, 300), rect) is False
    assert main.is_over_window((740, 300), rect) is False
    assert main.is_over_window((400, 580), rect) is False


def test_an_unknown_window_or_cursor_guards_nothing():
    """An unreported window should cost the safety net, never the click."""
    assert main.is_over_window((400, 300), None) is False
    assert main.is_over_window(None, (100, 100, 640, 480)) is False


def test_the_guard_reaches_above_the_image_to_cover_the_title_bar():
    """The minimise button is in the decorations, which sit outside the rect
    OpenCV reports — guarding the image alone would miss the button that
    started this."""
    with patch("main.cv2") as mock_cv2:
        mock_cv2.getWindowImageRect.return_value = (100, 200, 640, 480)
        mock_cv2.error = Exception
        rect = main.guarded_window_rect()

    x, y, width, height = rect
    assert y < 200 and x < 100
    assert main.is_over_window((400, 199), rect) is True
    assert width >= 640 and height >= 480


def test_a_window_the_backend_will_not_report_guards_nothing():
    with patch("main.cv2") as mock_cv2:
        mock_cv2.getWindowImageRect.return_value = (0, 0, 0, 0)
        mock_cv2.error = Exception

        assert main.guarded_window_rect() is None
