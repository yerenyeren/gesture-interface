"""Tests for the pure helpers in main.py.

main.py is mostly the capture loop and untestable without a camera, but four
parts of it are worth pinning down:
- the metrics HUD, which the gesture thresholds get tuned against;
- the guard that keeps a gesture click from landing on the app's own window;
- the screen rebuild, the one path out of the dragging state that does not run
  the loop's own release call;
- whether there is an arrow on the bowstring, which both bows are told rather
  than left to decide.
"""

from unittest.mock import MagicMock, patch

import main
from gestures import FINGER_CURLED_RATIO, FINGER_EXTENDED_RATIO
from mouse_control import MouseController


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


@patch("main.HorseBow")
@patch("main.DesktopOverlay")
@patch("main.MouseController")
@patch("mouse_control.pyautogui")
def test_a_screen_rebuild_releases_the_button_it_is_about_to_orphan(
    mock_pyautogui, mock_controller, mock_overlay, mock_bow
):
    """A monitor change replaces the MouseController, and the replacement starts
    with the button up. So a rebuild during a drag left the real button held
    with nothing left that knew it — not even the `finally` on the way out,
    which by then releases the *new* controller and finds nothing to do. Every
    other path out of ACTIVE releases; this is the one that does not go through
    them."""
    mock_pyautogui.size.return_value = (2560, 1440)
    # The outgoing controller is the real class on purpose: a mock would report
    # back whatever the code did to it rather than what a held button means.
    mouse = MouseController(frame_width=640, frame_height=480)
    mouse.press()
    # Its replacement needs real numbers — OverlayGeometry.size_scale takes a
    # min() across the two axis ratios, which bare mocks cannot be compared on.
    mock_controller.return_value.screen_width = 1920
    mock_controller.return_value.screen_height = 1080

    main.rebuild_for_screen(mouse, MagicMock(), (1920, 1080), 640, 480)

    mock_pyautogui.mouseUp.assert_called_once()
    assert mouse.is_pressed is False


def _pose(draw, scale=50.0):
    """An archery pose (grip, nock, scale) drawn `draw` hand scales long."""
    return (100 + draw * scale, 200), (100, 200), scale


def _feed(on_string, *draws):
    """One frame per draw length, None for a frame without the pose."""
    return [on_string.update(None if draw is None else _pose(draw)) for draw in draws]


def test_one_overdrawn_frame_does_not_drop():
    """Landmark jitter puts the odd frame past the limit during a steady draw."""
    on_string = main.ArrowOnString()

    assert _feed(on_string, 2.0, 3.5, 2.0, 3.5, 2.0) == [False] * 5
    assert on_string.loaded is True


def test_two_overdrawn_frames_drop_the_arrow_once():
    on_string = main.ArrowOnString()

    assert _feed(on_string, 2.0, 3.5, 3.5, 3.5, 3.5) == [False, False, True, False, False]
    assert on_string.loaded is False


def test_a_dropped_arrow_is_not_loosed_and_renocks_on_release():
    """Opening the hand on an empty bow fires nothing. The next time the pose
    is struck it starts with an arrow on the string, and the arrow stays there.
    That second part needs the over-draw detector fed on the frames without a
    pose too. Otherwise it is still on when the next pose arrives, and the
    fresh arrow falls off on that pose's very first frame."""
    on_string = main.ArrowOnString()
    _feed(on_string, 3.5, 3.5, None, None)

    assert on_string.release(_pose(3.5)) is False
    assert on_string.loaded is True
    assert _feed(on_string, 2.0) == [False]
    assert on_string.release(_pose(2.0)) is True


def test_a_one_frame_dip_does_not_renock_mid_debounce():
    """The race the re-nock guard exists for. Re-nocking on a single frame
    under `MIN_DRAW`, while the over-draw detector was still on, meant no fresh
    edge ever came. The new arrow was then loosed at full power from a bow
    drawn past its end."""
    on_string = main.ArrowOnString()

    fell = _feed(on_string, 2.0, 3.5, 3.5, 0.8, 3.5, 3.5, 3.5, 3.5)

    assert fell.count(True) == 1
    assert on_string.loaded is False
    assert on_string.release(_pose(3.5)) is False


def test_a_draw_too_short_to_count_is_not_loosed():
    """Judged here, once, of the camera pose. If each bow judged `MIN_DRAW` on
    its own rounded pose, a draw right at the minimum would shoot on one
    surface only."""
    on_string = main.ArrowOnString()

    assert on_string.release(_pose(main.MIN_DRAW * 0.9)) is False
    assert on_string.loaded is True
    assert on_string.release(_pose(main.MIN_DRAW)) is True


def test_easing_below_min_draw_renocks():
    on_string = main.ArrowOnString()
    _feed(on_string, 3.5, 3.5)
    assert on_string.loaded is False

    _feed(on_string, 0.8, 0.8)
    assert on_string.loaded is True

    assert _feed(on_string, 3.5, 3.5) == [False, True]


def test_easing_off_only_part_way_does_not_renock():
    on_string = main.ArrowOnString()

    _feed(on_string, 3.5, 3.5, 2.0, 2.0, 2.0)

    assert on_string.loaded is False


def test_metrics_readout_shows_power_and_the_fall_limit():
    import sys
    sys.path.insert(0, "tests")
    from test_gestures import _hand

    nocked = ((230, 200), (100, 200), 50.0)  # exactly a full draw
    text = "".join(
        run
        for line in main.metrics_readout([("hand", _hand())], nocked)
        for run, _ in line
    )

    assert "power 1.00" in text
    assert f"falls past {main.ARROW_LENGTH}" in text
