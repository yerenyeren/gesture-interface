from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hand_tracker import FINGER_CHAINS, FINGER_COLORS, PALM_COLOR, HandTracker


def test_landmark_positions_converts_normalized_to_pixel_coords():
    hand_landmarks = [
        SimpleNamespace(x=0.0, y=0.0),
        SimpleNamespace(x=0.5, y=0.5),
        SimpleNamespace(x=1.0, y=1.0),
    ]

    positions = HandTracker.landmark_positions(hand_landmarks, frame_width=640, frame_height=480)

    assert positions == [(0, 0), (320, 240), (640, 480)]


def _fake_hand():
    return [SimpleNamespace(x=0.1 * i, y=0.1 * i) for i in range(21)]


def _tracker_with(mock_landmarker_cls, mock_cv2, mock_mp, hands=None):
    mock_landmarker = MagicMock()
    mock_landmarker_cls.create_from_options.return_value = mock_landmarker
    mock_landmarker.detect_for_video.return_value = SimpleNamespace(
        hand_landmarks=hands if hands is not None else [_fake_hand()]
    )
    mock_cv2.cvtColor.return_value = "rgb_frame"
    mock_mp.Image.return_value = "mp_image"
    return HandTracker(), mock_landmarker


def _frame():
    frame = MagicMock()
    frame.shape = (480, 640, 3)
    return frame


@patch("hand_tracker.HandLandmarker")
@patch("hand_tracker.mp")
@patch("hand_tracker.cv2")
def test_find_hands_returns_frame_and_landmarks_and_draws_when_requested(
    mock_cv2, mock_mp, mock_landmarker_cls
):
    fake_hand = _fake_hand()
    tracker, mock_landmarker = _tracker_with(
        mock_landmarker_cls, mock_cv2, mock_mp, hands=[fake_hand]
    )
    frame = _frame()

    result_frame, result_landmarks = tracker.find_hands(frame, draw=True)

    assert result_frame is frame
    assert result_landmarks == [fake_hand]
    image, timestamp = mock_landmarker.detect_for_video.call_args.args
    assert image == "mp_image"
    assert isinstance(timestamp, int)
    assert mock_cv2.line.called
    assert mock_cv2.circle.called


@patch("hand_tracker.HandLandmarker")
@patch("hand_tracker.mp")
@patch("hand_tracker.cv2")
def test_find_hands_does_not_draw_when_draw_is_false(mock_cv2, mock_mp, mock_landmarker_cls):
    tracker, _ = _tracker_with(mock_landmarker_cls, mock_cv2, mock_mp)

    tracker.find_hands(_frame(), draw=False)

    mock_cv2.line.assert_not_called()
    mock_cv2.circle.assert_not_called()


@patch("hand_tracker.HandLandmarker")
@patch("hand_tracker.mp")
@patch("hand_tracker.cv2")
def test_video_timestamps_strictly_increase(mock_cv2, mock_mp, mock_landmarker_cls):
    tracker, mock_landmarker = _tracker_with(mock_landmarker_cls, mock_cv2, mock_mp)

    for _ in range(5):
        tracker.find_hands(_frame(), draw=False)

    timestamps = [call.args[1] for call in mock_landmarker.detect_for_video.call_args_list]
    assert all(later > earlier for earlier, later in zip(timestamps, timestamps[1:]))


@patch("hand_tracker.HandLandmarker")
@patch("hand_tracker.mp")
@patch("hand_tracker.cv2")
def test_video_timestamps_advance_even_on_a_frozen_clock(
    mock_cv2, mock_mp, mock_landmarker_cls
):
    # detect_for_video raises on a timestamp that does not advance, and a loop
    # faster than the clock's resolution reads the same millisecond twice.
    tracker, mock_landmarker = _tracker_with(mock_landmarker_cls, mock_cv2, mock_mp)

    with patch("hand_tracker.time.monotonic", return_value=12.345):
        for _ in range(4):
            tracker.find_hands(_frame(), draw=False)

    timestamps = [call.args[1] for call in mock_landmarker.detect_for_video.call_args_list]
    assert timestamps == [12345, 12346, 12347, 12348]


def test_draw_landmarks_draws_from_pixel_coordinates():
    """The skeleton is drawn from the same pixel list the gestures are computed
    from. Rescaling here off frame.shape instead would disagree with the caller
    whenever the driver hands back a size different from the one it reports."""
    frame = _frame()
    points = [(10 * i, 20 * i) for i in range(21)]

    with patch("hand_tracker.cv2") as mock_cv2:
        HandTracker.draw_landmarks(frame, points)

    assert mock_cv2.line.called
    assert mock_cv2.circle.call_count == 21
    # Every circle sits on a point it was handed, untouched.
    # Passed straight through: had draw_landmarks rescaled by frame.shape, these
    # would not come back as the coordinates that went in.
    drawn = [call.args[1] for call in mock_cv2.circle.call_args_list]
    assert drawn == points


def _indexed_points():
    """Each landmark parked at its own index, so a drawn line can be traced
    back to the two landmarks it connects."""
    return [(i, 0) for i in range(21)]


def _drawn_lines(points):
    with patch("hand_tracker.cv2") as mock_cv2:
        HandTracker.draw_landmarks(_frame(), points)
    return {
        frozenset((call.args[1][0], call.args[2][0])): call.args[3]
        for call in mock_cv2.line.call_args_list
    }


def _chain_colors(lines, name):
    chain = FINGER_CHAINS[name]
    return {lines[frozenset(pair)] for pair in zip(chain, chain[1:])}


def test_each_finger_chain_is_drawn_in_its_own_colour():
    lines = _drawn_lines(_indexed_points())

    for name in FINGER_CHAINS:
        assert _chain_colors(lines, name) == {FINGER_COLORS[name]}


def test_the_index_and_middle_chains_never_share_a_colour():
    """The reason this drawing exists at all.

    MediaPipe hands back the index and middle chains transposed when the thumb
    closes on the middle finger, and which of those two the thumb is nearest is
    what decides left click against right. Drawn in one colour the swap was
    invisible, so the skeleton hid the one thing worth looking at.
    """
    lines = _drawn_lines(_indexed_points())

    assert _chain_colors(lines, "I").isdisjoint(_chain_colors(lines, "M"))


def test_connections_spanning_the_palm_belong_to_no_finger():
    lines = _drawn_lines(_indexed_points())

    # Knuckle to knuckle across the palm, and the wrist's own two spans. The
    # palm hangs off landmark 1 rather than the wrist, so (0, 5) is not a
    # connection MediaPipe draws at all.
    for span in ((5, 9), (9, 13), (1, 5), (0, 1), (0, 17)):
        assert lines[frozenset(span)] == PALM_COLOR


def test_every_fingertip_is_labelled_with_its_initial():
    with patch("hand_tracker.cv2") as mock_cv2:
        HandTracker.draw_landmarks(_frame(), _indexed_points())

    assert {call.args[1] for call in mock_cv2.putText.call_args_list} == set(FINGER_CHAINS)


def test_a_label_sits_beside_its_own_fingertip():
    """Beside, not on: a letter centred on the tip would hide the landmark it
    is naming."""
    points = _indexed_points()
    with patch("hand_tracker.cv2") as mock_cv2:
        HandTracker.draw_landmarks(_frame(), points)

    for call in mock_cv2.putText.call_args_list:
        tip = points[FINGER_CHAINS[call.args[1]][-1]]
        offset = call.args[2]
        assert offset != tip
        assert abs(offset[0] - tip[0]) <= 10 and abs(offset[1] - tip[1]) <= 10
