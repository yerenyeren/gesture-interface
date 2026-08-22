from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hand_tracker import HandTracker


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
