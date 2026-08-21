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


@patch("hand_tracker.HandLandmarker")
@patch("hand_tracker.mp")
@patch("hand_tracker.cv2")
def test_find_hands_returns_frame_and_landmarks_and_draws_when_requested(
    mock_cv2, mock_mp, mock_landmarker_cls
):
    mock_landmarker = MagicMock()
    mock_landmarker_cls.create_from_options.return_value = mock_landmarker
    fake_hand = _fake_hand()
    mock_landmarker.detect.return_value = SimpleNamespace(hand_landmarks=[fake_hand])
    mock_cv2.cvtColor.return_value = "rgb_frame"
    mock_mp.Image.return_value = "mp_image"

    tracker = HandTracker()
    frame = MagicMock()
    frame.shape = (480, 640, 3)

    result_frame, result_landmarks = tracker.find_hands(frame, draw=True)

    assert result_frame is frame
    assert result_landmarks == [fake_hand]
    mock_landmarker.detect.assert_called_once_with("mp_image")
    assert mock_cv2.line.called
    assert mock_cv2.circle.called


@patch("hand_tracker.HandLandmarker")
@patch("hand_tracker.mp")
@patch("hand_tracker.cv2")
def test_find_hands_does_not_draw_when_draw_is_false(mock_cv2, mock_mp, mock_landmarker_cls):
    mock_landmarker = MagicMock()
    mock_landmarker_cls.create_from_options.return_value = mock_landmarker
    mock_landmarker.detect.return_value = SimpleNamespace(hand_landmarks=[_fake_hand()])
    mock_cv2.cvtColor.return_value = "rgb_frame"
    mock_mp.Image.return_value = "mp_image"

    tracker = HandTracker()
    frame = MagicMock()
    frame.shape = (480, 640, 3)

    tracker.find_hands(frame, draw=False)

    mock_cv2.line.assert_not_called()
    mock_cv2.circle.assert_not_called()
