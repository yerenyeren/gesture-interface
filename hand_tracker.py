import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

DEFAULT_MODEL_PATH = "assets/hand_landmarker.task"


class HandTracker:
    def __init__(
        self,
        max_hands=1,
        detection_confidence=0.7,
        tracking_confidence=0.7,
        model_path=DEFAULT_MODEL_PATH,
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def _next_timestamp_ms(self):
        """A strictly increasing millisecond clock for VIDEO mode.

        `detect_for_video` raises on a timestamp that does not advance, and the
        clock is coarse enough to hand back the same millisecond twice in a row,
        so step past the previous value rather than trusting it.
        """
        self._last_timestamp_ms = max(
            int(time.monotonic() * 1000), self._last_timestamp_ms + 1
        )
        return self._last_timestamp_ms

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._next_timestamp_ms())

        if draw and result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                self._draw_landmarks(frame, hand_landmarks)

        return frame, result.hand_landmarks

    @staticmethod
    def _draw_landmarks(frame, hand_landmarks):
        height, width = frame.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

    @staticmethod
    def landmark_positions(hand_landmarks, frame_width, frame_height):
        return [
            (int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in hand_landmarks
        ]
