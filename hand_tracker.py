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

# MediaPipe's landmark topology is fixed: four joints per finger, in order out
# from the palm, after the wrist at 0.
FINGER_CHAINS = {
    "T": (1, 2, 3, 4),
    "I": (5, 6, 7, 8),
    "M": (9, 10, 11, 12),
    "R": (13, 14, 15, 16),
    "P": (17, 18, 19, 20),
}

# A colour per finger, and the tip carries its initial. The pair that matters
# is index and middle: those are the two chains MediaPipe transposes when the
# thumb closes on the middle finger, and they are also the two that decide left
# click against right. Cyan and magenta are the furthest apart of the five so
# that swap is unmistakable rather than something to squint at.
FINGER_COLORS = {
    "T": (0, 165, 255, 255),
    "I": (255, 255, 0, 255),
    "M": (255, 0, 255, 255),
    "R": (0, 255, 255, 255),
    "P": (0, 255, 0, 255),
}

# The wrist and the knuckle span across the palm belong to no single finger.
PALM_COLOR = (170, 170, 170, 255)

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FINGER_OF = {point: name for name, chain in FINGER_CHAINS.items() for point in chain}


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
            height, width = frame.shape[:2]
            for hand_landmarks in result.hand_landmarks:
                self.draw_landmarks(
                    frame, self.landmark_positions(hand_landmarks, width, height)
                )

        return frame, result.hand_landmarks

    @staticmethod
    def draw_landmarks(frame, points):
        """Draw one hand's skeleton from pixel coordinates.

        Takes the pixel list `landmark_positions` produces rather than raw
        normalized landmarks, so the skeleton is drawn from exactly the
        coordinates the gestures are computed from. Scaling here independently —
        off `frame.shape` — would disagree with the caller whenever the driver
        hands back a frame size different from the one it reports.

        Public because the caller decides *when* to draw: the mode is not known
        until after detection, and the bow pose wants a clean frame.

        Every finger gets its own colour and its tip carries its initial,
        because the interesting failure here is not where the landmarks are but
        *which finger they claim to be*: reaching the thumb to the middle
        finger makes MediaPipe hand back the index chain and the middle chain
        transposed. Drawn in one colour, as this was, a transposed skeleton
        looks exactly like a correct one — so the bug that decides left click
        against right was invisible in the very view meant to expose it.
        """
        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            finger = _FINGER_OF.get(connection.start)
            # A connection spanning two groups — or leaving the wrist — is
            # palm, not finger.
            if finger is not None and finger == _FINGER_OF.get(connection.end):
                color = FINGER_COLORS[finger]
            else:
                color = PALM_COLOR
            cv2.line(frame, points[connection.start], points[connection.end], color, 2)

        for index, point in enumerate(points):
            color = FINGER_COLORS.get(_FINGER_OF.get(index), PALM_COLOR)
            cv2.circle(frame, point, 4, color, -1)

        for name, chain in FINGER_CHAINS.items():
            tip = points[chain[-1]]
            HandTracker._draw_label(frame, name, (tip[0] + 7, tip[1] - 7),
                                    FINGER_COLORS[name])

    @staticmethod
    def _draw_label(frame, text, position, color):
        """Outlined so a letter stays readable over skin, sleeve or wall."""
        for shade, thickness in (((0, 0, 0, 255), 3), (color, 1)):
            cv2.putText(
                frame, text, position, LABEL_FONT, 0.5, shade, thickness, cv2.LINE_AA,
            )

    @staticmethod
    def landmark_positions(hand_landmarks, frame_width, frame_height):
        return [
            (int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in hand_landmarks
        ]
