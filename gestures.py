import math

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

PINCH_THRESHOLD_PX = 40
PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# (tip, pip) for the four non-thumb fingers, in finger order.
FINGERS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)

# A finger reads as extended when its tip sits meaningfully further from the
# wrist than its middle joint does, and as curled when it sits closer. These are
# ratios rather than pixel counts so they hold up as the hand rotates or moves
# away from the camera. The gap between the two leaves a half-curled finger in
# neither state instead of flickering between them frame to frame.
FINGER_EXTENDED_RATIO = 1.15
FINGER_CURLED_RATIO = 1.0


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _finger_ratio(landmarks, tip, pip):
    wrist = landmarks[WRIST]
    pip_distance = _dist(wrist, landmarks[pip])
    if pip_distance == 0:
        return 0.0
    return _dist(wrist, landmarks[tip]) / pip_distance


def _finger_extended(landmarks, tip, pip):
    return _finger_ratio(landmarks, tip, pip) > FINGER_EXTENDED_RATIO


def _finger_curled(landmarks, tip, pip):
    return _finger_ratio(landmarks, tip, pip) < FINGER_CURLED_RATIO


def is_pinch(landmarks):
    return _dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) < PINCH_THRESHOLD_PX


def is_middle_pinch(landmarks):
    """Thumb against the middle finger — the right-click counterpart to is_pinch."""
    return _dist(landmarks[THUMB_TIP], landmarks[MIDDLE_TIP]) < PINCH_THRESHOLD_PX


def is_open_palm(landmarks):
    return all(_finger_extended(landmarks, tip, pip) for tip, pip in FINGERS)


def is_fist(landmarks):
    return all(_finger_curled(landmarks, tip, pip) for tip, pip in FINGERS)


def is_two_fingers_up(landmarks):
    """Index and middle up, ring and pinky down — the scroll pose."""
    return (
        _finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
        and _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and _finger_curled(landmarks, RING_TIP, RING_PIP)
        and _finger_curled(landmarks, PINKY_TIP, PINKY_PIP)
    )


def is_ok_sign(landmarks):
    """The Mongolian thumb draw, which reads as an OK sign to a camera.

    The thumb hooks the string and the index folds over the thumbnail to lock
    it, leaving the other three fingers extended. Those three fingers are the
    only thing separating this from an ordinary click pinch, where they curl.
    """
    if not is_pinch(landmarks):
        return False
    return all(
        _finger_extended(landmarks, tip, pip)
        for tip, pip in FINGERS[1:]
    )


def palm_center(landmarks):
    xs = [landmarks[i][0] for i in PALM_LANDMARKS]
    ys = [landmarks[i][1] for i in PALM_LANDMARKS]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def pinch_point(landmarks):
    """Midpoint of the thumb/index pinch — where the bowstring is nocked."""
    (x1, y1), (x2, y2) = landmarks[THUMB_TIP], landmarks[INDEX_TIP]
    return (x1 + x2) // 2, (y1 + y2) // 2


def hand_scale(landmarks):
    """A reference length for this hand, in pixels.

    Lets sizes and distances be expressed per-hand instead of in absolute
    pixels, so they track how close the hand is to the camera.
    """
    return _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
