import math

from gestures import (
    is_fist,
    is_middle_pinch,
    is_ok_sign,
    is_open_palm,
    is_pinch,
    is_two_fingers_up,
    hand_scale,
    palm_center,
    pinch_point,
    FINGERS,
    FINGER_CURLED_RATIO,
    FINGER_EXTENDED_RATIO,
    MIDDLE_MCP,
    PALM_LANDMARKS,
    PINCH_THRESHOLD_PX,
    THUMB_TIP,
    INDEX_TIP,
    WRIST,
)

WRIST_POINT = (100, 300)
PIP_OFFSET = 120
# Fingers fan out from the wrist rather than stacking on one line, so that two
# extended fingertips sit further apart than PINCH_THRESHOLD_PX and the thumb
# can be placed on one without touching its neighbour.
FINGER_ANGLES = (-0.525, -0.175, 0.175, 0.525)

# Tip-to-PIP distance ratios comfortably either side of the two thresholds.
EXTENDED = FINGER_EXTENDED_RATIO + 0.25
CURLED = FINGER_CURLED_RATIO - 0.3


def _landmarks(thumb, index):
    landmarks = [(0, 0)] * 9
    landmarks[THUMB_TIP] = thumb
    landmarks[INDEX_TIP] = index
    return landmarks


def _hand(index=EXTENDED, middle=EXTENDED, ring=EXTENDED, pinky=EXTENDED):
    """A hand pointing straight up from the wrist, one ratio per finger.

    Each argument is that finger tip's distance from the wrist as a multiple of
    its PIP joint's distance — which is exactly what the extended/curled
    thresholds compare. The thumb starts far off to the side so the hand does
    not accidentally read as pinching.
    """
    landmarks = [(0, 0)] * 21
    landmarks[WRIST] = WRIST_POINT
    landmarks[MIDDLE_MCP] = (WRIST_POINT[0], WRIST_POINT[1] - 60)
    landmarks[THUMB_TIP] = (WRIST_POINT[0] + 300, WRIST_POINT[1])

    ratios = (index, middle, ring, pinky)
    for ratio, angle, (tip, pip) in zip(ratios, FINGER_ANGLES, FINGERS):
        # Both joints lie on one ray out of the wrist, so the tip/PIP distance
        # ratio is exactly `ratio` whatever direction the finger points.
        direction = (math.sin(angle), -math.cos(angle))
        landmarks[pip] = _along(direction, PIP_OFFSET)
        landmarks[tip] = _along(direction, PIP_OFFSET * ratio)
    return landmarks


def _along(direction, distance):
    return (
        round(WRIST_POINT[0] + direction[0] * distance),
        round(WRIST_POINT[1] + direction[1] * distance),
    )


def _pinched(landmarks):
    """Move the thumb tip onto the index tip so the hand reads as pinching."""
    landmarks = list(landmarks)
    landmarks[THUMB_TIP] = landmarks[INDEX_TIP]
    return landmarks


def test_identical_points_is_pinch():
    assert is_pinch(_landmarks((100, 100), (100, 100))) is True


def test_distance_just_under_threshold_is_pinch():
    assert is_pinch(_landmarks((0, 0), (PINCH_THRESHOLD_PX - 1, 0))) is True


def test_distance_at_threshold_is_not_pinch():
    assert is_pinch(_landmarks((0, 0), (PINCH_THRESHOLD_PX, 0))) is False


def test_distance_just_over_threshold_is_not_pinch():
    assert is_pinch(_landmarks((0, 0), (PINCH_THRESHOLD_PX + 1, 0))) is False


def test_negative_coordinates_use_absolute_distance():
    assert is_pinch(_landmarks((-20, -20), (20, 20))) is False
    assert is_pinch(_landmarks((-5, -5), (5, 5))) is True


def test_palm_center_averages_palm_landmarks():
    landmarks = [(0, 0)] * 18
    points = [(0, 0), (10, 20), (20, 40), (30, 60), (40, 80)]
    for index, point in zip(PALM_LANDMARKS, points):
        landmarks[index] = point

    assert palm_center(landmarks) == (20, 40)


def test_middle_pinch_measures_thumb_against_the_middle_finger():
    landmarks = _hand()
    assert is_middle_pinch(landmarks) is False

    middle_tip = FINGERS[1][0]
    landmarks[THUMB_TIP] = landmarks[middle_tip]
    assert is_middle_pinch(landmarks) is True
    # The index finger is untouched, so this must not read as a left-click pinch.
    assert is_pinch(landmarks) is False


def test_all_fingers_extended_is_an_open_palm():
    assert is_open_palm(_hand()) is True


def test_one_curled_finger_is_not_an_open_palm():
    assert is_open_palm(_hand(pinky=CURLED)) is False


def test_all_fingers_curled_is_a_fist():
    assert is_fist(_hand(EXTENDED, CURLED, CURLED, CURLED)) is False
    assert is_fist(_hand(CURLED, CURLED, CURLED, CURLED)) is True


def test_fist_and_open_palm_never_agree():
    fist = _hand(CURLED, CURLED, CURLED, CURLED)
    palm = _hand()

    assert (is_fist(fist), is_open_palm(fist)) == (True, False)
    assert (is_fist(palm), is_open_palm(palm)) == (False, True)


def test_index_and_middle_up_is_two_fingers_up():
    assert is_two_fingers_up(_hand(ring=CURLED, pinky=CURLED)) is True


def test_two_fingers_up_requires_the_last_two_fingers_down():
    assert is_two_fingers_up(_hand()) is False
    assert is_two_fingers_up(_hand(ring=CURLED)) is False


def test_pinch_with_three_fingers_extended_is_an_ok_sign():
    assert is_ok_sign(_pinched(_hand(index=CURLED))) is True


def test_pinch_with_curled_fingers_is_a_plain_click_pinch():
    landmarks = _pinched(_hand(CURLED, CURLED, CURLED, CURLED))

    assert is_pinch(landmarks) is True
    assert is_ok_sign(landmarks) is False


def test_ok_sign_requires_a_pinch():
    assert is_ok_sign(_hand(index=CURLED)) is False


def test_ok_sign_and_two_fingers_up_never_agree():
    ok_sign = _pinched(_hand(index=CURLED))
    scrolling = _hand(ring=CURLED, pinky=CURLED)

    assert (is_ok_sign(ok_sign), is_two_fingers_up(ok_sign)) == (True, False)
    assert (is_ok_sign(scrolling), is_two_fingers_up(scrolling)) == (False, True)


def test_pinch_point_is_the_midpoint_of_thumb_and_index():
    assert pinch_point(_landmarks((100, 200), (140, 260))) == (120, 230)


def test_hand_scale_measures_wrist_to_middle_knuckle():
    landmarks = [(0, 0)] * 21
    landmarks[WRIST] = (100, 100)
    landmarks[MIDDLE_MCP] = (130, 140)

    assert hand_scale(landmarks) == 50.0
