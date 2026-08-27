import math

from gestures import (
    gesture_metrics,
    is_fist,
    is_middle_pinch,
    is_ok_sign,
    is_open_palm,
    is_pinch,
    is_two_fingers_up,
    hand_scale,
    palm_center,
    pinch_point,
    pinch_threshold,
    FINGERS,
    FINGER_CURLED_RATIO,
    FINGER_EXTENDED_RATIO,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PALM_LANDMARKS,
    PINCH_ENTER_RATIO,
    PINCH_RELEASE_RATIO,
    pinched_finger,
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

# The pinch threshold is a fraction of the hand scale, so a hand under test has
# to have one. 80px was picked because 0.55 x 80 lands on a whole 44px, which
# keeps the boundary tests below exact rather than fighting float rounding.
HAND_SCALE_PX = 80
PINCH_ENTER_PX = PINCH_ENTER_RATIO * HAND_SCALE_PX
PINCH_RELEASE_PX = PINCH_RELEASE_RATIO * HAND_SCALE_PX

# Somewhere a thumb can sit that starts no pinch but does not end one either.
IN_THE_DEAD_BAND = (PINCH_ENTER_PX + PINCH_RELEASE_PX) / 2

# Far enough that a hand built without a middle finger position cannot have one
# accidentally win the pinch.
PARKED = (900, 900)

# Tip-to-PIP distance ratios comfortably either side of the two thresholds.
EXTENDED = FINGER_EXTENDED_RATIO + 0.25
CURLED = FINGER_CURLED_RATIO - 0.3


def _landmarks(thumb, index, scale=HAND_SCALE_PX, middle=PARKED):
    """A minimal hand: a thumb, two fingertips, and a wrist-to-knuckle scale.

    The scale has to be here because the pinch threshold is measured against
    `hand_scale`, not against a fixed pixel count.

    The middle tip has to be here because only the fingertip *nearest* the
    thumb pinches. Left at the (0, 0) every unset landmark starts as, it would
    sit right on top of a thumb placed at the origin and win every pinch in the
    file. It is parked far off to the side instead, the way `_hand` parks the
    thumb, and moved in deliberately by the tests that care.
    """
    landmarks = [(0, 0)] * 21
    landmarks[WRIST] = (0, 0)
    landmarks[MIDDLE_MCP] = (0, -scale)
    landmarks[THUMB_TIP] = thumb
    landmarks[INDEX_TIP] = index
    landmarks[MIDDLE_TIP] = middle
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
    assert is_pinch(_landmarks((0, 0), (PINCH_ENTER_PX - 1, 0))) is True


def test_distance_at_threshold_is_not_pinch():
    assert is_pinch(_landmarks((0, 0), (PINCH_ENTER_PX, 0))) is False


def test_distance_just_over_threshold_is_not_pinch():
    assert is_pinch(_landmarks((0, 0), (PINCH_ENTER_PX + 1, 0))) is False


def test_pinch_threshold_tracks_the_hand_scale():
    assert pinch_threshold(_landmarks((0, 0), (0, 0), scale=80)) == 44.0
    assert pinch_threshold(_landmarks((0, 0), (0, 0), scale=40)) == 22.0


def test_the_release_threshold_is_the_looser_of_the_two():
    landmarks = _landmarks((0, 0), (0, 0), scale=80)
    assert pinch_threshold(landmarks, held=True) == 60.0
    assert pinch_threshold(landmarks, held=True) > pinch_threshold(landmarks)


def test_a_middle_pinch_in_flight_still_reads_as_an_index_pinch():
    """Pinned as a known defect, not as desired behaviour.

    A thumb reaching for the middle finger drags the index tip along behind it,
    and the index starts nearer and travels less, so for most of the approach
    it is genuinely the closer of the two and wins the pinch outright. This is
    what fires a left click on the way into a right one. Winner-take-all does
    not help — the index wins fairly — and neither does a tighter threshold,
    which only shortens the window. The fix has to be temporal.
    """
    in_flight = _landmarks((0, 0), (19, 0), middle=(21, 0))

    assert is_pinch(in_flight) is True
    assert is_middle_pinch(in_flight) is False


def test_the_nearer_fingertip_owns_the_pinch():
    """Both tips inside the threshold at once. The further one is not pinching
    at all — not merely outranked — so the two predicates cannot both fire."""
    landmarks = _landmarks((0, 0), (20, 0), middle=(8, 0))
    assert 20 < PINCH_ENTER_PX and 8 < PINCH_ENTER_PX

    assert pinched_finger(landmarks) == MIDDLE_TIP
    assert is_middle_pinch(landmarks) is True
    assert is_pinch(landmarks) is False


def test_a_held_pinch_survives_drifting_past_the_enter_threshold():
    drifted = _landmarks((0, 0), (IN_THE_DEAD_BAND, 0))

    assert is_pinch(drifted, held=True) is True


def test_a_pinch_cannot_start_inside_the_release_band():
    """The same hand at the same distance as the test above — only whether a
    pinch was already running separates them. Without that gap a thumb resting
    on the boundary chatters the click on and off frame by frame."""
    drifted = _landmarks((0, 0), (IN_THE_DEAD_BAND, 0))

    assert is_pinch(drifted, held=False) is False


def test_only_the_held_finger_gets_the_looser_threshold():
    """Holding a left click must not make it easier to start a right one."""
    drifted = _landmarks((0, 0), (PARKED[0], 0), middle=(IN_THE_DEAD_BAND, 0))

    assert is_middle_pinch(drifted, held=True) is True
    assert is_middle_pinch(drifted, held=False) is False


def test_the_same_pose_pinches_at_any_camera_distance():
    """The whole point of a ratio: a pinch must not depend on how close the
    hand is to the camera. The same gesture is built at two scales, with every
    distance scaled alike, and has to read the same both times."""
    for scale, gap in ((80, 20), (40, 10)):
        assert is_pinch(_landmarks((0, 0), (gap, 0), scale=scale)) is True
    for scale, gap in ((80, 60), (40, 30)):
        assert is_pinch(_landmarks((0, 0), (gap, 0), scale=scale)) is False


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


def test_a_pinching_hand_is_never_scrolling():
    """The overlap that was eating half the clicks.

    A click pinch barely bends the index — on a real hand it reads right at
    FINGER_EXTENDED_RATIO — so with the middle up and the last two down it
    otherwise satisfies the scroll pose exactly. The loop resolves scroll
    first and feeds the click detectors False, so the click was surviving or
    dying on which side of 1.15 the index happened to land that frame.
    """
    for ratio in (FINGER_EXTENDED_RATIO - 0.01, FINGER_EXTENDED_RATIO + 0.01, 1.30):
        pinching = _pinched(_hand(index=ratio, ring=CURLED, pinky=CURLED))

        assert is_pinch(pinching) is True
        assert is_two_fingers_up(pinching) is False


def test_scrolling_survives_with_the_thumb_clear():
    """The other half of the guard: keeping the thumb out of it must not cost
    the scroll pose itself."""
    scrolling = _hand(ring=CURLED, pinky=CURLED)

    assert is_pinch(scrolling) is False
    assert is_two_fingers_up(scrolling) is True


def test_a_held_pinch_cannot_leak_into_the_scroll_pose():
    """The veto is measured against the release threshold, not the entry one,
    so a pinch drifting inside the hysteresis band is still a pinch as far as
    every other gesture is concerned — it does not briefly become a scroll on
    its way back out."""
    drifting = _hand(index=FINGER_EXTENDED_RATIO + 0.01, ring=CURLED, pinky=CURLED)
    # Sized off this hand, whose scale is not the one _landmarks uses.
    band = (pinch_threshold(drifting) + pinch_threshold(drifting, held=True)) / 2
    index_tip = drifting[INDEX_TIP]
    drifting[THUMB_TIP] = (index_tip[0], index_tip[1] + band)

    assert pinched_finger(drifting, held=INDEX_TIP) == INDEX_TIP

    assert is_pinch(drifting, held=False) is False
    assert is_pinch(drifting, held=True) is True
    assert is_two_fingers_up(drifting) is False


def test_pinch_with_three_fingers_extended_is_an_ok_sign():
    assert is_ok_sign(_pinched(_hand(index=CURLED))) is True


def test_pinch_with_curled_fingers_is_a_plain_click_pinch():
    landmarks = _pinched(_hand(CURLED, CURLED, CURLED, CURLED))

    assert is_pinch(landmarks) is True
    assert is_ok_sign(landmarks) is False


def test_ok_sign_requires_a_pinch():
    assert is_ok_sign(_hand(index=CURLED)) is False


def test_the_ok_sign_pinches_at_the_looser_threshold():
    """The archery draw is held for the length of a draw, and what separates it
    from a click pinch is its three extended fingers, not how close the thumb
    is. Judging it by the tight entry threshold would only make the bow hard to
    start and prone to dropping mid-draw, so it gets the release one."""
    ok_sign = _hand(index=CURLED)
    index_tip = ok_sign[INDEX_TIP]
    gap = pinch_threshold(ok_sign) + 1
    assert gap < pinch_threshold(ok_sign, held=True)

    # Away from the middle finger, so the index stays the nearest tip.
    ok_sign[THUMB_TIP] = (index_tip[0] - gap, index_tip[1])

    assert is_pinch(ok_sign) is False
    assert is_ok_sign(ok_sign) is True


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


def test_gesture_metrics_agree_with_the_predicates_they_explain():
    """The readout has to be gating on exactly what the predicates gate on. A
    parallel calculation that drifted would make the tuning HUD actively
    misleading, which is worse than having none."""
    for landmarks in (
        _hand(),
        _hand(CURLED, CURLED, CURLED, CURLED),
        _hand(ring=CURLED, pinky=CURLED),
        _pinched(_hand(index=CURLED)),
    ):
        metrics = gesture_metrics(landmarks)

        extended = [r > FINGER_EXTENDED_RATIO for r in metrics["ratios"]]
        curled = [r < FINGER_CURLED_RATIO for r in metrics["ratios"]]
        assert all(extended) is is_open_palm(landmarks)
        assert all(curled) is is_fist(landmarks)

        assert (metrics["pinched"] == INDEX_TIP) is is_pinch(landmarks)
        assert (metrics["pinched"] == MIDDLE_TIP) is is_middle_pinch(landmarks)


def test_gesture_metrics_reports_the_dead_band_between_the_thresholds():
    """A finger sitting between the two ratios is neither extended nor curled,
    so every gesture using it silently fails. That state is invisible in the
    booleans and is the reason the metrics exist."""
    midway = (FINGER_EXTENDED_RATIO + FINGER_CURLED_RATIO) / 2
    landmarks = _hand(CURLED, CURLED, CURLED, midway)

    pinky_ratio = gesture_metrics(landmarks)["ratios"][3]
    assert FINGER_CURLED_RATIO < pinky_ratio < FINGER_EXTENDED_RATIO
    assert is_fist(landmarks) is False
    assert is_open_palm(landmarks) is False


def test_gesture_metrics_names_the_finger_that_won_the_pinch():
    """Both distances are still measured independently, but the verdict is not:
    only the nearer tip pinches, so the readout has to say which one did rather
    than leave it to be inferred from two numbers."""
    landmarks = _hand()
    landmarks[THUMB_TIP] = landmarks[MIDDLE_TIP]
    metrics = gesture_metrics(landmarks)

    assert metrics["middle_pinch_px"] == 0.0
    assert metrics["index_pinch_px"] > metrics["pinch_enter_px"]
    assert metrics["pinched"] == MIDDLE_TIP
    assert metrics["scale_px"] == 60.0
