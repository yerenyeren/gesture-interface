import time

import cv2

from animations import HorseBow
from gesture_state import EdgeDetector
from gestures import (
    hand_scale,
    is_fist,
    is_middle_pinch,
    is_ok_sign,
    is_open_palm,
    is_pinch,
    is_two_fingers_up,
    palm_center,
    pinch_point,
)
from hand_tracker import HandTracker
from mouse_control import MouseController

SCROLL_DEADZONE_PX = 12
SCROLL_GAIN = 0.4
HUD_COLOR = (240, 240, 240)

WINDOW_NAME = "Gesture Interface"

# Requested capture format. The camera is free to ignore any of this, which is
# why the frame size is read back off the capture rather than assumed.
#
# Deliberately no MJPG fourcc here: measured on this machine, every mode the
# camera offers — MJPG, YUYV, 640x480 through 1280x720 — delivers the same
# 15 fps, and asking for MJPG made the driver hand back 848x480 instead of the
# 640x480 the default gives. Bigger frames for no extra frames is a bad trade.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30

# Auto-exposure lengthens the shutter in a dim room, which on some webcams costs
# frame rate. Measured here it does not: with the room lit, auto delivers the
# camera's full 15 fps and a well-exposed image, so this is left on auto. A
# too-dark frame costs far more in missed hand detections than any plausible
# frame-rate win. Kept as a knob only for the low-light case — if the HUD shows
# the camera starving, try a V4L2 exposure value, roughly 40 (dark, fast) to
# 250 (bright, slow).
CAPTURE_EXPOSURE = None

# V4L2 exposure mode values. These are device-level controls: they persist on
# /dev/video* after the process exits, so a manual value left behind by anything
# else — another app, a benchmark, a previous run of this one — is inherited as
# a near-black frame that detects no hands at all. Always asking for auto is
# what keeps that from being a mystery to debug.
EXPOSURE_MANUAL = 1
EXPOSURE_AUTO = 3


def open_camera(index=0):
    """Open the webcam and ask for a format that will not throttle the loop."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    # Keeps read() returning the newest frame instead of the oldest queued one.
    # Frequently ignored by the V4L2 backend, so it is a bonus, not a guarantee.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if CAPTURE_EXPOSURE is None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSURE_AUTO)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, EXPOSURE_MANUAL)
        cap.set(cv2.CAP_PROP_EXPOSURE, CAPTURE_EXPOSURE)
    return cap


class StageTimer:
    """Exponentially smoothed per-stage frame timings for the HUD.

    Raw per-frame numbers flicker too fast to read, and the whole point of
    putting them on screen is to be able to watch them while moving a hand
    around.
    """

    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self._stages = {}

    def record(self, stage, seconds):
        milliseconds = seconds * 1000
        previous = self._stages.get(stage)
        self._stages[stage] = (
            milliseconds
            if previous is None
            else previous * self.smoothing + milliseconds * (1 - self.smoothing)
        )

    def summary(self):
        total = sum(self._stages.values())
        fps = 1000 / total if total else 0.0
        stages = "  ".join(f"{name} {ms:.1f}" for name, ms in self._stages.items())
        return f"{fps:4.1f} fps   {stages}"


def find_archery_hands(hands):
    """Return (grip_hand, string_hand) for the steppe archery pose, or None.

    A closed fist grips the bow and the Mongolian thumb draw — which reads as an
    OK sign — holds the string. Requiring both hands at once is what keeps an
    ordinary one-handed click pinch from ever loosing an arrow.
    """
    if len(hands) < 2:
        return None

    first, second = hands[0], hands[1]
    if is_fist(first) and is_ok_sign(second):
        return first, second
    if is_fist(second) and is_ok_sign(first):
        return second, first
    return None


def control_hand(hands):
    """The hand that drives the cursor: the right-most one on screen.

    MediaPipe's ordering is not stable between frames, so picking by position
    stops the cursor hopping between hands when both are visible.
    """
    return max(hands, key=lambda landmarks: palm_center(landmarks)[0])


def draw_hud(frame, mode, stats=None):
    lines = [(mode, 0.8, 34, 5, 2)]
    if stats is not None:
        lines.append((stats, 0.5, 62, 3, 1))

    for text, scale, y, outline, weight in lines:
        for color, thickness in (((0, 0, 0), outline), (HUD_COLOR, weight)):
            cv2.putText(
                frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA,
            )


def main():
    cap = open_camera()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # WINDOW_GUI_NORMAL selects the plain highgui window. The default here is
    # WINDOW_GUI_EXPANDED, whose Qt toolbar and right-click menu run a nested
    # modal event loop that blocks waitKey — and the gesture cursor right-clicks
    # on this very window, so the app could freeze itself.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    tracker = HandTracker(max_hands=2)
    mouse = MouseController(frame_width, frame_height)
    bow = HorseBow()
    timer = StageTimer()

    # Actions fire on gesture edges, never on every frame the gesture is held.
    left_click = EdgeDetector()
    right_click = EdgeDetector()
    pause_toggle = EdgeDetector(min_frames=3)
    at_full_draw = EdgeDetector()

    paused = False
    show_stats = True
    last_scroll_y = None
    nocked = None  # (grip, nock, scale) from the most recent archery frame

    while True:
        started = time.perf_counter()
        success, frame = cap.read()
        if not success:
            break
        timer.record("cap", time.perf_counter() - started)

        detect_started = time.perf_counter()
        frame = cv2.flip(frame, 1)
        frame, raw_hands = tracker.find_hands(frame)
        hands = [
            tracker.landmark_positions(hand, frame_width, frame_height)
            for hand in raw_hands
        ]
        timer.record("det", time.perf_counter() - detect_started)

        draw_started = time.perf_counter()
        archery = find_archery_hands(hands)
        at_full_draw.update(archery is not None)

        # The pose has already broken by the frame the string hand opens, so the
        # shot is taken from the last state the bow was actually drawn in.
        if at_full_draw.fell and nocked is not None:
            bow.loose(*nocked)
            nocked = None

        if archery is not None:
            grip_hand, string_hand = archery
            nocked = (
                palm_center(grip_hand),
                pinch_point(string_hand),
                hand_scale(grip_hand),
            )
            bow.draw(frame, *nocked)
            mode = "DRAWING BOW"
            # The mouse deliberately sits idle: the string hand is pinching,
            # which would otherwise read as a click.
            for detector in (pause_toggle, left_click, right_click):
                detector.update(False)
            last_scroll_y = None
            mouse.reset()

        elif hands:
            landmarks = control_hand(hands)

            pause_toggle.update(is_open_palm(landmarks))
            if pause_toggle.rose:
                paused = not paused

            if paused:
                mode = "PAUSED"
                last_scroll_y = None
                left_click.update(False)
                right_click.update(False)
                mouse.reset()

            elif is_two_fingers_up(landmarks):
                mode = "SCROLL"
                palm_y = palm_center(landmarks)[1]
                if last_scroll_y is None:
                    last_scroll_y = palm_y
                else:
                    delta = last_scroll_y - palm_y
                    if abs(delta) > SCROLL_DEADZONE_PX:
                        mouse.scroll(int(delta * SCROLL_GAIN))
                        last_scroll_y = palm_y
                left_click.update(False)
                right_click.update(False)
                mouse.reset()

            else:
                mode = "ACTIVE"
                last_scroll_y = None
                mouse.move_to(*palm_center(landmarks))

                middle = is_middle_pinch(landmarks)
                right_click.update(middle)
                # The index tip drifts close to the thumb during a middle pinch,
                # so a plain is_pinch would fire a left click at the same time.
                left_click.update(is_pinch(landmarks) and not middle)

                if right_click.rose:
                    mouse.right_click()
                if left_click.rose:
                    mouse.click()

        else:
            mode = "PAUSED" if paused else "NO HAND"
            last_scroll_y = None
            for detector in (pause_toggle, left_click, right_click):
                detector.update(False)
            mouse.reset()

        bow.update(frame)
        draw_hud(frame, mode, timer.summary() if show_stats else None)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        timer.record("ui", time.perf_counter() - draw_started)

        if key == ord("q"):
            break
        if key == ord("d"):
            show_stats = not show_stats

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
