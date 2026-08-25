import math
import time

import cv2

from animations import HorseBow, draw_ratio, BOW_HALF_LENGTH, MAX_DRAW, MIN_DRAW
from desktop_overlay import DesktopOverlay, OverlayGeometry, arrow_bounds
from gesture_state import EdgeDetector
from gestures import (
    gesture_metrics,
    hand_scale,
    is_fist,
    is_middle_pinch,
    is_ok_sign,
    is_pinch,
    is_two_fingers_up,
    palm_center,
    pinch_point,
    FINGER_CURLED_RATIO,
    FINGER_EXTENDED_RATIO,
)
from hand_tracker import HandTracker
from mouse_control import MouseController

# How big the bow is drawn on the desktop overlay, relative to the size it is
# drawn in the camera window. 1.0 keeps the same proportions it has on camera,
# which also means it fills the screen the way it fills the frame — the first
# thing worth turning down if that is too much. Shown on the `t` readout.
OVERLAY_SCALE = 1.0

SCROLL_DEADZONE_PX = 12
SCROLL_GAIN = 0.4
HUD_COLOR = (240, 240, 240)
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX

# A finger between the two ratios is neither extended nor curled, so every
# gesture that uses it silently fails to fire. That state has no boolean of its
# own, which is exactly why the tuning readout gives it a colour and a mark.
FINGER_MARKS = {"extended": "^", "curled": "v", "dead": "?"}
FINGER_STATE_COLORS = {
    "extended": (140, 240, 140),
    "curled": (150, 200, 255),
    "dead": (90, 90, 255),
}
FINGER_NAMES = ("I", "M", "R", "P")

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


def finger_state(ratio):
    """Which of the three states a finger ratio falls in."""
    if ratio > FINGER_EXTENDED_RATIO:
        return "extended"
    if ratio < FINGER_CURLED_RATIO:
        return "curled"
    return "dead"


def metric_segments(label, landmarks):
    """Two lines of (text, colour) runs describing what the predicates see.

    Returned as runs rather than strings so each finger can be coloured by its
    own state — the numbers alone do not show which side of a threshold they
    fell on, and that is the whole reason for looking at them.
    """
    metrics = gesture_metrics(landmarks)
    threshold = metrics["pinch_threshold_px"]

    ratios = [(f"{label:<7}", HUD_COLOR)]
    for name, ratio in zip(FINGER_NAMES, metrics["ratios"]):
        state = finger_state(ratio)
        ratios.append(
            (f"{name} {ratio:.2f}{FINGER_MARKS[state]}  ", FINGER_STATE_COLORS[state])
        )

    pinch = [
        (" " * 7, HUD_COLOR),
        (
            f"pinch idx {metrics['index_pinch_px']:.0f}  mid "
            f"{metrics['middle_pinch_px']:.0f}  < {threshold:.0f}   "
            f"scale {metrics['scale_px']:.0f}px",
            HUD_COLOR,
        ),
    ]
    return [ratios, pinch]


def draw_hud(frame, mode, stats=None, metrics=None):
    """Mode, optional timings, and optional per-hand tuning numbers."""
    y = 34
    _draw_runs(frame, [(mode, HUD_COLOR)], y, scale=0.8, outline=5, weight=2)

    y += 28
    if stats is not None:
        _draw_runs(frame, [(stats, HUD_COLOR)], y)
        y += 20

    for runs in metrics or ():
        _draw_runs(frame, runs, y)
        y += 20


def _draw_runs(frame, runs, y, scale=0.5, outline=3, weight=1):
    x = 14
    for text, color in runs:
        for shade, thickness in (((0, 0, 0), outline), (color, weight)):
            cv2.putText(
                frame, text, (x, y), HUD_FONT, scale, shade, thickness, cv2.LINE_AA,
            )
        x += cv2.getTextSize(text, HUD_FONT, scale, weight)[0][0]


def overlay_geometry(mouse, frame_width, frame_height):
    return OverlayGeometry(
        (frame_width, frame_height),
        (mouse.screen_width, mouse.screen_height),
        anchor=mouse.to_screen,
        bow_reach=BOW_HALF_LENGTH,
        overlay_scale=OVERLAY_SCALE,
    )


def rebuild_for_screen(overlay, size, frame_width, frame_height):
    """Rebuild everything that cached the screen size, after a monitor change.

    `MouseController` reads `pyautogui.size()` once in its constructor, so it
    goes stale on a display change too — the same staleness that used to crash
    this app from the other direction.
    """
    overlay.close()
    mouse = MouseController(frame_width, frame_height)
    overlay = DesktopOverlay(*size)
    overlay.open()
    geometry = overlay_geometry(mouse, frame_width, frame_height)
    return mouse, overlay, geometry, HorseBow(speed_scale=geometry.size_scale)


def metrics_readout(measured, nocked, overlay=None, geometry=None):
    """Tuning lines for the hands in play, plus the draw length when drawing."""
    lines = []
    for label, landmarks in measured:
        lines.extend(metric_segments(label, landmarks))

    if nocked is not None:
        grip, nock, scale = nocked
        length = math.hypot(grip[0] - nock[0], grip[1] - nock[1]) / scale if scale else 0
        lines.append([
            (
                f"{'draw':<7}{length:.2f}x hand   ratio "
                f"{draw_ratio(grip, nock, scale):.2f}   "
                f"loose > {MIN_DRAW}   full at {MAX_DRAW}",
                HUD_COLOR,
            )
        ])

    if overlay is not None and geometry is not None:
        rect = overlay.last_rect
        pushed = f"{rect[2]}x{rect[3]}" if rect else "idle"
        lines.append([
            (
                f"{'overlay':<7}{'on' if overlay.available else 'OFF'}   "
                f"{overlay.width}x{overlay.height}   "
                f"size x{geometry.size_scale:.2f} "
                f"(OVERLAY_SCALE {geometry.overlay_scale})   push {pushed}",
                HUD_COLOR,
            )
        ])
    return lines


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

    overlay = DesktopOverlay(mouse.screen_width, mouse.screen_height)
    overlay.open()
    geometry = overlay_geometry(mouse, frame_width, frame_height)
    # A second bow, in screen coordinates. Not the same instance drawn twice:
    # `update` both advances and draws arrows, so sharing one would advance
    # every arrow twice per frame.
    screen_bow = HorseBow(speed_scale=geometry.size_scale)
    overlay_on = True

    # Actions fire on gesture edges, never on every frame the gesture is held.
    left_click = EdgeDetector()
    right_click = EdgeDetector()
    at_full_draw = EdgeDetector()

    paused = False
    show_stats = True
    show_skeleton = True
    show_metrics = False
    last_scroll_y = None
    nocked = None  # (grip, nock, scale) from the most recent archery frame

    try:
        while True:
            started = time.perf_counter()
            success, frame = cap.read()
            if not success:
                break
            timer.record("cap", time.perf_counter() - started)

            detect_started = time.perf_counter()
            frame = cv2.flip(frame, 1)
            frame, raw_hands = tracker.find_hands(frame, draw=False)
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
                if overlay.available and overlay_on:
                    screen_bow.loose(*geometry.map_pose(*nocked))
                nocked = None

            measured = []
            if archery is not None:
                grip_hand, string_hand = archery
                nocked = (
                    palm_center(grip_hand),
                    pinch_point(string_hand),
                    hand_scale(grip_hand),
                )
                bow.draw(frame, *nocked)
                if overlay.available and overlay_on:
                    pose = geometry.map_pose(*nocked)
                    screen_bow.draw(overlay.canvas, *pose)
                    overlay.mark(geometry.pose_bounds(*pose))
                mode = "DRAWING BOW"
                measured = [("grip", grip_hand), ("string", string_hand)]
                # The mouse deliberately sits idle: the string hand is pinching,
                # which would otherwise read as a click.
                for detector in (left_click, right_click):
                    detector.update(False)
                last_scroll_y = None
                mouse.reset()

            elif hands:
                landmarks = control_hand(hands)
                measured = [("hand", landmarks)]

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
                for detector in (left_click, right_click):
                    detector.update(False)
                mouse.reset()

            # Drawn here rather than inside detection because only now is the mode
            # known — and the bow pose wants the frame to itself. Being after the
            # bow also puts the skeleton under the arrows instead of over them.
            if show_skeleton and mode != "DRAWING BOW":
                for landmarks in hands:
                    tracker.draw_landmarks(frame, landmarks)

            bow.update(frame)

            overlay_started = time.perf_counter()
            if overlay.available:
                if overlay_on:
                    screen_bow.update(overlay.canvas)
                    overlay.mark(arrow_bounds(screen_bow.arrows))
                # Committed even when switched off, so the last bow drawn gets
                # erased instead of staying burned onto the desktop.
                overlay.commit()
                resized = overlay.poll(time.monotonic())
                if resized is not None:
                    mouse, overlay, geometry, screen_bow = rebuild_for_screen(
                        overlay, resized, frame_width, frame_height
                    )
            timer.record("ovl", time.perf_counter() - overlay_started)

            draw_hud(
                frame,
                mode,
                timer.summary() if show_stats else None,
                metrics_readout(measured, nocked if archery else None,
                                overlay if overlay_on else None, geometry)
                if show_metrics
                else None,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            timer.record("ui", time.perf_counter() - draw_started)

            if key == ord("q"):
                break
            if key == ord("d"):
                show_stats = not show_stats
            if key == ord("s"):
                show_skeleton = not show_skeleton
            if key == ord("t"):
                show_metrics = not show_metrics
            if key == ord("o"):
                overlay_on = not overlay_on
            if key == ord("p"):
                # A key rather than a gesture: a pause that fires when it was
                # not asked for takes the cursor away, which is the most
                # expensive false positive the app has.
                paused = not paused
                mouse.reset()

    finally:
        # Runs even if the loop raises. Before this, an exception mid-loop left
        # the camera held and — now — a fullscreen window on the desktop.
        overlay.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
