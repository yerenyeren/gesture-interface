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


def draw_hud(frame, mode):
    for color, thickness in (((0, 0, 0), 5), (HUD_COLOR, 2)):
        cv2.putText(
            frame, mode, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness,
            cv2.LINE_AA,
        )


def main():
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = HandTracker(max_hands=2)
    mouse = MouseController(frame_width, frame_height)
    bow = HorseBow()

    # Actions fire on gesture edges, never on every frame the gesture is held.
    left_click = EdgeDetector()
    right_click = EdgeDetector()
    pause_toggle = EdgeDetector(min_frames=3)
    at_full_draw = EdgeDetector()

    paused = False
    last_scroll_y = None
    nocked = None  # (grip, nock, scale) from the most recent archery frame

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame, raw_hands = tracker.find_hands(frame)
        hands = [
            tracker.landmark_positions(hand, frame_width, frame_height)
            for hand in raw_hands
        ]

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

        bow.update(frame)
        draw_hud(frame, mode)

        cv2.imshow("Gesture Interface", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
