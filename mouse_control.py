import math

import pyautogui

# pyautogui aborts if the cursor is *already* in a screen corner when a call is
# made — a runaway-script brake that backfires here. Driving the cursor is this
# app's entire job, and the natural thing to do before using it is to shove the
# physical mouse out of the way, which usually means into a corner. The app then
# died on the first frame that saw a hand. A display change parks the pointer at
# (0, 0) too, which is the same crash wearing a different hat.
#
# The escape hatches that replace it are better suited to this app anyway: `q`
# in the camera window quits, and an open palm toggles PAUSED. `_clamp` below
# still keeps the cursor on-screen.
pyautogui.FAILSAFE = False

# pyautogui sleeps PAUSE seconds after *every* call it makes. The 0.1s default
# put a 100ms tax on the move_to that runs once per tracked frame, capping the
# capture loop at ~10 FPS whenever a hand was visible and letting it run free
# when none was — which is what the stutter actually was. For scale, hand
# detection itself costs ~30ms on this machine. Do not restore the default.
pyautogui.PAUSE = 0

# Cursor smoothing, in screen pixels. MediaPipe's landmarks wobble a few pixels
# frame to frame, which reads as a shaking cursor, but a fixed blend factor
# would damp that by adding lag to every real movement too. Blending in
# proportion to speed instead means a hand held still is smoothed hard while a
# hand moving fast is passed through almost untouched.
CURSOR_MIN_ALPHA = 0.18  # blend factor when the hand is stationary
CURSOR_ALPHA_SPAN = 90.0  # movement at which smoothing is fully off
CURSOR_DEADZONE_PX = 2  # below this the cursor is left alone entirely


class MouseController:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.screen_width, self.screen_height = pyautogui.size()
        self._smoothed = None
        self._last_sent = None

    def to_screen(self, x, y):
        """Frame coordinates to screen coordinates, before smoothing or clamping.

        Public so the desktop overlay can anchor the bow through exactly the
        function that positions the cursor, rather than through a second
        mapping that could drift from it.
        """
        return (
            x / self.frame_width * self.screen_width,
            y / self.frame_height * self.screen_height,
        )

    def move_to(self, x, y):
        target_x, target_y = self.to_screen(x, y)

        if self._smoothed is None:
            # The first move after a reset snaps. Easing in from wherever the
            # cursor happened to be left would show up as a visible drift every
            # time a hand re-enters the frame.
            self._smoothed = (target_x, target_y)
        else:
            previous_x, previous_y = self._smoothed
            moved = math.hypot(target_x - previous_x, target_y - previous_y)
            alpha = min(1.0, CURSOR_MIN_ALPHA + moved / CURSOR_ALPHA_SPAN)
            self._smoothed = (
                previous_x + (target_x - previous_x) * alpha,
                previous_y + (target_y - previous_y) * alpha,
            )

        position = self._clamp(int(self._smoothed[0]), int(self._smoothed[1]))
        if self._last_sent is not None and self._is_negligible(position):
            return

        pyautogui.moveTo(*position)
        self._last_sent = position

    def _is_negligible(self, position):
        # Measured against the last position actually sent, not the last
        # smoothed one, so that a slow drift still accumulates into a move
        # instead of being discarded a fraction of a pixel at a time.
        return (
            abs(position[0] - self._last_sent[0]) < CURSOR_DEADZONE_PX
            and abs(position[1] - self._last_sent[1]) < CURSOR_DEADZONE_PX
        )

    def reset(self):
        """Forget the smoothing state, so the next move_to snaps to its target."""
        self._smoothed = None
        self._last_sent = None

    def _clamp(self, x, y):
        # Scaling reaches the exact screen edges, so keep a pixel of margin: it
        # keeps the cursor on-screen and off GNOME's top-left hot corner, which
        # landmark jitter would otherwise trip into the Activities overview.
        return (
            min(max(x, 1), self.screen_width - 2),
            min(max(y, 1), self.screen_height - 2),
        )

    def click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    def scroll(self, clicks):
        pyautogui.scroll(clicks)
