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
        self._button_down = False

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

    def snap_to(self, x, y):
        """Put the cursor on the target now, skipping smoothing and the deadzone.

        For the frame a click fires on. `move_to`'s speed-adaptive blend leaves
        the cursor trailing the hand by up to
        `(1.0 - CURSOR_MIN_ALPHA) * CURSOR_ALPHA_SPAN` screen pixels — about 74
        here — and pressing there means the following frames ease off that lag
        with the button held. That residual travel is a short drag on the end of
        every click, which is exactly what a click must not be.

        `_smoothed` is set to the unrounded target rather than the clamped
        integer so the `move_to` on the next frame starts from the target and
        re-introduces no step of its own, and the move is emitted
        unconditionally: the deadzone exists to stop jitter from generating
        traffic, but a click has to land on its point however small the
        correction is.
        """
        target_x, target_y = self.to_screen(x, y)
        self._smoothed = (target_x, target_y)

        position = self._clamp(int(target_x), int(target_y))
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

    @property
    def position(self):
        """The last screen position actually sent, or None before the first move.

        The caller needs it to tell whether a click would land on this app's
        own window; reading it back from pyautogui would report where the
        physical mouse is, which is not the same thing.
        """
        return self._last_sent

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

    @property
    def is_pressed(self):
        return self._button_down

    def _button_action(self, action):
        """Fire a pyautogui button event at the point this app last sent.

        Never call these with no coordinates. pyautogui then fills them in from
        `position()`, which is `query_pointer` on the X root — and under
        XWayland that reports XWayland's *cached* pointer, not the
        compositor's. The cache only updates while the pointer is over an
        XWayland surface, so it goes stale the moment the cursor crosses a
        native Wayland window, and pyautogui warps the real pointer back to
        that stale value before the button event — twice, once in `mouseDown`
        and again inside the X11 backend's `_mouseDown`. Every click therefore
        landed at the *previous* click's position, and the following frames
        moved forward with the button held: not a click, a selection dragged
        from wherever the last one happened to be.

        Note what `position` above already says — asking pyautogui where the
        cursor is answers the wrong question. This module knew that; pyautogui
        was asking anyway, behind its back. Passing coordinates explicitly
        makes `_normalizeXYArgs` hand them straight back, so no query happens
        and both warps land on the point the app actually chose.
        """
        if self._last_sent is None:
            action()  # before the first move the app has no opinion
        else:
            action(*self._last_sent)

    def press(self):
        """Hold the left button down. Idempotent.

        A left click is a press held for as long as the pinch is, rather than a
        complete click on the pinch's leading edge, because that is what a drag
        is: holding the button while the cursor moves is how text gets
        highlighted and how anything gets dragged. A quick pinch still reads as
        an ordinary click, since press and release land a frame or two apart.

        Goes through `_button_action` so the press lands where this app put the
        cursor rather than where pyautogui guesses it is.
        """
        if not self._button_down:
            self._button_action(pyautogui.mouseDown)
            self._button_down = True

    def release(self):
        """Let the left button up. Idempotent, and safe to call from anywhere.

        Called on *every* path out of the dragging state — hand lost, paused,
        scrolling, an exception on the way to the exit — because a left button
        left down is not a missed click. It is a desktop that keeps selecting
        everything the cursor touches until the app is killed, and the app owns
        the cursor, so recovering by hand is awkward.

        Goes through `_button_action` for the same reason `press` does: a
        release warped back to a stale pointer ends the drag somewhere the hand
        never was.
        """
        if self._button_down:
            self._button_action(pyautogui.mouseUp)
            self._button_down = False

    def right_click(self):
        # Same stale-pointer trap as press/release: `rightClick()` with no
        # coordinates takes the identical `_normalizeXYArgs(None, None)` path.
        self._button_action(pyautogui.rightClick)

    def scroll(self, clicks):
        pyautogui.scroll(clicks)
