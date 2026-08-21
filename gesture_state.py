class EdgeDetector:
    """Debounced rising/falling edge detection for a per-frame boolean signal.

    The capture loop sees a gesture on every frame it is held, so acting on the
    raw predicate fires an action dozens of times per gesture. Feed the raw
    signal in once per frame and act on `rose` / `fell` instead.

    Landmark detection is also noisy enough that a held gesture flickers off for
    the odd frame, so the signal has to hold for `min_frames` consecutive frames
    before `is_on` flips. That debounces both directions: a dropped frame will
    not fire a spurious release, and a twitch will not toggle a mode.
    """

    def __init__(self, min_frames=2):
        self.min_frames = max(1, min_frames)
        self.is_on = False
        self.rose = False
        self.fell = False
        self._pending = 0

    def update(self, active):
        self.rose = False
        self.fell = False

        if bool(active) == self.is_on:
            self._pending = 0
            return

        self._pending += 1
        if self._pending < self.min_frames:
            return

        self._pending = 0
        self.is_on = not self.is_on
        if self.is_on:
            self.rose = True
        else:
            self.fell = True
