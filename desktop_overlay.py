"""A transparent, click-through overlay window over the whole desktop.

The bow is drawn into an ordinary BGRA numpy array with the same `cv2` calls
that draw it into the camera frame, and that array is pushed to an X window
that sits above everything and swallows no input.

Why X and not Wayland, on a Wayland session: Mutter 18 implements no
layer-shell protocol at all — the only surface role it offers a client is
`xdg_toplevel`, which means no client-side always-on-top and no click-through.
The route that does work is an override-redirect ARGB window through XWayland,
made click-through with an empty SHAPE input region.

Geometry is separated from I/O the same way `animations.py` separates
`bow_profile` from rendering: everything above `DesktopOverlay` is pure and
testable with no display, and importing this module opens no connection.
"""

import math

import numpy as np
from Xlib import X, Xatom, Xutil, display

# The X request-length field is a Card16 counting 4-byte units and python-xlib
# implements no BIG-REQUESTS extension, so no single PutImage may carry more
# than this. A full-screen frame is far larger and has to go in bands.
MAX_REQUEST_BYTES = (65535 - 6) * 4


def union_rect(first, second):
    """Smallest rect covering both, or whichever one exists."""
    if first is None:
        return second
    if second is None:
        return first
    x0 = min(first[0], second[0])
    y0 = min(first[1], second[1])
    x1 = max(first[0] + first[2], second[0] + second[2])
    y1 = max(first[1] + first[3], second[1] + second[3])
    return (x0, y0, x1 - x0, y1 - y0)


def clip_rect(rect, width, height):
    """Trim a rect to the surface, or None if nothing of it is left."""
    if rect is None:
        return None
    x0 = max(0, rect[0])
    y0 = max(0, rect[1])
    x1 = min(width, rect[0] + rect[2])
    y1 = min(height, rect[1] + rect[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def arrow_bounds(arrows, margin=8):
    """Rect covering arrows in flight, or None.

    Duck-typed on `.x`, `.y` and `.reach` rather than importing `animations`,
    which keeps this module free of cross-imports.
    """
    if not arrows:
        return None
    # An arrow is drawn entirely *behind* the point it reports — shaft, then
    # fletching — so `reach`, not `length`, is what bounds it. Taking that
    # either side covers it whichever way it is flying, and getting it wrong
    # leaves ink on the desktop that nothing ever pushes over.
    reach = [arrow.reach + margin for arrow in arrows]
    x0 = min(arrow.x - r for arrow, r in zip(arrows, reach))
    y0 = min(arrow.y - r for arrow, r in zip(arrows, reach))
    x1 = max(arrow.x + r for arrow, r in zip(arrows, reach))
    y1 = max(arrow.y + r for arrow, r in zip(arrows, reach))
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


class OverlayGeometry:
    """Camera-frame points to screen points: anchored map, isotropic body.

    The screen map is anisotropic here — 640 to 2560 is 4.0x across but 480 to
    1440 is only 3.0x down — so mapping the bow's points per-axis would skew the
    *aim*, not merely the size: the bow takes its direction from `grip - nock`,
    and a 45 degree draw would come out at 36.9 degrees and rotate the wrong way
    as the hands moved.

    So only the grip is mapped through the anisotropic screen map — the same
    function that positions the cursor, so the bow lands where the app already
    says that hand is — and the nock and the hand scale are derived from it with
    one isotropic factor, which leaves every angle and proportion intact.
    """

    def __init__(self, frame_size, screen_size, anchor, bow_reach,
                 overlay_scale=1.0):
        self.frame_width, self.frame_height = frame_size
        self.screen_width, self.screen_height = screen_size
        self.anchor = anchor
        self.bow_reach = bow_reach
        self.overlay_scale = overlay_scale

    @property
    def size_scale(self):
        """The single isotropic factor sizes and offsets are multiplied by.

        `min` of the two axes rather than `max` or their mean, because it is the
        only choice that cannot make the bow overflow the screen worse than it
        already fills the camera frame.
        """
        return self.overlay_scale * min(
            self.screen_width / self.frame_width,
            self.screen_height / self.frame_height,
        )

    def map_pose(self, grip, nock, hand_scale):
        anchored = self.anchor(*grip)
        factor = self.size_scale
        mapped_nock = (
            anchored[0] + (nock[0] - grip[0]) * factor,
            anchored[1] + (nock[1] - grip[1]) * factor,
        )
        return (
            (int(anchored[0]), int(anchored[1])),
            (int(mapped_nock[0]), int(mapped_nock[1])),
            hand_scale * factor,
        )

    def pose_bounds(self, grip, nock, scale, margin=24):
        """Rect covering the drawn bow, its string and its nocked arrow.

        Closed form rather than scanning the canvas for ink: measured, a
        full-screen `max` over the alpha channel costs ~92 ms, which is more
        than a whole frame.
        """
        reach = self.bow_reach * scale
        drawn = math.hypot(grip[0] - nock[0], grip[1] - nock[1])
        # The nocked arrow points from the nock through the grip and out past it.
        arrow_tip = drawn + reach * 0.45
        if drawn > 0:
            direction = ((grip[0] - nock[0]) / drawn, (grip[1] - nock[1]) / drawn)
        else:
            direction = (0.0, 0.0)
        tip = (nock[0] + direction[0] * arrow_tip,
               nock[1] + direction[1] * arrow_tip)

        xs = (grip[0] - reach, grip[0] + reach, nock[0], tip[0])
        ys = (grip[1] - reach, grip[1] + reach, nock[1], tip[1])
        x0, y0 = int(min(xs)) - margin, int(min(ys)) - margin
        x1, y1 = int(max(xs)) + margin, int(max(ys)) + margin
        return (x0, y0, x1 - x0, y1 - y0)


class DesktopOverlay:
    """The X window and the pixels that go on it.

    Every public method swallows its own errors and degrades to a no-op with
    `available` false. The camera window is the app's quit path, so an overlay
    failure must never take down a loop that is actively driving the cursor.
    """

    def __init__(self, width, height, connect=None):
        self.width = width
        self.height = height
        self.available = False
        self.canvas = None
        self.last_rect = None
        self._connect = connect or display.Display
        self._display = None
        self._window = None
        self._gc = None
        self._previous = None
        self._current = None
        self._checked_at = 0.0

    # --- lifecycle --------------------------------------------------------

    def open(self):
        try:
            self._display = self._connect()
            screen = self._display.screen()
            visual_id = self._argb_visual(screen)
            if visual_id is None:
                raise RuntimeError("no depth-32 TrueColor visual for alpha")

            colormap = screen.root.create_colormap(visual_id, X.AllocNone)
            # A depth-32 window on a depth-24 root needs visual, colormap,
            # background *and* border pixel spelled out; omitting border_pixel
            # is a BadMatch that reads as nothing in particular.
            self._window = screen.root.create_window(
                0, 0, self.width, self.height, 0, 32, X.InputOutput, visual_id,
                colormap=colormap, background_pixel=0, border_pixel=0,
                override_redirect=True, event_mask=0,
            )
            self._set_click_through()
            self._set_properties()
            self._window.map()
            self._display.sync()
            self._gc = self._window.create_gc()
            self.canvas = np.zeros((self.height, self.width, 4), np.uint8)
            self.available = True
        except Exception as error:  # noqa: BLE001 - never take the app down
            print(f"desktop overlay unavailable, carrying on without it: {error}")
            self.available = False
        return self.available

    def _argb_visual(self, screen):
        for depth in screen.allowed_depths:
            if depth.depth == 32:
                for visual in depth.visuals:
                    if visual.visual_class == X.TrueColor:
                        return visual.visual_id
        return None

    def _set_click_through(self):
        # An empty input region: every click falls through to what is beneath.
        # python-xlib's xfixes has no SetWindowShapeRegion, so SHAPE is the only
        # route to this.
        from Xlib.ext import shape

        self._window.shape_rectangles(shape.SO.Set, shape.SK.Input, 0, 0, 0, [])

    def _set_properties(self):
        atom = self._display.intern_atom
        self._window.change_property(
            atom("_NET_WM_WINDOW_TYPE"), Xatom.ATOM, 32,
            [atom("_NET_WM_WINDOW_TYPE_DOCK")],
        )
        self._window.change_property(
            atom("_NET_WM_STATE"), Xatom.ATOM, 32,
            [atom("_NET_WM_STATE_ABOVE"), atom("_NET_WM_STATE_SKIP_TASKBAR"),
             atom("_NET_WM_STATE_SKIP_PAGER")],
        )
        # Never accept keyboard focus: `q` in the camera window has to keep
        # reaching the camera window.
        self._window.set_wm_hints(flags=Xutil.InputHint, input=0)
        self._window.set_wm_name("gesture interface overlay")

    def close(self):
        self.available = False
        try:
            if self._gc is not None:
                self._gc.free()
            if self._window is not None:
                self._window.destroy()
            if self._display is not None:
                self._display.sync()
                self._display.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
        self._gc = self._window = self._display = None

    # --- per frame --------------------------------------------------------

    def mark(self, rect):
        """Declare where this frame drew, so `commit` knows what to push."""
        self._current = union_rect(self._current, rect)

    def commit(self):
        """Push what changed, then wipe the canvas ready for the next frame.

        The union with the previous frame's rect is what erases the last bow:
        when the pose breaks, `_current` is empty but `_previous` is not, and
        that one final push is the difference between the bow disappearing and
        it staying burned onto the desktop until the process exits.
        """
        if not self.available:
            return
        rect = clip_rect(union_rect(self._previous, self._current),
                         self.width, self.height)
        self._previous = self._current
        self._current = None
        self.last_rect = rect
        if rect is None:
            return
        try:
            self._push(rect)
        except Exception as error:  # noqa: BLE001
            print(f"desktop overlay lost, carrying on without it: {error}")
            self.available = False
            return
        x, y, width, height = rect
        self.canvas[y:y + height, x:x + width] = 0

    def _push(self, rect):
        x, y, width, height = rect
        region = self.canvas[y:y + height, x:x + width]
        rows = max(1, MAX_REQUEST_BYTES // (width * 4))
        for offset in range(0, height, rows):
            band = np.ascontiguousarray(region[offset:offset + rows])
            self._window.put_image(
                self._gc, x, y + offset, width, band.shape[0],
                X.ZPixmap, 32, 0, band.tobytes(),
            )
            # Flushing every band is not optional. python-xlib appends each
            # request to one growing bytes buffer, so without this the cost of a
            # full-screen push is quadratic: measured 64 ms against 7.6 ms.
            self._display.flush()

    def poll(self, now, interval=1.0):
        """Report a screen resize, at most once per `interval` seconds.

        The window is sized once at `open`, and a monitor change has broken this
        app before — see the FAILSAFE note in `mouse_control.py`.
        """
        if not self.available or now - self._checked_at < interval:
            return None
        self._checked_at = now
        try:
            screen = self._display.screen()
            size = (screen.width_in_pixels, screen.height_in_pixels)
        except Exception:  # noqa: BLE001
            self.available = False
            return None
        if size == (self.width, self.height):
            return None
        return size
