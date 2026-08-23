"""Tests for the desktop overlay's pure half, plus its push protocol.

The I/O half is exercised through an injected fake connection: the real one
opens an X display, and the whole point of the module's shape is that none of
this needs a screen.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from desktop_overlay import (
    DesktopOverlay,
    OverlayGeometry,
    arrow_bounds,
    clip_rect,
    union_rect,
    MAX_REQUEST_BYTES,
)


def test_module_imports_without_a_display():
    """The direct analogue of the pyautogui hazard conftest.py works around:
    a module that connects at import time cannot be tested headlessly."""
    import desktop_overlay

    assert desktop_overlay.MAX_REQUEST_BYTES > 0


# --- rect algebra ---------------------------------------------------------

def test_union_rect_covers_both():
    assert union_rect((0, 0, 10, 10), (20, 5, 10, 10)) == (0, 0, 30, 15)


def test_union_rect_passes_through_a_missing_side():
    assert union_rect(None, (1, 2, 3, 4)) == (1, 2, 3, 4)
    assert union_rect((1, 2, 3, 4), None) == (1, 2, 3, 4)
    assert union_rect(None, None) is None


def test_clip_rect_trims_to_the_surface():
    assert clip_rect((-10, -10, 30, 30), 100, 100) == (0, 0, 20, 20)
    assert clip_rect((90, 90, 30, 30), 100, 100) == (90, 90, 10, 10)


def test_clip_rect_drops_a_rect_entirely_off_screen():
    assert clip_rect((-50, 0, 10, 10), 100, 100) is None
    assert clip_rect((200, 0, 10, 10), 100, 100) is None
    assert clip_rect(None, 100, 100) is None


def test_arrow_bounds_covers_every_arrow():
    arrows = [SimpleNamespace(x=100, y=100, reach=20),
              SimpleNamespace(x=200, y=50, reach=40)]

    x, y, width, height = arrow_bounds(arrows, margin=0)

    assert (x, y) == (80, 10)
    assert (x + width, y + height) == (240, 120)


def test_arrow_bounds_covers_the_streak_drawn_behind_the_arrow():
    """An arrow is drawn entirely behind the point it reports — shaft one length
    back, speed streak one length back again. Covering only one length leaves
    the streak on the desktop with nothing ever pushing over it."""
    arrow = SimpleNamespace(x=1000, y=500, reach=900)

    x, _, width, _ = arrow_bounds([arrow], margin=0)

    assert x <= 1000 - 900
    assert x + width >= 1000 + 900


def test_arrow_bounds_is_none_with_nothing_in_flight():
    assert arrow_bounds([]) is None


# --- geometry -------------------------------------------------------------

def _geometry(overlay_scale=1.0):
    # 640x480 frame onto a 2560x1440 screen: 4.0x across but only 3.0x down.
    return OverlayGeometry(
        (640, 480), (2560, 1440),
        anchor=lambda x, y: (x / 640 * 2560, y / 480 * 1440),
        bow_reach=3.4,
        overlay_scale=overlay_scale,
    )


def test_size_scale_takes_the_smaller_axis():
    assert _geometry().size_scale == 3.0
    assert _geometry(overlay_scale=0.5).size_scale == 1.5


def test_grip_lands_exactly_where_the_cursor_would_go():
    geometry = _geometry()

    grip, _, _ = geometry.map_pose((320, 240), (400, 240), 60)

    assert grip == (1280, 720)


def test_the_draw_angle_survives_the_anisotropic_screen():
    """The reason the nock is not mapped per-axis. Under a naive per-axis map a
    45 degree draw comes out at 36.9 degrees, so the bow would rotate the wrong
    way as the hands moved."""
    import math

    geometry = _geometry()
    grip, nock, _ = geometry.map_pose((320, 240), (220, 340), 60)

    angle = math.degrees(math.atan2(nock[1] - grip[1], nock[0] - grip[0]))
    assert angle == pytest.approx(135.0, abs=0.2)


def test_a_square_stays_square():
    geometry = _geometry()

    _, right, _ = geometry.map_pose((320, 240), (420, 240), 60)
    _, down, _ = geometry.map_pose((320, 240), (320, 340), 60)

    assert right[0] - 1280 == down[1] - 720 == 300


def test_hand_scale_is_multiplied_by_the_same_factor():
    _, _, scale = _geometry().map_pose((320, 240), (400, 240), 60)

    assert scale == 180.0


def test_pose_bounds_contains_the_bow_and_the_arrow_tip():
    geometry = _geometry()
    grip, nock, scale = (1280, 720), (1000, 720), 100.0

    x, y, width, height = geometry.pose_bounds(grip, nock, scale, margin=0)

    assert x <= grip[0] - 3.4 * scale and x <= nock[0]
    assert x + width >= grip[0] + 3.4 * scale
    assert y <= grip[1] - 3.4 * scale
    assert y + height >= grip[1] + 3.4 * scale


# --- the push protocol ----------------------------------------------------

class FakeDisplay:
    """Just enough X to record what the overlay would have sent."""

    def __init__(self, width=800, height=600):
        self.calls = []
        self._size = (width, height)
        visual = SimpleNamespace(visual_class=4, visual_id=0x75)
        depth = SimpleNamespace(depth=32, visuals=[visual])
        self.window = MagicMock()
        self.window.put_image.side_effect = self._record_put
        root = MagicMock()
        root.create_window.return_value = self.window
        self._screen = SimpleNamespace(
            allowed_depths=[depth], root=root,
            width_in_pixels=width, height_in_pixels=height,
        )

    def _record_put(self, gc, x, y, w, h, *rest):
        self.calls.append(("put", x, y, w, h, len(rest[-1])))

    def screen(self):
        return self._screen

    def intern_atom(self, name):
        return hash(name) % 1000

    def sync(self):
        pass

    def flush(self):
        self.calls.append(("flush",))

    def close(self):
        pass


def _overlay(width=800, height=600):
    fake = FakeDisplay(width, height)
    overlay = DesktopOverlay(width, height, connect=lambda: fake)
    assert overlay.open() is True
    fake.calls.clear()
    return overlay, fake


def test_an_idle_frame_sends_nothing():
    overlay, fake = _overlay()

    for _ in range(5):
        overlay.commit()

    assert fake.calls == []


def test_releasing_the_pose_sends_one_final_clearing_push():
    """The bug this guards is a bow left burned onto the desktop: the frame the
    pose breaks draws nothing, so without the union against the previous rect
    nothing would ever overwrite it."""
    overlay, fake = _overlay()

    overlay.mark((100, 100, 50, 50))
    overlay.commit()
    fake.calls.clear()

    overlay.commit()
    pushes = [call for call in fake.calls if call[0] == "put"]
    assert len(pushes) == 1
    assert pushes[0][1:5] == (100, 100, 50, 50)

    fake.calls.clear()
    overlay.commit()
    assert fake.calls == []


def test_the_pushed_rect_is_the_union_of_both_frames():
    overlay, fake = _overlay()

    overlay.mark((0, 0, 10, 10))
    overlay.commit()
    fake.calls.clear()

    overlay.mark((100, 100, 10, 10))
    overlay.commit()

    pushes = [call for call in fake.calls if call[0] == "put"]
    assert pushes[0][1:5] == (0, 0, 110, 110)


def test_every_chunk_is_followed_by_a_flush():
    """Without this the send buffer grows quadratically: a full-screen push
    measured 64 ms unflushed against 7.6 ms flushed, with no error to explain
    the difference."""
    overlay, fake = _overlay(2560, 1440)

    overlay.mark((0, 0, 2560, 1440))
    overlay.commit()

    kinds = [call[0] for call in fake.calls]
    assert kinds.count("put") > 1, "a full screen must be split into bands"
    assert kinds == ["put", "flush"] * kinds.count("put")


def test_no_chunk_exceeds_the_protocol_limit():
    overlay, fake = _overlay(2560, 1440)

    overlay.mark((0, 0, 2560, 1440))
    overlay.commit()

    for call in fake.calls:
        if call[0] == "put":
            assert call[5] <= MAX_REQUEST_BYTES


def test_the_canvas_is_wiped_after_it_is_pushed():
    overlay, _ = _overlay()
    overlay.canvas[100:150, 100:150] = 255
    overlay.mark((100, 100, 50, 50))

    overlay.commit()

    assert overlay.canvas.max() == 0


def test_a_failed_connection_leaves_a_silent_no_op():
    def refuse():
        raise OSError("no display")

    overlay = DesktopOverlay(800, 600, connect=refuse)

    assert overlay.open() is False
    assert overlay.available is False
    assert overlay.canvas is None
    overlay.mark((0, 0, 10, 10))
    overlay.commit()
    overlay.close()


def test_poll_reports_a_screen_resize_once_it_happens():
    overlay, fake = _overlay()

    assert overlay.poll(now=10.0) is None
    fake._screen.width_in_pixels = 1920
    fake._screen.height_in_pixels = 1080
    assert overlay.poll(now=11.5) == (1920, 1080)


def test_poll_is_rate_limited():
    overlay, fake = _overlay()
    overlay.poll(now=10.0)
    fake._screen.width_in_pixels = 1920

    assert overlay.poll(now=10.1) is None


def test_every_drawn_pixel_falls_inside_the_rect_that_gets_pushed():
    """The property that actually matters, and the one that caught a real bug:
    anything drawn but not pushed stays burned onto the desktop, because the
    canvas is wiped only where it was pushed. Reasoning about the bow's extent
    missed the arrow's speed streak; drawing it and looking did not."""
    import math

    from animations import BOW_HALF_LENGTH, HorseBow

    width, height = 1280, 720
    geometry = OverlayGeometry(
        (640, 480), (width, height),
        anchor=lambda x, y: (x / 640 * width, y / 480 * height),
        bow_reach=BOW_HALF_LENGTH,
    )
    bow = HorseBow(speed_scale=geometry.size_scale)
    canvas = np.zeros((height, width, 4), np.uint8)

    previous = pose = None
    drawn_last = False
    for frame in range(90):
        elapsed = frame / 15.0
        phase = elapsed % 3.0
        current = None
        if phase < 2.0:
            pull = min(1.0, phase / 1.6)
            grip = (420 + 40 * math.sin(elapsed), 240 + 30 * math.cos(elapsed))
            nock = (grip[0] - 40 - 110 * pull, grip[1] + 10 * pull)
            pose = geometry.map_pose(grip, nock, 55.0)
            bow.draw(canvas, *pose)
            current = geometry.pose_bounds(*pose)
            drawn_last = True
        elif drawn_last:
            bow.loose(*pose)
            drawn_last = False

        bow.update(canvas)
        current = union_rect(current, arrow_bounds(bow.arrows))
        pushed = clip_rect(union_rect(previous, current), width, height)

        rows = np.where(canvas[:, :, 3].any(axis=1))[0]
        cols = np.where(canvas[:, :, 3].any(axis=0))[0]
        if len(rows):
            assert pushed is not None, f"ink on frame {frame} with nothing pushed"
            x, y, pushed_width, pushed_height = pushed
            assert x <= cols[0] and y <= rows[0], f"ink escapes left/top, frame {frame}"
            assert x + pushed_width > cols[-1], f"ink escapes right, frame {frame}"
            assert y + pushed_height > rows[-1], f"ink escapes bottom, frame {frame}"

        previous = current
        if pushed:
            x, y, pushed_width, pushed_height = pushed
            canvas[y:y + pushed_height, x:x + pushed_width] = 0

    assert canvas[:, :, 3].max() == 0, "canvas left dirty after the last commit"
