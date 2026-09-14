"""Procedurally drawn overlay animations that follow the hands.

Nothing here loads sprite assets: the bow is drawn from a control-point profile
every frame, which is what lets it scale, rotate and flex with the hands for
free. Every entry point takes a frame plus explicit points and returns nothing,
so the same calls will work against a transparent desktop overlay later.
"""

import math

import cv2
import numpy as np

# Colours are BGRA, matching OpenCV. The alpha is carried explicitly because
# OpenCV's colour argument is always a four-component scalar: on the 3-channel
# camera frame the fourth component is ignored, but on a 4-channel BGRA overlay
# canvas a 3-tuple would leave alpha at 0 and draw the stroke invisible. One set
# of constants therefore serves both surfaces.
BOW_COLOR = (32, 78, 140, 255)  # dark horn-and-sinew brown
BOW_HIGHLIGHT = (60, 140, 220, 255)  # lacquered highlight along the limb
STRING_COLOR = (225, 235, 245, 255)
SHAFT_COLOR = (150, 200, 235, 255)
HEAD_COLOR = (215, 235, 250, 255)
FLETCHING_COLOR = (70, 90, 225, 255)

# Sizes below are multiples of the grip hand's `hand_scale`, so the bow is drawn
# proportional to the hand rather than to a fixed pixel count.
BOW_HALF_LENGTH = 3.4
MAX_DRAW = 2.6  # draw length that counts as a full draw
MIN_DRAW = 1.0  # anything shorter than this is not a shot

# The arrow is one fixed length, nock to point, and that length is also the
# over-draw limit. Not by coincidence: draw the string back further than the
# arrow is long and its point has been pulled back past the grip, off the bow,
# so it falls. Kept clear of MAX_DRAW so a steady full draw has room for jitter.
ARROW_LENGTH = 3.2

ARROW_MIN_SPEED = 12.0  # pixels per frame at the minimum draw
ARROW_MAX_SPEED = 72.0  # pixels per frame at a full draw

# An arrow falling off the bow. Gravity is in hand scales per frame squared, like
# the sizes above, so the overlay's bigger pose already scales it: multiplying by
# `speed_scale` as well would make the desktop arrow fall faster than the camera's.
FALL_GRAVITY = 0.1
FALL_SPIN = 0.12  # radians per frame, tipping the point down

# Upper half-limb of a composite recurve, in bow-space units: the grip sits at
# the origin, +y runs out along the limb and +x points away from the archer.
# Read outward from the grip, the points are the riser shoulder, the working
# limb bellying forward, the string bridge, the base of the rigid siyah, and
# finally the ear tip. That last point hooks back past the limb axis (negative
# x) — the reflexed ear is what makes this a steppe horse bow rather than a
# plain arc, and it is where the string is tied on.
BRACED_PROFILE = (
    (0.00, 0.00),
    (0.06, 0.16),
    (0.22, 0.40),
    (0.30, 0.60),
    (0.22, 0.79),
    (-0.14, 0.90),
)

# The same limb at a full draw: it flexes back toward the archer and flattens
# out, which lengthens it slightly.
DRAWN_PROFILE = (
    (0.00, 0.00),
    (0.02, 0.17),
    (0.11, 0.43),
    (0.15, 0.66),
    (0.07, 0.84),
    (-0.22, 0.97),
)

# The first points of the profile are the flexing lath; the remainder is the
# rigid siyah, which is drawn straight and thinner because that is what it is.
WORKING_LIMB_POINTS = 4


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _normalize(vector):
    x, y = vector
    length = math.hypot(x, y)
    if length == 0:
        return (1.0, 0.0)
    return (x / length, y / length)


def bow_profile(ratio):
    """Control points of the upper half-limb at the given draw (0.0 to 1.0)."""
    t = min(1.0, max(0.0, ratio))
    return [
        (bx + (dx - bx) * t, by + (dy - by) * t)
        for (bx, by), (dx, dy) in zip(BRACED_PROFILE, DRAWN_PROFILE)
    ]


def _catmull_rom(points, samples_per_segment=8):
    """Smooth a control polygon into a curve that passes through every point."""
    if len(points) < 3:
        return list(points)

    padded = [points[0]] + list(points) + [points[-1]]
    curve = []
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i : i + 4]
        for step in range(samples_per_segment):
            t = step / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            curve.append(
                tuple(
                    0.5
                    * (
                        2 * p1[axis]
                        + (-p0[axis] + p2[axis]) * t
                        + (2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]) * t2
                        + (-p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]) * t3
                    )
                    for axis in (0, 1)
                )
            )
    curve.append(points[-1])
    return curve


def draw_ratio(grip, nock, scale):
    """How far the bow is drawn, 0.0 to 1.0, as a fraction of a full draw.

    This is how far the limbs flex, and nothing else. How fast the arrow leaves
    is `draw_power`, which starts counting at `MIN_DRAW` rather than at zero.
    """
    if scale <= 0:
        return 0.0
    return min(1.0, _dist(grip, nock) / (MAX_DRAW * scale))


def is_drawn(grip, nock, scale):
    """Is the bow drawn far enough for a shot to count at all?

    Like `is_overdrawn`, asked once by the caller for both bows, never by a bow.
    """
    return scale > 0 and _dist(grip, nock) >= MIN_DRAW * scale


def is_overdrawn(grip, nock, scale):
    """Has the string been drawn back past the end of the arrow?

    The bow never asks this of itself. The camera bow and the desktop bow are
    handed the same pose at two sizes, each rounded to whole pixels, so near the
    limit they would answer differently and one would drop an arrow the other
    still held. `main.py` asks once, of the camera pose, and tells both.
    """
    return scale > 0 and _dist(grip, nock) > ARROW_LENGTH * scale


def draw_power(grip, nock, scale):
    """How hard a shot from this draw is: 0.0 at `MIN_DRAW`, 1.0 at a full draw.

    Counted from the shortest draw that shoots rather than from zero. Counted
    from zero, the weakest possible shot already had 38% power, and the whole
    range a hand could feel was 40 to 70 pixels a frame.
    """
    if scale <= 0:
        return 0.0
    drawn = _dist(grip, nock) - MIN_DRAW * scale
    return min(1.0, max(0.0, drawn / ((MAX_DRAW - MIN_DRAW) * scale)))


def limb_sections(ratio):
    """The upper limb split into (working_lath, siyah), in bow-space.

    The lath is smoothed into a curve because it bends; the siyah is left as
    straight segments because on a composite bow it is a rigid horn ear. Drawing
    them separately is what makes the reflexed tip read as a hook rather than as
    one more bend in a bent stick.
    """
    profile = bow_profile(ratio)
    lath = _catmull_rom(profile[:WORKING_LIMB_POINTS])
    siyah = profile[WORKING_LIMB_POINTS - 1 :]
    return lath, siyah


def transform_points(points, origin, forward, scale):
    """Map bow-space points into frame pixels.

    `forward` is the unit aim vector (nock towards grip) and becomes bow-space
    +x; the limb axis is its perpendicular.
    """
    fx, fy = forward
    px, py = -fy, fx
    ox, oy = origin
    return [
        (
            int(ox + (x * fx + y * px) * scale),
            int(oy + (x * fy + y * py) * scale),
        )
        for x, y in points
    ]


def _fletching(length):
    """Feather length, and how far the feathers reach back past the nock.

    Shared with `Arrow.reach` rather than duplicated there: the overhang is the
    hindmost ink on an arrow, so a second copy of these numbers could drift and
    leave the tail of the fletching outside the rect the overlay pushes.
    """
    fletch = max(7.0, length * 0.11)
    return fletch, fletch * 0.2


def draw_arrow(frame, tip, direction, length, thickness):
    """Draw an arrow whose point is at `tip`, running back along `direction`."""
    dx, dy = direction
    px, py = -dy, dx
    tail = (tip[0] - dx * length, tip[1] - dy * length)

    head_length = max(8.0, length * 0.09)
    head_width = head_length * 0.5
    head_base = (tip[0] - dx * head_length, tip[1] - dy * head_length)

    cv2.line(
        frame,
        (int(tail[0]), int(tail[1])),
        (int(head_base[0]), int(head_base[1])),
        SHAFT_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    head = np.array(
        [
            (int(tip[0]), int(tip[1])),
            (int(head_base[0] + px * head_width), int(head_base[1] + py * head_width)),
            (int(head_base[0] - px * head_width), int(head_base[1] - py * head_width)),
        ],
        np.int32,
    )
    cv2.fillConvexPoly(frame, head, HEAD_COLOR, cv2.LINE_AA)

    # Fletching as swept-back feathers rather than two straight lines, which at
    # this size read as a second arrowhead pointing the wrong way.
    fletch_length, overhang = _fletching(length)
    fletch_width = fletch_length * 0.42
    fletch_base = (tail[0] + dx * fletch_length, tail[1] + dy * fletch_length)
    for side in (1, -1):
        # Widest at the nock and tapering forward along the shaft, so it reads
        # as a feather rather than as a chevron pointing back down the shaft.
        feather = np.array(
            [
                (int(tail[0]), int(tail[1])),
                (int(fletch_base[0]), int(fletch_base[1])),
                (
                    int(tail[0] - dx * overhang + px * fletch_width * side),
                    int(tail[1] - dy * overhang + py * fletch_width * side),
                ),
            ],
            np.int32,
        )
        cv2.fillConvexPoly(frame, feather, FLETCHING_COLOR, cv2.LINE_AA)


class Arrow:
    """An arrow flying across the frame until it leaves it.

    `HorseBow.nocked_arrow` also builds one at rest, for the arrow still on the
    string, so the arrow drawn nocked and the arrow loosed share all their code.
    """

    def __init__(self, position, velocity, length, direction=None):
        self.x, self.y = float(position[0]), float(position[1])
        self.vx, self.vy = velocity
        self.length = length
        # Stored rather than read off the velocity: an arrow still on the string
        # has no velocity to read, and a falling one points wherever it has
        # tumbled to rather than the way it is moving.
        self.direction = _normalize(velocity) if direction is None else direction
        self.alive = True

    def update(self, frame_width, frame_height):
        self.x += self.vx
        self.y += self.vy
        self._cull(frame_width, frame_height)

    def _cull(self, frame_width, frame_height):
        # `reach`, not `length`: the point tracked here is the arrowhead and
        # every other stroke is drawn behind it, so a `length` margin retires an
        # arrow while the last of its shaft and fletching is still inside the
        # near edge. It is the same bound the overlay pushes on, for the same
        # reason — what is still being drawn is still on screen.
        margin = self.reach
        self.alive = (
            -margin <= self.x <= frame_width + margin
            and -margin <= self.y <= frame_height + margin
        )

    @property
    def reach(self):
        """How far the drawing extends behind the tip.

        Callers that have to know where an arrow put ink — the desktop overlay,
        which pushes only what changed — need this rather than `length`, since
        the arrow is drawn entirely behind the point it reports.
        """
        # Ink runs a little past the nock, because the fletching's feathers are
        # swept back behind it — and every stroke has width. Both scale with the
        # arrow, so a fixed margin would stop covering them once the bow is
        # drawn big enough.
        return self.length + _fletching(self.length)[1] + max(2, int(self.length * 0.018))

    def draw(self, frame):
        thickness = max(2, int(self.length * 0.018))

        draw_arrow(frame, (self.x, self.y), self.direction, self.length, thickness)


class FallingArrow(Arrow):
    """An arrow that has slipped off an over-drawn bow and is falling.

    It turns about its middle rather than about the head it reports. That is
    what makes it look like it is falling, not swinging from a pin. The head is
    still the point tracked, and every stroke is still drawn behind it within
    `reach` whichever way it has turned, so culling and the overlay's
    `arrow_bounds` need to know nothing about it.
    """

    def __init__(self, position, direction, length, gravity, spin):
        super().__init__(position, (0.0, 0.0), length, direction=direction)
        self.gravity = gravity
        self.spin = spin

    def update(self, frame_width, frame_height):
        half = self.length / 2
        dx, dy = self.direction
        self.vy += self.gravity
        middle_x = self.x - dx * half + self.vx
        middle_y = self.y - dy * half + self.vy
        turn_cos, turn_sin = math.cos(self.spin), math.sin(self.spin)
        self.direction = (dx * turn_cos - dy * turn_sin, dx * turn_sin + dy * turn_cos)
        self.x = middle_x + self.direction[0] * half
        self.y = middle_y + self.direction[1] * half
        self._cull(frame_width, frame_height)


class HorseBow:
    """A nomadic composite recurve that follows the archer's hands.

    `draw` renders the bow held between the grip and string hands, with or
    without an arrow on the string. `loose` launches that arrow when the string
    hand opens, and `drop` lets it fall off an over-drawn bow. `update` keeps
    arrows moving, in flight or falling, after the pose has broken.

    Whether an arrow is on the string is not the bow's to decide (see
    `is_overdrawn`), so it is passed in rather than remembered here.
    """

    def __init__(self, speed_scale=1.0):
        self.arrows = []
        # ARROW_MIN_SPEED/ARROW_MAX_SPEED are the only quantities in this module
        # counted in absolute pixels rather than hand scales, so a bow drawn at
        # desktop size would fire arrows that crawl. Scaling them alongside the
        # bow is what keeps the shot looking the same on either surface.
        self.speed_scale = speed_scale

    @staticmethod
    def _aim(grip, nock):
        return _normalize((grip[0] - nock[0], grip[1] - nock[1]))

    def draw(self, frame, grip, nock, scale, *, arrow):
        if scale <= 0:
            return

        aim = self._aim(grip, nock)
        half_length = BOW_HALF_LENGTH * scale
        ratio = draw_ratio(grip, nock, scale)
        lath, siyah = limb_sections(ratio)
        shoulder = bow_profile(ratio)[1]

        limb = max(3, int(scale * 0.15))
        ear = max(2, limb * 2 // 3)
        ear_tips = []
        riser = []

        for mirror in (1, -1):
            for section, thickness in ((lath, limb), (siyah, ear)):
                points = transform_points(
                    [(x, y * mirror) for x, y in section], grip, aim, half_length
                )
                polyline = np.array(points, np.int32)
                cv2.polylines(
                    frame, [polyline], False, BOW_COLOR, thickness + 4, cv2.LINE_AA
                )
                cv2.polylines(
                    frame, [polyline], False, BOW_HIGHLIGHT, thickness, cv2.LINE_AA
                )
            ear_tips.append(points[-1])
            riser.append(
                transform_points(
                    [(shoulder[0], shoulder[1] * mirror)], grip, aim, half_length
                )[0]
            )

        # The riser: a thicker section spanning the grip between both limbs.
        cv2.line(frame, riser[0], riser[1], BOW_COLOR, limb + 8, cv2.LINE_AA)

        # The string runs from each ear tip down to the nocking point.
        string = max(1, limb // 3)
        for tip in ear_tips:
            cv2.line(frame, tip, nock, STRING_COLOR, string, cv2.LINE_AA)

        if arrow:
            self.nocked_arrow(grip, nock, scale).draw(frame)

    @classmethod
    def nocked_arrow(cls, grip, nock, scale):
        """The arrow on the string, at rest: its nock on the string, point ahead.

        `draw`, `loose` and `drop` all take their arrow from here, so an arrow
        leaving the bow starts exactly where the one on the string was drawn.
        That matters: `Arrow` reports its head and draws the shaft behind it.
        When `loose` once spawned the arrow at the nock instead, the head jumped
        a whole shaft backwards and every shot slid in from behind the archer.
        `main.py` bounds it from here too, so the overlay pushes exactly the
        ink that was drawn.
        """
        aim = cls._aim(grip, nock)
        length = ARROW_LENGTH * scale
        head = (nock[0] + aim[0] * length, nock[1] + aim[1] * length)
        return Arrow(head, (0.0, 0.0), length, direction=aim)

    def loose(self, grip, nock, scale):
        """Launch the arrow on the string. False only with no hand to draw from.

        Deliberately judges neither end of the draw. Whether it was long enough
        to shoot, and whether it was too long to still hold an arrow, are both
        decided once by the caller for both bows, for the reason `is_overdrawn`
        gives.
        """
        if scale <= 0:
            return False

        arrow = self.nocked_arrow(grip, nock, scale)
        power = draw_power(grip, nock, scale)
        speed = ARROW_MIN_SPEED + power * (ARROW_MAX_SPEED - ARROW_MIN_SPEED)
        speed *= self.speed_scale
        arrow.vx = arrow.direction[0] * speed
        arrow.vy = arrow.direction[1] * speed
        self.arrows.append(arrow)
        return True

    def drop(self, grip, nock, scale):
        """Let the arrow on the string fall off the bow. False with no hand."""
        if scale <= 0:
            return False

        nocked = self.nocked_arrow(grip, nock, scale)
        # y runs down the frame, so turning the way the arrow points tips its
        # point downward whichever way the bow is aimed.
        spin = math.copysign(FALL_SPIN, nocked.direction[0])
        self.arrows.append(
            FallingArrow(
                (nocked.x, nocked.y), nocked.direction, nocked.length,
                FALL_GRAVITY * scale, spin,
            )
        )
        return True

    def update(self, frame):
        if not self.arrows:
            return

        height, width = frame.shape[:2]
        for arrow in self.arrows:
            arrow.update(width, height)
            if arrow.alive:
                arrow.draw(frame)
        self.arrows = [arrow for arrow in self.arrows if arrow.alive]
