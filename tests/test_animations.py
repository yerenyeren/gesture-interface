import math

import numpy as np
import pytest

import animations
from animations import (
    Arrow,
    FallingArrow,
    HorseBow,
    bow_profile,
    draw_power,
    draw_ratio,
    is_drawn,
    is_overdrawn,
    transform_points,
    ARROW_LENGTH,
    ARROW_MAX_SPEED,
    ARROW_MIN_SPEED,
    BOW_HALF_LENGTH,
    BOW_COLOR,
    BOW_HIGHLIGHT,
    FLETCHING_COLOR,
    HEAD_COLOR,
    MAX_DRAW,
    MIN_DRAW,
    SHAFT_COLOR,
    STRING_COLOR,
)


def test_bow_profile_starts_at_the_grip():
    assert bow_profile(0.0)[0] == (0.0, 0.0)
    assert bow_profile(1.0)[0] == (0.0, 0.0)


def test_bow_profile_ear_tip_hooks_back_past_the_limb_axis():
    # The reflexed ear is the horse-bow signature: it must sit behind the grip
    # (negative x) at every draw length, not just when braced.
    for draw_ratio in (0.0, 0.5, 1.0):
        assert bow_profile(draw_ratio)[-1][0] < 0


def test_drawing_the_bow_flexes_the_limb_toward_the_archer():
    braced = bow_profile(0.0)
    drawn = bow_profile(1.0)

    for (braced_x, _), (drawn_x, _) in zip(braced[1:], drawn[1:]):
        assert drawn_x < braced_x


def test_drawing_the_bow_lengthens_the_flattened_limb():
    assert bow_profile(1.0)[-1][1] > bow_profile(0.0)[-1][1]


def test_draw_ratio_is_clamped_to_the_braced_and_full_profiles():
    assert bow_profile(-2.0) == bow_profile(0.0)
    assert bow_profile(5.0) == bow_profile(1.0)


def test_transform_points_maps_bow_space_x_onto_the_aim_direction():
    assert transform_points([(1.0, 0.0)], (0, 0), (1.0, 0.0), 10) == [(10, 0)]


def test_transform_points_maps_bow_space_y_onto_the_limb_axis():
    assert transform_points([(0.0, 1.0)], (0, 0), (1.0, 0.0), 10) == [(0, 10)]


def test_transform_points_rotates_with_the_aim_direction():
    # Aiming straight down the screen swings the bow's forward axis with it.
    assert transform_points([(1.0, 0.0)], (0, 0), (0.0, 1.0), 10) == [(0, 10)]


def test_transform_points_translates_to_the_grip():
    assert transform_points([(0.0, 0.0)], (300, 200), (1.0, 0.0), 10) == [(300, 200)]


def test_arrow_stays_alive_inside_the_frame():
    arrow = Arrow((100, 100), (10, 0), length=50)

    arrow.update(640, 480)

    assert (arrow.x, arrow.y) == (110.0, 100.0)
    assert arrow.alive is True


def test_arrow_dies_once_it_leaves_the_frame():
    arrow = Arrow((600, 100), (100, 0), length=10)

    arrow.update(640, 480)

    assert arrow.alive is False


def test_an_arrow_is_not_culled_while_its_fletching_is_still_on_screen():
    """The same anchor mismatch as the spawn, at the far end of the flight: the
    arrow reports its head and is drawn entirely behind it, so a `length` margin
    retires it with the tail of the shaft still inside the frame and the last
    thing the viewer sees is a shot vanishing early."""
    arrow = Arrow((-295.0, 200.0), (-10.0, 0.0), length=300.0)

    arrow.update(640, 480)

    # The head is past the left edge by more than `length` but less than
    # `reach`, which is exactly the band the fletching occupies.
    assert arrow.length < -arrow.x < arrow.reach
    assert arrow.alive is True


def test_loose_launches_an_arrow_toward_the_grip():
    bow = HorseBow()
    # Grip to the right of the nock, so the arrow should fly right.
    nock, scale = (100, 200), 50.0
    grip = (100 + int(MAX_DRAW * scale), 200)

    assert bow.loose(grip, nock, scale) is True
    assert len(bow.arrows) == 1

    arrow = bow.arrows[0]
    # An arrow reports its head, and the head sits a full shaft ahead of the
    # nock — where the bow was already drawing it while it was on the string.
    assert (arrow.x, arrow.y) == (100.0 + arrow.length, 200.0)
    assert arrow.vx > 0
    assert arrow.vy == 0


def test_a_loosed_arrow_starts_where_the_nocked_arrow_was_left(monkeypatch):
    """The regression test for the arrow that flew in from beyond the screen.

    `draw` puts the nocked arrow's head a full shaft ahead of the nock and
    `Arrow` reports its head too, so constructing one *at* the nock threw the
    head one whole arrow length backwards, behind the archer, and the shot spent
    its first frames sliding in from off-screen towards a point it had already
    occupied. Release has to be seamless: the loosed arrow's first frame puts
    its tip where the nocked arrow's tip was."""
    nocked_tips = []
    real_draw_arrow = animations.draw_arrow

    def spy(frame, tip, direction, length, thickness):
        nocked_tips.append(tip)
        real_draw_arrow(frame, tip, direction, length, thickness)

    monkeypatch.setattr(animations, "draw_arrow", spy)

    bow = HorseBow()
    nock, scale = (100, 200), 50.0
    grip = (100 + int(MAX_DRAW * scale), 200)

    bow.draw(np.zeros((400, 600, 4), np.uint8), grip, nock, scale, arrow=True)
    assert bow.loose(grip, nock, scale) is True

    # Asking the bow where it actually drew the head rather than recomputing the
    # shaft length here: a second copy of that formula in the test would drift
    # alongside the one in `draw` and stop pinning the two together at all.
    (nocked_tip,) = nocked_tips
    arrow = bow.arrows[0]
    assert (arrow.x, arrow.y) == pytest.approx(nocked_tip)


def test_a_fuller_draw_launches_a_faster_arrow():
    bow = HorseBow()
    nock, scale = (100, 200), 50.0

    bow.loose((100 + int(MIN_DRAW * scale) + 5, 200), nock, scale)
    bow.loose((100 + int(MAX_DRAW * scale), 200), nock, scale)

    weak, full = bow.arrows
    assert full.vx > weak.vx


def test_scaleless_hand_draws_nothing_and_shoots_nothing():
    bow = HorseBow()

    assert bow.loose((300, 200), (100, 200), 0.0) is False
    assert bow.arrows == []
    # A zero scale would blow up the bow geometry, so draw must bail out too.
    assert bow.draw(None, (300, 200), (100, 200), 0.0, arrow=True) is None


def test_bow_half_length_is_expressed_in_scales():
    # Sizes are multiples of the pose's scale rather than pixel counts. The
    # caller fixes that scale; it is no longer the grip hand's size.
    assert BOW_HALF_LENGTH > 0
    assert MIN_DRAW < MAX_DRAW


def test_draw_ratio_is_zero_for_a_scaleless_hand():
    # Guards the division: a degenerate hand must not blow the geometry up.
    assert draw_ratio((300, 200), (100, 200), 0.0) == 0.0


def test_draw_ratio_clamps_at_a_full_draw():
    scale = 50.0
    beyond_full = (100 + int(MAX_DRAW * scale) + 200, 200)

    assert draw_ratio(beyond_full, (100, 200), scale) == 1.0


def test_draw_ratio_grows_with_the_draw_length():
    scale, nock = 50.0, (100, 200)
    short = draw_ratio((100 + int(MIN_DRAW * scale), 200), nock, scale)
    long = draw_ratio((100 + int(MAX_DRAW * scale * 0.75), 200), nock, scale)

    assert 0.0 < short < long < 1.0


def test_colours_carry_an_explicit_alpha():
    # OpenCV's colour argument is a four-component scalar. A 3-tuple against the
    # 4-channel overlay canvas would leave alpha at 0 and draw nothing at all,
    # while looking perfectly correct in the source.
    for colour in (BOW_COLOR, BOW_HIGHLIGHT, STRING_COLOR, SHAFT_COLOR,
                   HEAD_COLOR, FLETCHING_COLOR):
        assert len(colour) == 4
        assert colour[3] == 255


def test_speed_scale_multiplies_the_arrow_speed():
    """ARROW_MIN_SPEED/ARROW_MAX_SPEED are the only absolute-pixel quantities
    in the module, so a bow drawn at desktop size would fire arrows that crawl
    unless their speed is scaled alongside it."""
    nock, scale = (100, 200), 50.0
    grip = (100 + int(MAX_DRAW * scale), 200)

    plain, scaled = HorseBow(), HorseBow(speed_scale=3.0)
    plain.loose(grip, nock, scale)
    scaled.loose(grip, nock, scale)

    assert scaled.arrows[0].vx == pytest.approx(plain.arrows[0].vx * 3.0)


def test_speed_scale_defaults_to_leaving_the_speed_alone():
    nock, scale = (100, 200), 50.0
    grip = (100 + int(MAX_DRAW * scale), 200)

    default, explicit = HorseBow(), HorseBow(speed_scale=1.0)
    default.loose(grip, nock, scale)
    explicit.loose(grip, nock, scale)

    assert default.arrows[0].vx == explicit.arrows[0].vx


def test_drawing_on_a_four_channel_canvas_keeps_its_alpha():
    """The overlay canvas is BGRA. If a colour constant lost its fourth
    component the bow would draw perfectly and be completely invisible."""
    canvas = np.zeros((600, 600, 4), np.uint8)

    HorseBow().draw(
        canvas, (150 + int(MAX_DRAW * 40.0), 300), (150, 300), 40.0, arrow=True
    )

    assert canvas[:, :, 3].max() == 255
    opaque = canvas[:, :, 3] == 255
    assert opaque.sum() > 0


def test_nothing_is_drawn_behind_the_fletching():
    """The arrow used to trail a motion streak a full frame's travel behind the
    nock, which read as a tail hanging off the fletching. The feathers are now
    the hindmost ink there is, and `reach` still has to cover them — anything
    drawn outside it is left burned onto the desktop overlay."""
    bow = HorseBow()
    nock, scale = (100, 200), 50.0
    bow.loose((100 + int(MAX_DRAW * scale), 200), nock, scale)
    arrow = bow.arrows[0]
    # Loosed straight along +x, so distance behind the tip is a column offset.
    assert arrow.vy == 0 and arrow.vx > 0
    arrow.x, arrow.y = 400.0, 200.0

    canvas = np.zeros((400, 600, 4), np.uint8)
    arrow.draw(canvas)

    inked = np.nonzero(canvas.any(axis=2).any(axis=0))[0]
    behind = 400 - inked.min()
    assert behind <= arrow.reach
    # The shaft is `length` long and the feathers sweep a little past its end.
    # A streak would put ink a whole frame's travel — tens of pixels — further.
    assert behind < arrow.length * 1.05


def test_reach_covers_the_fletching_swept_behind_the_nock():
    """`reach` bounds what the overlay pushes, so it has to lead `length` by
    more than the stroke width alone: the feathers hang off the back."""
    arrow = Arrow((100, 200), (40.0, 0.0), 300.0)

    assert arrow.reach > arrow.length + max(2, int(arrow.length * 0.018))


def test_the_nocked_arrow_is_the_same_length_at_every_draw(monkeypatch):
    """The bug this replaced: the arrow on the string was the draw length plus
    an overhang, so it stretched as the string came back."""
    lengths = []
    monkeypatch.setattr(
        animations, "draw_arrow",
        lambda frame, tip, direction, length, thickness: lengths.append(length),
    )
    nock, scale = (100, 200), 50.0
    canvas = np.zeros((400, 600, 4), np.uint8)

    for draw in (MIN_DRAW, MAX_DRAW, ARROW_LENGTH - 0.1):
        grip = (100 + int(draw * scale), 200)
        HorseBow().draw(canvas, grip, nock, scale, arrow=True)

    assert lengths == [ARROW_LENGTH * scale] * 3


def test_the_arrow_falls_once_its_point_passes_the_grip():
    nock, scale = (100, 200), 50.0
    limit = 100 + ARROW_LENGTH * scale

    assert not is_overdrawn((limit - 1, 200), nock, scale)
    assert not is_overdrawn((limit, 200), nock, scale)
    assert is_overdrawn((limit + 1, 200), nock, scale)
    assert not is_overdrawn((limit + 1, 200), nock, 0.0)


def test_full_power_is_reachable_before_the_arrow_falls():
    """If the drop came at or before a full draw, holding a steady full draw
    would lose the arrow to tracking jitter alone."""
    assert MIN_DRAW < MAX_DRAW < ARROW_LENGTH


def test_draw_power_is_zero_at_min_draw_and_one_at_full():
    nock, scale = (100, 200), 50.0

    def power(draw):
        return draw_power((100 + draw * scale, 200), nock, scale)

    assert power(MIN_DRAW) == pytest.approx(0.0)
    assert power((MIN_DRAW + MAX_DRAW) / 2) == pytest.approx(0.5)
    assert power(MAX_DRAW) == pytest.approx(1.0)
    assert power(MIN_DRAW / 2) == 0.0
    assert power(ARROW_LENGTH) == 1.0
    assert draw_power((300, 200), nock, 0.0) == 0.0


def test_min_draw_leaves_at_min_speed_and_full_draw_at_max():
    """Power has to span the whole speed range. Keyed to the old draw ratio,
    the weakest shot that could be loosed already left at 58% of full speed,
    so a short draw never felt like a weak one."""
    nock, scale = (100, 200), 50.0
    bow = HorseBow()

    bow.loose((100 + MIN_DRAW * scale, 200), nock, scale)
    bow.loose((100 + MAX_DRAW * scale, 200), nock, scale)

    weak, full = bow.arrows
    assert weak.vx == pytest.approx(ARROW_MIN_SPEED)
    assert full.vx == pytest.approx(ARROW_MAX_SPEED)


def test_a_bow_without_an_arrow_draws_none(monkeypatch):
    drawn = []
    monkeypatch.setattr(animations, "draw_arrow", lambda *args: drawn.append(args))
    nock, scale = (100, 200), 50.0
    canvas = np.zeros((400, 600, 4), np.uint8)

    HorseBow().draw(canvas, (100 + int(MAX_DRAW * scale), 200), nock, scale, arrow=False)

    assert drawn == []
    assert canvas.any(), "the bow and its string are still drawn"


def test_a_dropped_arrow_starts_where_the_nocked_arrow_was_left(monkeypatch):
    """The same seam as the loose. The falling arrow's first position is the
    nocked arrow's, or it would jump off the string rather than slip off it."""
    nocked = []
    real_draw_arrow = animations.draw_arrow

    def spy(frame, tip, direction, length, thickness):
        nocked.append((tip, direction, length))
        real_draw_arrow(frame, tip, direction, length, thickness)

    monkeypatch.setattr(animations, "draw_arrow", spy)

    bow = HorseBow()
    nock, scale = (100, 200), 50.0
    grip = (100 + int(ARROW_LENGTH * scale) + 5, 190)

    bow.draw(np.zeros((400, 600, 4), np.uint8), grip, nock, scale, arrow=True)
    assert bow.drop(grip, nock, scale) is True

    ((tip, direction, length),) = nocked
    (fallen,) = bow.arrows
    assert isinstance(fallen, FallingArrow)
    assert (fallen.x, fallen.y) == pytest.approx(tip)
    assert fallen.direction == pytest.approx(direction)
    assert fallen.length == length


@pytest.mark.parametrize("aim_x", [1, -1])
def test_a_dropped_arrow_turns_nose_down_whichever_way_it_aimed(aim_x):
    bow = HorseBow()
    nock, scale = (1000, 300), 50.0
    grip = (1000 + aim_x * int(ARROW_LENGTH * scale + 5), 300)
    bow.drop(grip, nock, scale)
    (arrow,) = bow.arrows
    start_y = arrow.y

    downward = []
    for _ in range(8):
        arrow.update(2000, 2000)
        downward.append(arrow.direction[1])

    # The point tips further down every frame, still on the side it was aimed,
    # and the whole arrow falls.
    assert all(later > earlier > 0 for earlier, later in zip(downward, downward[1:]))
    assert arrow.direction[0] * aim_x > 0
    assert arrow.y > start_y
    assert arrow.alive


def test_a_drop_falls_identically_in_scales_on_both_surfaces():
    """Gravity is in scales, and the desktop pose's scale already carries
    the overlay's size factor. Multiplying by `speed_scale` as well would make
    the desktop arrow fall faster than the camera arrow it mirrors."""
    nock, grip, scale = (100, 200), (270, 180), 50.0
    camera, desktop = HorseBow(), HorseBow(speed_scale=3.0)
    camera.drop(grip, nock, scale)
    desktop.drop((grip[0] * 3, grip[1] * 3), (nock[0] * 3, nock[1] * 3), scale * 3)
    (small,), (big,) = camera.arrows, desktop.arrows
    small_start, big_start = (small.x, small.y), (big.x, big.y)

    for _ in range(6):
        small.update(10_000, 10_000)
        big.update(10_000, 10_000)

    assert big.x - big_start[0] == pytest.approx(3 * (small.x - small_start[0]))
    assert big.y - big_start[1] == pytest.approx(3 * (small.y - small_start[1]))
    assert big.direction == pytest.approx(small.direction)


def test_every_inked_pixel_of_a_tumbling_arrow_is_within_reach():
    """The overlay bounds a falling arrow by `reach` either side of its head,
    whichever way it has turned. Ink outside that is left burned onto the
    desktop."""
    for length in (60.0, 400.0):
        for step in range(16):
            angle = step * math.tau / 16
            direction = (math.cos(angle), math.sin(angle))
            arrow = FallingArrow((500.0, 500.0), direction, length, 0.0, 0.0)
            canvas = np.zeros((1000, 1000, 4), np.uint8)

            arrow.draw(canvas)

            rows, cols = np.nonzero(canvas[:, :, 3])
            assert np.abs(cols - 500).max() <= arrow.reach, (length, step)
            assert np.abs(rows - 500).max() <= arrow.reach, (length, step)


def test_the_bow_never_judges_the_draw_itself():
    """Whether a draw is long enough to shoot, and whether it is too long to
    still hold an arrow, are asked once by the caller for both bows. A bow that
    also judged either on its own rounded pose would, right at a limit, act on
    one surface and not the other."""
    nock, scale = (100, 200), 50.0
    too_short = (100 + int(MIN_DRAW * scale) - 1, 200)
    past_the_limit = (100 + int(ARROW_LENGTH * scale) + 20, 200)
    inside_the_limit = (100 + int(MAX_DRAW * scale), 200)
    assert not is_drawn(too_short, nock, scale)
    assert is_overdrawn(past_the_limit, nock, scale)
    assert not is_overdrawn(inside_the_limit, nock, scale)

    bow = HorseBow()

    assert bow.loose(too_short, nock, scale) is True
    assert bow.loose(past_the_limit, nock, scale) is True
    assert bow.drop(inside_the_limit, nock, scale) is True
    assert len(bow.arrows) == 3


def test_draw_will_not_guess_whether_there_is_an_arrow():
    """No default. A call site that forgot to say would draw an arrow the other
    bow, told correctly, does not have."""
    with pytest.raises(TypeError):
        HorseBow().draw(np.zeros((400, 600, 4), np.uint8), (230, 200), (100, 200), 50.0)
