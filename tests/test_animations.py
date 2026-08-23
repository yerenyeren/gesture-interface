import numpy as np
import pytest

from animations import (
    Arrow,
    HorseBow,
    bow_profile,
    draw_ratio,
    transform_points,
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


def test_loose_launches_an_arrow_toward_the_grip():
    bow = HorseBow()
    # Grip to the right of the nock, so the arrow should fly right.
    nock, grip, scale = (100, 200), (300, 200), 50.0

    assert bow.loose(grip, nock, scale) is True
    assert len(bow.arrows) == 1

    arrow = bow.arrows[0]
    assert (arrow.x, arrow.y) == (100.0, 200.0)
    assert arrow.vx > 0
    assert arrow.vy == 0


def test_loose_ignores_a_bow_that_was_never_drawn():
    bow = HorseBow()
    scale = 50.0
    barely_drawn = (100 + int(MIN_DRAW * scale) - 1, 200)

    assert bow.loose(barely_drawn, (100, 200), scale) is False
    assert bow.arrows == []


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
    assert bow.draw(None, (300, 200), (100, 200), 0.0) is None


def test_bow_half_length_is_expressed_in_hand_scales():
    # Sizes are multiples of hand_scale so the bow tracks the hand's distance
    # from the camera rather than sitting at a fixed pixel size.
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

    HorseBow().draw(canvas, (300, 300), (150, 300), 40.0)

    assert canvas[:, :, 3].max() == 255
    opaque = canvas[:, :, 3] == 255
    assert opaque.sum() > 0


def test_the_streak_is_the_distance_covered_in_one_frame():
    """A motion smear is how far the thing moved, so the streak has to follow
    speed. Tying it to the shaft length instead made it as long as the whole
    arrow, which at desktop scale read as a tail hanging off the fletching."""
    bow = HorseBow()
    nock, scale = (100, 200), 50.0
    bow.loose((100 + int(MAX_DRAW * scale), 200), nock, scale)

    arrow = bow.arrows[0]
    assert arrow.speed == pytest.approx(abs(arrow.vx))
    assert arrow.reach > arrow.length + arrow.speed  # plus the stroke width
    # And the streak is a modest fraction of the arrow, not a second arrow.
    assert arrow.speed < arrow.length * 0.6


def test_reach_grows_with_speed_scale():
    nock, scale = (100, 200), 50.0
    grip = (100 + int(MAX_DRAW * scale), 200)
    plain, scaled = HorseBow(), HorseBow(speed_scale=3.0)
    plain.loose(grip, nock, scale)
    scaled.loose(grip, nock, scale)

    assert scaled.arrows[0].reach > plain.arrows[0].reach
