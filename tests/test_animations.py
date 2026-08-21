from animations import (
    Arrow,
    HorseBow,
    bow_profile,
    transform_points,
    BOW_HALF_LENGTH,
    MAX_DRAW,
    MIN_DRAW,
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
