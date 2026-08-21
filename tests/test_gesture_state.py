from gesture_state import EdgeDetector


def _feed(detector, signals):
    """Run a signal through the detector, returning (rose, fell) per frame."""
    edges = []
    for signal in signals:
        detector.update(signal)
        edges.append((detector.rose, detector.fell))
    return edges


def test_held_signal_rises_exactly_once():
    detector = EdgeDetector(min_frames=1)

    edges = _feed(detector, [True] * 4)

    assert edges == [(True, False), (False, False), (False, False), (False, False)]
    assert detector.is_on is True


def test_releasing_the_signal_falls_exactly_once():
    detector = EdgeDetector(min_frames=1)

    edges = _feed(detector, [True, True, False, False])

    assert edges == [(True, False), (False, False), (False, True), (False, False)]
    assert detector.is_on is False


def test_signal_must_hold_for_min_frames_before_it_rises():
    detector = EdgeDetector(min_frames=3)

    edges = _feed(detector, [True, True, True])

    assert edges == [(False, False), (False, False), (True, False)]


def test_blip_shorter_than_min_frames_is_swallowed():
    detector = EdgeDetector(min_frames=3)

    edges = _feed(detector, [True, True, False, True, True, False])

    assert all(edge == (False, False) for edge in edges)
    assert detector.is_on is False


def test_dropped_frame_does_not_release_a_held_signal():
    detector = EdgeDetector(min_frames=2)

    edges = _feed(detector, [True, True, False, True, True])

    assert edges == [(False, False), (True, False), (False, False), (False, False), (False, False)]
    assert detector.is_on is True


def test_a_detector_starts_off():
    detector = EdgeDetector()

    assert (detector.is_on, detector.rose, detector.fell) == (False, False, False)


def test_min_frames_is_never_below_one():
    detector = EdgeDetector(min_frames=0)

    detector.update(True)

    assert detector.rose is True
