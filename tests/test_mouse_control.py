from unittest.mock import patch

from mouse_control import MouseController


@patch("mouse_control.pyautogui")
def test_move_to_scales_frame_coords_to_screen_coords(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(320, 240)

    mock_pyautogui.moveTo.assert_called_once_with(960, 540)


@patch("mouse_control.pyautogui")
def test_move_to_keeps_edges_just_inside_the_screen(mock_pyautogui):
    # Scaling lands exactly on 0 and screen_height here, so both edges get
    # pulled inside — on-screen, and off the top-left hot corner.
    mock_pyautogui.size.return_value = (2000, 1000)
    mouse = MouseController(frame_width=1000, frame_height=500)

    mouse.move_to(0, 500)

    mock_pyautogui.moveTo.assert_called_once_with(1, 998)


@patch("mouse_control.pyautogui")
def test_a_press_and_release_is_an_ordinary_click(mock_pyautogui):
    """A left click is a held press rather than a complete click, so that a
    pinch held while the hand moves drags — which is how text is highlighted.
    A quick pinch still lands as a click, press and release a frame apart."""
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.press()
    mouse.release()

    mock_pyautogui.mouseDown.assert_called_once_with()
    mock_pyautogui.mouseUp.assert_called_once_with()


@patch("mouse_control.pyautogui")
def test_pressing_twice_holds_the_button_once(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.press()
    mouse.press()

    assert mouse.is_pressed is True
    mock_pyautogui.mouseDown.assert_called_once_with()


@patch("mouse_control.pyautogui")
def test_release_is_safe_to_call_without_a_press(mock_pyautogui):
    """Called on every path out of the dragging state, most of which never
    pressed anything. A button left down is worse than a lost click: it selects
    everything the cursor touches until the app is killed."""
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.release()
    mouse.release()

    assert mouse.is_pressed is False
    mock_pyautogui.mouseUp.assert_not_called()


@patch("mouse_control.pyautogui")
def test_a_released_button_can_be_pressed_again(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    for _ in range(3):
        mouse.press()
        mouse.release()

    assert mock_pyautogui.mouseDown.call_count == 3
    assert mock_pyautogui.mouseUp.call_count == 3


@patch("mouse_control.pyautogui")
def test_right_click_calls_pyautogui_right_click(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.right_click()

    mock_pyautogui.rightClick.assert_called_once_with()
    mock_pyautogui.mouseDown.assert_not_called()


@patch("mouse_control.pyautogui")
def test_scroll_passes_clicks_straight_through(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.scroll(-4)

    mock_pyautogui.scroll.assert_called_once_with(-4)


@patch("mouse_control.pyautogui")
def test_second_move_is_pulled_back_towards_the_previous_position(mock_pyautogui):
    # Landmark jitter shows up as a shaking cursor, so a slow move is damped
    # rather than passed straight through.
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(100, 100)  # snaps to (300, 225)
    mouse.move_to(110, 100)  # would be (330, 225) unsmoothed

    smoothed_x = mock_pyautogui.moveTo.call_args.args[0]
    assert 300 < smoothed_x < 330


@patch("mouse_control.pyautogui")
def test_a_fast_move_is_not_smoothed_at_all(mock_pyautogui):
    # Damping every movement equally would trade jitter for exactly the lag the
    # smoothing exists to avoid, so past a threshold it stops applying.
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(100, 100)
    mouse.move_to(200, 100)

    mock_pyautogui.moveTo.assert_called_with(600, 225)


@patch("mouse_control.pyautogui")
def test_reset_makes_the_next_move_snap_to_its_target(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(100, 100)
    mouse.reset()
    mouse.move_to(110, 100)

    mock_pyautogui.moveTo.assert_called_with(330, 225)


@patch("mouse_control.pyautogui")
def test_a_stationary_hand_issues_no_further_calls(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(100, 100)
    for _ in range(5):
        mouse.move_to(100, 100)

    assert mock_pyautogui.moveTo.call_count == 1


@patch("mouse_control.pyautogui")
def test_a_slow_drift_still_eventually_moves_the_cursor(mock_pyautogui):
    # The deadzone is measured against the last position actually sent, so
    # sub-threshold movement accumulates instead of being discarded every frame.
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(100, 100)
    for _ in range(20):
        mouse.move_to(101, 100)

    assert mock_pyautogui.moveTo.call_count > 1


@patch("mouse_control.pyautogui")
def test_a_press_names_the_point_the_app_last_sent(mock_pyautogui):
    """The regression test for the drag-from-a-stale-anchor bug. Called with no
    coordinates, pyautogui fills them in from `query_pointer`, which under
    XWayland reports a cached pointer position that goes stale the moment the
    cursor crosses a native Wayland window — and warps the real pointer there
    before pressing. Every click then anchored at the previous one."""
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(320, 240)
    mouse.press()

    mock_pyautogui.mouseDown.assert_called_once_with(960, 540)


@patch("mouse_control.pyautogui")
def test_a_release_names_the_point_the_app_last_sent(mock_pyautogui):
    # A release warped back to a stale pointer ends the drag somewhere the hand
    # never was, so it takes coordinates for the same reason the press does.
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(320, 240)
    mouse.press()
    mouse.release()

    mock_pyautogui.mouseUp.assert_called_once_with(960, 540)


@patch("mouse_control.pyautogui")
def test_a_right_click_names_the_point_the_app_last_sent(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(320, 240)
    mouse.right_click()

    mock_pyautogui.rightClick.assert_called_once_with(960, 540)


@patch("mouse_control.pyautogui")
def test_a_button_before_any_move_falls_back_to_no_coordinates(mock_pyautogui):
    """Only when nothing has been sent yet: with no move behind it the app has
    no opinion about where the cursor is, and pyautogui's guess is the best
    available answer."""
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.press()
    mouse.release()
    mouse.right_click()

    mock_pyautogui.mouseDown.assert_called_once_with()
    mock_pyautogui.mouseUp.assert_called_once_with()
    mock_pyautogui.rightClick.assert_called_once_with()


@patch("mouse_control.pyautogui")
def test_a_button_after_a_reset_falls_back_to_no_coordinates(mock_pyautogui):
    # reset() clears the record along with the smoothing, so the app is back to
    # having no opinion — the same state as before the first move.
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(320, 240)
    mouse.reset()
    mouse.press()

    mock_pyautogui.mouseDown.assert_called_once_with()


@patch("mouse_control.pyautogui")
def test_snap_to_sends_a_move_the_deadzone_would_have_swallowed(mock_pyautogui):
    """A click has to land on its point however small the correction is. The
    deadzone exists to stop jitter generating traffic, not to drop the one move
    that decides where the button goes down."""
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(100, 100)  # snaps to (300, 225)
    mouse.snap_to(100.2, 100)  # (300, 225) again — under CURSOR_DEADZONE_PX

    assert mock_pyautogui.moveTo.call_count == 2
    mock_pyautogui.moveTo.assert_called_with(300, 225)


@patch("mouse_control.pyautogui")
def test_snap_to_lands_on_the_target_rather_than_easing_towards_it(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(100, 100)
    mouse.snap_to(200, 100)

    mock_pyautogui.moveTo.assert_called_with(600, 225)
    assert mouse.position == (600, 225)


@patch("mouse_control.pyautogui")
def test_a_move_after_a_snap_to_the_same_point_adds_no_movement(mock_pyautogui):
    """The whole point of the snap: the smoothing is left *on* the target, not
    behind it, so the next frame has no residual lag to ease off — which with
    the button held would be a drag on the end of the click."""
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(100, 100)
    mouse.snap_to(200, 100)
    mouse.move_to(200, 100)

    assert mock_pyautogui.moveTo.call_count == 2  # the third move is negligible
    assert mouse.position == (600, 225)


@patch("mouse_control.pyautogui")
def test_snap_to_keeps_edges_just_inside_the_screen(mock_pyautogui):
    mock_pyautogui.size.return_value = (2000, 1000)
    mouse = MouseController(frame_width=1000, frame_height=500)

    mouse.snap_to(0, 500)

    mock_pyautogui.moveTo.assert_called_once_with(1, 998)


@patch("mouse_control.pyautogui")
def test_pressing_twice_after_a_move_still_holds_the_button_once(mock_pyautogui):
    # Idempotency is what keeps a held pinch a single drag; passing coordinates
    # must not turn each frame of it into a fresh press.
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(320, 240)
    mouse.press()
    mouse.press()
    mouse.release()
    mouse.release()

    assert mouse.is_pressed is False
    mock_pyautogui.mouseDown.assert_called_once_with(960, 540)
    mock_pyautogui.mouseUp.assert_called_once_with(960, 540)
