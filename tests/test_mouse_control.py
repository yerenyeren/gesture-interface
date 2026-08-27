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
