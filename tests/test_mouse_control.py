from unittest.mock import patch

from mouse_control import MouseController


@patch("mouse_control.pyautogui")
def test_move_to_scales_frame_coords_to_screen_coords(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.move_to(320, 240)

    mock_pyautogui.moveTo.assert_called_once_with(960, 540)


@patch("mouse_control.pyautogui")
def test_move_to_keeps_edges_clear_of_the_failsafe_corners(mock_pyautogui):
    # pyautogui.FAILSAFE raises in a screen corner, and scaling lands exactly on
    # 0 and screen_height here, so both edges have to be pulled inside.
    mock_pyautogui.size.return_value = (2000, 1000)
    mouse = MouseController(frame_width=1000, frame_height=500)

    mouse.move_to(0, 500)

    mock_pyautogui.moveTo.assert_called_once_with(1, 998)


@patch("mouse_control.pyautogui")
def test_click_calls_pyautogui_click(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.click()

    mock_pyautogui.click.assert_called_once_with()


@patch("mouse_control.pyautogui")
def test_right_click_calls_pyautogui_right_click(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.right_click()

    mock_pyautogui.rightClick.assert_called_once_with()
    mock_pyautogui.click.assert_not_called()


@patch("mouse_control.pyautogui")
def test_scroll_passes_clicks_straight_through(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    mouse = MouseController(frame_width=640, frame_height=480)

    mouse.scroll(-4)

    mock_pyautogui.scroll.assert_called_once_with(-4)
