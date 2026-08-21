import pyautogui

pyautogui.FAILSAFE = True


class MouseController:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.screen_width, self.screen_height = pyautogui.size()

    def move_to(self, x, y):
        screen_x = int(x / self.frame_width * self.screen_width)
        screen_y = int(y / self.frame_height * self.screen_height)
        pyautogui.moveTo(*self._clamp(screen_x, screen_y))

    def _clamp(self, x, y):
        # FAILSAFE raises as soon as the cursor lands in a screen corner, and the
        # scaling above reaches the exact edges, so keep a pixel of margin.
        return (
            min(max(x, 1), self.screen_width - 2),
            min(max(y, 1), self.screen_height - 2),
        )

    def click(self):
        pyautogui.click()

    def right_click(self):
        pyautogui.rightClick()

    def scroll(self, clicks):
        pyautogui.scroll(clicks)
