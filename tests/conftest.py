import sys
from unittest.mock import MagicMock

# pyautogui opens a live X connection at import time (via its mouseinfo
# dependency), which fails in headless/sandboxed environments before any
# test can even patch it. Stub it out before mouse_control imports it.
sys.modules.setdefault("pyautogui", MagicMock())
