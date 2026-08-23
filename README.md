# Gesture Interface

A webcam-based gesture control tool for the desktop: track your hands with
MediaPipe, drive the mouse cursor with your hand, and draw
bow across the camera feed when you strike the pose.

## Setup
One-time setup:
```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
mkdir -p assets
curl -fsSL -o assets/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

To run the app:

```bash
venv/bin/python main.py
```

To run the unit tests:

```bash
venv/bin/python -m pytest tests/ -v
```

Press `q` in the camera window to quit, `d` to toggle the FPS and per-stage
timing readout, or `s` to toggle the hand skeleton — which hides itself while
the bow is drawn either way.

## Gestures

The mode currently in effect is shown in the top-left of the camera window.

| Gesture | Action |
| --- | --- |
| Hand in view | Cursor follows your palm |
| Thumb + index pinch | Left click |
| Thumb + middle pinch | Right click |
| Index + middle up, ring and pinky down | Scroll — move the hand up or down |
| Flat open palm | Toggle pause, so you can rest or reposition your hand |
| Fist + OK sign, two hands | Draw the bow |

### The bow

The archery pose is the steppe **thumb draw**: a closed fist grips the bow while
the other hand hooks the string with the thumb, index folded over the thumbnail —
which is what makes it look like an OK sign to a camera. Move the hands apart to
draw; the limbs flex as the draw lengthens. Open the string hand to loose, and
the arrow flies off across the frame at a speed set by how far you drew.

Needing both hands at once is deliberate: it means a one-handed click pinch can
never loose an arrow by accident.

## Project layout

- `main.py` — camera loop, wires the pieces together and picks the active mode
- `hand_tracker.py` — wraps MediaPipe hand landmark detection
- `gestures.py` — turns raw landmarks into named gestures (pinch, fist, ...)
- `gesture_state.py` — debounced edge detection, so actions fire once per gesture
- `mouse_control.py` — maps hand position/gestures to mouse move, click and scroll
- `animations.py` — the procedurally drawn bow and its arrows
- `tests/` — unit tests

## Roadmap

- [x] Hand tracking working, landmarks drawn on camera feed
- [x] Cursor follows hand position
- [x] Pinch gesture triggers a click
- [x] Special gesture triggers an animation overlay
- [x] More gestures (right click, scroll, pause)
- [ ] Draw the animation over the desktop itself, not just the camera window
- [ ] More gestures (volume, media control, ...)

## How this was built

I built this with [Claude Code](https://claude.com/claude-code). I set the
direction, chose the approach and did the debugging against the real hardware;
much of the code itself was written with AI assistance.

I'm using it to learn this domain rather than to skip it — hand landmark
geometry, gesture debouncing, and the surprising amount of webcam and
display-server behaviour underneath both. A fair share of the work here was
things no model could have told me from the outside: that the frame rate
collapsed only while a hand was tracked, that the camera was returning black
frames, that a fail-safe meant to stop runaway scripts kills an app whose whole
purpose is moving the cursor. My plan is to build the next tool like this one
without the assistance, once I understand the pieces well enough to.
