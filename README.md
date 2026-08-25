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

Keys, in the camera window:

| Key | Does |
| --- | --- |
| `q` | Quit |
| `d` | FPS and per-stage timing readout |
| `s` | Hand skeleton overlay (hidden automatically while the bow is drawn) |
| `t` | Tuning readout — the raw numbers each gesture is being decided on |
| `o` | Desktop overlay on/off |
| `p` | Pause — park the cursor without leaving the frame |

Pause is a key and not a gesture. It was a gesture, and a pause that fires
when it was not asked for takes the cursor away mid-task — the most expensive
false positive the app has, and not worth the convenience.

The `t` readout exists because a gesture that fails to fire looks identical to
one that was never made. It shows each finger's extended/curled ratio against
the thresholds, both pinch distances against the threshold for that hand, and
the draw length while the bow is drawn — so a threshold can be moved against a
reading rather than a guess.

## Gestures

The mode currently in effect is shown in the top-left of the camera window.

| Gesture | Action |
| --- | --- |
| Hand in view | Cursor follows your palm |
| Thumb + index pinch | Left click |
| Thumb + middle pinch | Right click |
| Index + middle up, ring and pinky down | Scroll — move the hand up or down |
| Fist + OK sign, two hands | Draw the bow |

### The desktop overlay

The bow is drawn twice: once into the camera window, and once onto a
transparent, click-through window covering the desktop, so it is drawn over
whatever you are actually looking at. Clicks pass straight through it.

This session is GNOME on Wayland, where a client cannot ask to be always on top
or click-through at all — Mutter implements no layer-shell protocol, and the
only surface it offers a client is an ordinary toplevel. The overlay therefore
goes through XWayland: an override-redirect ARGB window with an empty `SHAPE`
input region, which is the one combination that stacks above everything and
swallows no input.

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
- `desktop_overlay.py` — the transparent click-through window the bow is drawn on
- `tests/` — unit tests

## Roadmap

- [x] Hand tracking working, landmarks drawn on camera feed
- [x] Cursor follows hand position
- [x] Pinch gesture triggers a click
- [x] Special gesture triggers an animation overlay
- [x] More gestures (right click, scroll)
- [x] Draw the animation over the desktop itself, not just the camera window
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
