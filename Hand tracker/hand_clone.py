import argparse
import math
import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp

model_path = r"C:\Users\rushi\OneDrive\Code\Hand tracker\hand_landmarker.task"

# Landmark indices
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
FINGERS = {            # name: (pip joint, tip)
    "Index":  (6, 8),
    "Middle": (10, 12),
    "Ring":   (14, 16),
    "Pinky":  (18, 20),
}
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17),                                  # Palm base
]

# Resolutions you can switch between while running (keys 1-4, or + / -)
PRESETS = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]

# BGR colours per hand
COLOURS = {"Left": (255, 120, 0), "Right": (0, 140, 255)}


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(pts):
    """Return {'Thumb': bool, 'Index': bool, ...}.

    Uses distances rather than 'is the tip higher on screen', so it still works
    when the hand is tilted, sideways or upside down.
    """
    state = {}
    # Thumb: extended if its tip is farther from the base of the pinky than its middle joint is
    state["Thumb"] = dist(pts[THUMB_TIP], pts[PINKY_MCP]) > dist(pts[THUMB_IP], pts[PINKY_MCP]) * 1.1
    # Other fingers: extended if the tip is farther from the wrist than the middle joint is
    for name, (pip, tip) in FINGERS.items():
        state[name] = dist(pts[tip], pts[WRIST]) > dist(pts[pip], pts[WRIST]) * 1.05
    return state


def recognise_gesture(pts, up):
    """Turn finger states + a few distances into a gesture name."""
    hand_size = dist(pts[WRIST], pts[MIDDLE_MCP]) or 1.0
    t, i, m, r, p = up["Thumb"], up["Index"], up["Middle"], up["Ring"], up["Pinky"]
    count = sum(up.values())

    pinch = dist(pts[THUMB_TIP], pts[8]) / hand_size
    if pinch < 0.35 and m and r and p:
        return "OK"

    if count == 0:
        return "Fist"
    if count == 5:
        return "Open Palm"
    if t and not (i or m or r or p):
        # Thumb only: up or down depends on where the tip is relative to the wrist
        lift = (pts[WRIST][1] - pts[THUMB_TIP][1]) / hand_size
        if lift > 0.4:
            return "Thumbs Up"
        if lift < -0.4:
            return "Thumbs Down"
        return "Thumb Out"
    if i and not (t or m or r or p):
        return "Pointing"
    if i and m and not (r or p):
        return "Peace"
    if t and p and not (i or m or r):
        return "Call Me"
    if i and m and r and not (p or t):
        return "Three"
    if i and m and r and p and not t:
        return "Four"
    return f"{count} Fingers"


def open_camera(index, width, height):
    """Open the webcam at (roughly) the requested size.

    On Windows the default backend often fails or crashes when the size is changed on a
    running camera, so we use DirectShow there and re-open the camera for every size change.
    MJPG is requested because many webcams can only do 720p/1080p in that format.
    """
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def parse_args():
    parser = argparse.ArgumentParser(description="Hand tracker")
    parser.add_argument("--res", default="640x480",
                        help="starting camera resolution, e.g. 1280x720 (default 640x480)")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()
    try:
        w, h = (int(v) for v in args.res.lower().split("x"))
    except ValueError:
        parser.error("--res must look like 1280x720")
    return w, h, args.camera


def main():
    req_w, req_h, cam_index = parse_args()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    cap, cur_w, cur_h = open_camera(cam_index, req_w, req_h)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cam_index}")
    failed_reads = 0

    # Resizable window: you can also just drag its edges to change the display size
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

    start = time.time()
    last_ts = -1
    history = {"Left": deque(maxlen=5), "Right": deque(maxlen=5)}  # smooths gesture flicker
    prev_time = start

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None:
            # A camera that is switching size can drop a few frames; only give up if it keeps failing
            failed_reads += 1
            if failed_reads > 30:
                print("Camera stopped returning frames.")
                break
            cv2.waitKey(30)
            continue
        failed_reads = 0

        # Mirror the image like a selfie camera. MediaPipe's Left/Right labels
        # assume a mirrored image, so this also makes them correct.
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Scale lines and text with the image so they look the same at any resolution
        sc = max(0.5, h / 720)
        line_t = max(1, int(3 * sc))
        dot_r = max(2, int(5 * sc))
        font = 0.7 * sc
        font_t = max(1, int(2 * sc))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must strictly increase (webcam timestamps from OpenCV are unreliable)
        ts = int((time.time() - start) * 1000)
        ts = max(ts, last_ts + 1)
        last_ts = ts
        result = detector.detect_for_video(mp_image, ts)

        total = 0
        seen = set()

        for lms, handed in zip(result.hand_landmarks, result.handedness):
            side = handed[0].category_name          # "Left" or "Right"
            seen.add(side)
            colour = COLOURS.get(side, (0, 255, 0))
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

            up = fingers_up(pts)
            gesture_now = recognise_gesture(pts, up)
            history[side].append(gesture_now)
            gesture = Counter(history[side]).most_common(1)[0][0]
            count = sum(up.values())
            total += count

            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], colour, line_t)
            for p in pts:
                cv2.circle(frame, p, dot_r, (0, 255, 0), -1)

            # Label above the hand
            x_min = min(p[0] for p in pts)
            y_min = min(p[1] for p in pts)
            label = f"{side} hand | {count} | {gesture}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, font_t)
            top = max(y_min - int(15 * sc), th + int(10 * sc))
            cv2.rectangle(frame, (x_min - 5, top - th - 8), (x_min + tw + 5, top + 6), colour, -1)
            cv2.putText(frame, label, (x_min, top), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), font_t)

        # Forget gestures for hands that left the frame
        for side in history:
            if side not in seen:
                history[side].clear()

        now = time.time()
        fps = 1 / (now - prev_time) if now > prev_time else 0
        prev_time = now

        big = 1.1 * sc
        cv2.putText(frame, f"Total fingers: {total}", (15, int(40 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, big, (255, 255, 255), max(2, int(4 * sc)))
        cv2.putText(frame, f"Total fingers: {total}", (15, int(40 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, big, (0, 0, 0), max(1, int(2 * sc)))
        info = f"{w}x{h}  |  {fps:.0f} FPS  |  1-4 or +/- = resolution, ESC = quit"
        cv2.putText(frame, info, (15, h - int(15 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * sc, (255, 255, 255), font_t)

        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        new_res = None
        if ord("1") <= key <= ord("4"):
            new_res = PRESETS[key - ord("1")]
        elif key in (ord("+"), ord("=")):
            bigger = [r for r in PRESETS if r[0] > cur_w]
            new_res = bigger[0] if bigger else None
        elif key in (ord("-"), ord("_")):
            smaller = [r for r in PRESETS if r[0] < cur_w]
            new_res = smaller[-1] if smaller else None

        if new_res and new_res != (cur_w, cur_h):
            cap.release()
            cap, cur_w, cur_h = open_camera(cam_index, *new_res)
            if not cap.isOpened():
                print("Could not reopen the camera; going back to the previous size.")
                cap, cur_w, cur_h = open_camera(cam_index, req_w, req_h)
            elif (cur_w, cur_h) != new_res:
                print(f"Camera can't do {new_res[0]}x{new_res[1]}; using {cur_w}x{cur_h}")
            req_w, req_h = cur_w, cur_h
            failed_reads = 0
            history["Left"].clear()
            history["Right"].clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
