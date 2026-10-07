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
    if i and p and not (m or r):
        return "Rock On"
    if t and p and not (i or m or r):
        return "Call Me"
    if i and m and r and not (p or t):
        return "Three"
    if i and m and r and p and not t:
        return "Four"
    return f"{count} Fingers"


def main():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    start = time.time()
    last_ts = -1
    history = {"Left": deque(maxlen=5), "Right": deque(maxlen=5)}  # smooths gesture flicker
    prev_time = start

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror the image like a selfie camera. MediaPipe's Left/Right labels
        # assume a mirrored image, so this also makes them correct.
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

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
                cv2.line(frame, pts[a], pts[b], colour, 3)
            for name_idx, p in enumerate(pts):
                cv2.circle(frame, p, 5, (0, 255, 0), -1)

            # Label above the hand
            x_min = min(p[0] for p in pts)
            y_min = min(p[1] for p in pts)
            label = f"{side} hand | {count} | {gesture}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            top = max(y_min - 15, th + 10)
            cv2.rectangle(frame, (x_min - 5, top - th - 8), (x_min + tw + 5, top + 6), colour, -1)
            cv2.putText(frame, label, (x_min, top), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Forget gestures for hands that left the frame
        for side in history:
            if side not in seen:
                history[side].clear()

        now = time.time()
        fps = 1 / (now - prev_time) if now > prev_time else 0
        prev_time = now
        cv2.putText(frame, f"Total fingers: {total}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4)
        cv2.putText(frame, f"Total fingers: {total}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
