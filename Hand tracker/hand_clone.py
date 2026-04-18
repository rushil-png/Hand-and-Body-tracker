import cv2
import mediapipe as mp
import os

model_path = r"C:\Users\rushi\OneDrive\Code\Hand tracker\hand_landmarker.task"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

# Line Code
mp_connections = [
    (0,1), (1,2), (2,3), (3,4),        # Thumb
    (0,5), (5,6), (6,7), (7,8),        # Index
    (5,9), (9,10), (10,11), (11,12),   # Middle
    (9,13), (13,14), (14,15), (15,16), # Ring
    (13,17), (17,18), (18,19), (19,20),# Pinky
    (0,17)                             # Palm Base Connection
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=10   # <--- allow multiple hands
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        h, w, _ = frame.shape

        # Loop through ALL detected hands
        for hand in result.hand_landmarks:
            points = []
            for lm in hand:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append((x, y))

            # Draw lines
            for c in mp_connections:
                cv2.line(frame, points[c[0]], points[c[1]], (255, 0, 0), 3)

            # Draw dots
            for p in points:
                cv2.circle(frame, p, 5, (0, 255, 0), -1)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()