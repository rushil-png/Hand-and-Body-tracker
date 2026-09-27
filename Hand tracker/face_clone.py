import cv2
import mediapipe as mp
import os

model_path = r"C:\Users\rushi\OneDrive\Code\Hand tracker\face_landmarker.task"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

# MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Face Landmarker options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=10
)

detector = FaceLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Timestamp
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Detect faces
    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    # If faces were detected
    if result.face_landmarks:

        h, w, _ = frame.shape

        # Loop through all detected faces
        for face in result.face_landmarks:

            points = []

            # Convert landmarks to OpenCV coordinates
            for lm in face:
                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))

            # Draw every landmark
            for p in points:
                cv2.circle(
                    frame,
                    p,
                    2,
                    (0, 255, 0),
                    -1
                )

    # Display
    cv2.imshow("Face Tracking", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
