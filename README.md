# Game-recommendation-website

Hand & Body Tracker
Real-time hand and body tracking using MediaPipe. Draws a chibi silhouette over your body and overlays hand landmarks on the webcam feed.

Files

hand_tracker.py — hand landmark overlay, tracks up to 10 hands
main.py — full body chibi silhouette with lavender gradient + purple edge glow

Setup
bashpip install mediapipe opencv-python numpy
Download the three .task model files and drop them in the project folder:

hand_landmarker.task
pose_landmarker.task
face_landmarker.task

Update the paths at the top of each script if needed.
Run
bashpython hand_tracker.py   # hand tracking only
python main.py           # full body silhouette
Press ESC to quit.
