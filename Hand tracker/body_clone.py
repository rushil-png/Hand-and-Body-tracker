import cv2
import mediapipe as mp
import os
import numpy as np
import time

hand_model_path = r"C:\Users\rushi\OneDrive\Code\Hand tracker\hand_landmarker.task"
pose_model_path = r"C:\Users\rushi\OneDrive\Code\Hand tracker\pose_landmarker.task"
face_model_path = r"C:\Users\rushi\OneDrive\Code\Hand tracker\face_landmarker.task"

for path in [hand_model_path, pose_model_path, face_model_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

# =========================================================
#  COLOURS
# =========================================================
LAVENDER_CENTER = np.array([207, 170, 255], dtype=np.float32)
LAVENDER_EDGE   = np.array([150, 120, 200], dtype=np.float32)

PURPLE_EDGE_COLOR = (180, 100, 255)
OUTLINE_COLOR = (0, 0, 0)

PURPLE_EDGE_THICKNESS = 12
OUTLINE_THICKNESS = 7

# =========================================================
#  SHAPES AND LIMB CON
# =========================================================
def draw_capsule(canvas, p1, p2, radius, color):
    p1 = np.array(p1, dtype=np.int32)
    p2 = np.array(p2, dtype=np.int32)

    cv2.circle(canvas, tuple(p1), radius, color, -1)
    cv2.circle(canvas, tuple(p2), radius, color, -1)

    v = p2 - p1
    norm = np.linalg.norm(v)
    if norm < 1:
        return

    v = v / norm
    perp = np.array([-v[1], v[0]]) * radius

    pts = np.array([p1 + perp, p1 - perp, p2 - perp, p2 + perp], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color)

def apply_radial_gradient(mask, torso_center):
    h, w = mask.shape[:2]
    y_idx, x_idx = np.indices((h, w))
    dist = np.sqrt((x_idx - torso_center[0])**2 + (y_idx - torso_center[1])**2)

    max_dist = np.max(dist) if np.max(dist) > 0 else 1
    t = np.clip((dist / max_dist) * 1.2, 0, 1)

    gradient = (LAVENDER_CENTER * (1 - t)[:, :, None] +
                LAVENDER_EDGE   * t[:, :, None]).astype(np.uint8)

    colored = np.zeros_like(gradient)
    colored[mask > 0] = gradient[mask > 0]
    return colored

def add_purple_edge(mask):
    dilated = cv2.dilate(mask, np.ones((PURPLE_EDGE_THICKNESS, PURPLE_EDGE_THICKNESS), np.uint8))
    edge = (dilated > 0) & (mask == 0)
    return (edge.astype(np.uint8) * 255)

def add_outline(mask):
    dilated = cv2.dilate(mask, np.ones((OUTLINE_THICKNESS, OUTLINE_THICKNESS), np.uint8))
    outline = (dilated > 0) & (mask == 0)
    return (outline.astype(np.uint8) * 255)

# =========================================================
# STICKMAN RENDER
# =========================================================
def render_chibi_silhouette(frame, pose_landmarks, w, h):
    if pose_landmarks is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]

    required = [11, 12, 23, 24, 13, 14, 15, 16, 25, 26, 27, 28]
    if any(i >= len(pts) for i in required):
        return np.zeros((h, w, 3), dtype=np.uint8)

    L_SH, R_SH = pts[11], pts[12]
    L_HP, R_HP = pts[23], pts[24]
    L_EL, R_EL = pts[13], pts[14]
    L_WR, R_WR = pts[15], pts[16]
    L_KN, R_KN = pts[25], pts[26]
    L_AN, R_AN = pts[27], pts[28]

    torso_center = (
        int((L_SH[0] + R_SH[0] + L_HP[0] + R_HP[0]) / 4),
        int((L_SH[1] + R_SH[1] + L_HP[1] + R_HP[1]) / 4)
    )

    # -----------------------------
    # PER-LIMB MASKS
    # -----------------------------
    head_mask = np.zeros((h, w), dtype=np.uint8)
    torso_mask = np.zeros((h, w), dtype=np.uint8)
    left_arm_mask = np.zeros((h, w), dtype=np.uint8)
    right_arm_mask = np.zeros((h, w), dtype=np.uint8)
    left_leg_mask = np.zeros((h, w), dtype=np.uint8)
    right_leg_mask = np.zeros((h, w), dtype=np.uint8)

    # Head
    head_center = ((L_SH[0] + R_SH[0]) // 2, (L_SH[1] + R_SH[1]) // 2 - 80)
    cv2.circle(head_mask, head_center, 70, 255, -1)

    # Torso
    torso_top = ((L_SH[0] + R_SH[0]) // 2, (L_SH[1] + R_SH[1]) // 2)
    torso_bottom = ((L_HP[0] + R_HP[0]) // 2, (L_HP[1] + R_HP[1]) // 2)
    draw_capsule(torso_mask, torso_top, torso_bottom, 40, 255)

    # Arms
    limb_r = 28
    draw_capsule(left_arm_mask, L_SH, L_EL, limb_r, 255)
    draw_capsule(left_arm_mask, L_EL, L_WR, limb_r, 255)
    cv2.circle(left_arm_mask, L_WR, 22, 255, -1)

    draw_capsule(right_arm_mask, R_SH, R_EL, limb_r, 255)
    draw_capsule(right_arm_mask, R_EL, R_WR, limb_r, 255)
    cv2.circle(right_arm_mask, R_WR, 22, 255, -1)

    # Legs
    draw_capsule(left_leg_mask, L_HP, L_KN, limb_r, 255)
    draw_capsule(left_leg_mask, L_KN, L_AN, limb_r, 255)
    cv2.circle(left_leg_mask, L_AN, 22, 255, -1)

    draw_capsule(right_leg_mask, R_HP, R_KN, limb_r, 255)
    draw_capsule(right_leg_mask, R_KN, R_AN, limb_r, 255)
    cv2.circle(right_leg_mask, R_AN, 22, 255, -1)

    # -----------------------------
    # PURPLE EDGE PER LIMB
    # -----------------------------
    purple_edges = np.zeros((h, w), dtype=np.uint8)

    for limb_mask in [
        head_mask, torso_mask,
        left_arm_mask, right_arm_mask,
        left_leg_mask, right_leg_mask
    ]:
        edge = add_purple_edge(limb_mask)
        purple_edges = np.maximum(purple_edges, edge)

    # -----------------------------
    # MERGE MASKS
    # -----------------------------
    full_mask = (
        head_mask | torso_mask |
        left_arm_mask | right_arm_mask |
        left_leg_mask | right_leg_mask
    )

    colored = apply_radial_gradient(full_mask, torso_center)

    # -----------------------------
    # APPLY PURPLE EDGE 
    # -----------------------------
    edge_mask = (purple_edges > 0)
    colored[edge_mask, 0] = PURPLE_EDGE_COLOR[0]
    colored[edge_mask, 1] = PURPLE_EDGE_COLOR[1]
    colored[edge_mask, 2] = PURPLE_EDGE_COLOR[2]

    # -----------------------------
    # APPLY BLACK OUTLINE
    # -----------------------------
    outline = add_outline(full_mask)
    outline_mask = (outline > 0)
    colored[outline_mask, 0] = OUTLINE_COLOR[0]
    colored[outline_mask, 1] = OUTLINE_COLOR[1]
    colored[outline_mask, 2] = OUTLINE_COLOR[2]

    return colored

# =========================================================
#  MEDIAPIPE SETUP
# =========================================================
mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
VisionRunningMode = mp_tasks.vision.RunningMode

HandLandmarker = mp_tasks.vision.HandLandmarker
PoseLandmarker = mp_tasks.vision.PoseLandmarker
FaceLandmarker = mp_tasks.vision.FaceLandmarker

hand_detector = HandLandmarker.create_from_options(
    mp_tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2
    )
)

pose_detector = PoseLandmarker.create_from_options(
    mp_tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
)

face_detector = FaceLandmarker.create_from_options(
    mp_tasks.vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )
)

# =========================================================
#  CAMERA LOOP
# =========================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Virtual Skeleton", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Skeleton", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = int((time.time() - prev_time) * 1000)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result = pose_detector.detect_for_video(mp_image, timestamp)

    h, w, _ = frame.shape

    # --------- CHANGED BLOCK: support multiple people ----------
    silhouette = np.zeros((h, w, 3), dtype=np.uint8)

    if pose_result.pose_landmarks:
        for person_landmarks in pose_result.pose_landmarks:
            person_sil = render_chibi_silhouette(frame, person_landmarks, w, h)
            silhouette = np.maximum(silhouette, person_sil)
    # -----------------------------------------------------------

    cv2.imshow("Virtual Skeleton", silhouette)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()