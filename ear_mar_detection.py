import cv2
import mediapipe as mp
import math
import time
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

MAR_THRESHOLD = 0.6
SLEEP_TIME = 2.0
YAWN_TIME = 1.0
DISTRACTION_TIME = 2.0

CALIBRATION_TIME = 3.0
EAR_RATIO = 0.8

YAW_THRESHOLD = 20
PITCH_THRESHOLD = 20

ear_threshold = None
ear_values = []
calibration_start_time = time.time()

sleep_start_time = None
yawn_start_time = None
distraction_start_time = None

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def calculate_ear(landmarks):
    p1 = landmarks[33]
    p2 = landmarks[160]
    p3 = landmarks[158]
    p4 = landmarks[133]
    p5 = landmarks[153]
    p6 = landmarks[144]

    return (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))

def calculate_mar(landmarks):
    left = landmarks[61]
    right = landmarks[291]
    upper = landmarks[13]
    lower = landmarks[14]

    return distance(upper, lower) / distance(left, right)

def calculate_head_pose(landmarks, frame_width, frame_height):
    # 2D image points from MediaPipe landmarks
    image_points = np.array([
        (landmarks[1].x * frame_width, landmarks[1].y * frame_height),       # nose tip
        (landmarks[152].x * frame_width, landmarks[152].y * frame_height),   # chin
        (landmarks[33].x * frame_width, landmarks[33].y * frame_height),     # left eye corner
        (landmarks[263].x * frame_width, landmarks[263].y * frame_height),   # right eye corner
        (landmarks[61].x * frame_width, landmarks[61].y * frame_height),     # left mouth corner
        (landmarks[291].x * frame_width, landmarks[291].y * frame_height)    # right mouth corner
    ], dtype="double")

    # Approximate 3D face model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # nose tip
        (0.0, -63.6, -12.5),         # chin
        (-43.3, 32.7, -26.0),        # left eye corner
        (43.3, 32.7, -26.0),         # right eye corner
        (-28.9, -28.9, -24.1),       # left mouth corner
        (28.9, -28.9, -24.1)         # right mouth corner
    ])

    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll

def judge_state(ear, mar, yaw, pitch):
    global sleep_start_time, yawn_start_time, distraction_start_time

    current_time = time.time()
    state = "normal"

    if ear < ear_threshold:
        if sleep_start_time is None:
            sleep_start_time = current_time

        if current_time - sleep_start_time >= SLEEP_TIME:
            state = "sleep"
    else:
        sleep_start_time = None

    if mar > MAR_THRESHOLD:
        if yawn_start_time is None:
            yawn_start_time = current_time

        if current_time - yawn_start_time >= YAWN_TIME:
            state = "yawn"
    else:
        yawn_start_time = None

    if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
        if distraction_start_time is None:
            distraction_start_time = current_time

        if current_time - distraction_start_time >= DISTRACTION_TIME:
            state = "distraction"
    else:
        distraction_start_time = None

    return state

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라를 열 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        ear = calculate_ear(landmarks)
        mar = calculate_mar(landmarks)
        pitch, yaw, roll = calculate_head_pose(landmarks, width, height)

        if ear_threshold is None:
            elapsed_time = time.time() - calibration_start_time
            ear_values.append(ear)

            cv2.putText(frame, "Calibrating... Look straight normally", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"Time: {elapsed_time:.1f}/{CALIBRATION_TIME}s", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if elapsed_time >= CALIBRATION_TIME:
                avg_ear = np.mean(ear_values)
                ear_threshold = avg_ear * EAR_RATIO
                print(f"Calibration complete. Average EAR: {avg_ear:.3f}, EAR_THRESHOLD: {ear_threshold:.3f}")

        else:
            state = judge_state(ear, mar, yaw, pitch)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"EAR_TH: {ear_threshold:.2f}", (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"YAW: {yaw:.1f}", (30, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.putText(frame, f"PITCH: {pitch:.1f}", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.putText(frame, f"ROLL: {roll:.1f}", (30, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.putText(frame, f"STATE: {state}", (30, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("EAR MAR Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()