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

YAW_THRESHOLD = 15
PITCH_THRESHOLD = 15

ear_threshold = None
ear_values = []
yaw_values = []
pitch_values = []

yaw_baseline = None
pitch_baseline = None

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
    image_points = np.array([
        (landmarks[1].x * frame_width, landmarks[1].y * frame_height),
        (landmarks[152].x * frame_width, landmarks[152].y * frame_height),
        (landmarks[33].x * frame_width, landmarks[33].y * frame_height),
        (landmarks[263].x * frame_width, landmarks[263].y * frame_height),
        (landmarks[61].x * frame_width, landmarks[61].y * frame_height),
        (landmarks[291].x * frame_width, landmarks[291].y * frame_height)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
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

    if not success:
        return 0, 0, 0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = angles[0]
    yaw = -angles[1]
    roll = angles[2]

    # pitch가 ±180도 근처로 튀는 경우를 0도 기준으로 보정
    if pitch < -90:
        pitch += 180
    elif pitch > 90:
        pitch -= 180

    return pitch, yaw, roll

def judge_state(ear, mar, adjusted_yaw, adjusted_pitch):
    global sleep_start_time, yawn_start_time, distraction_start_time

    current_time = time.time()

    sleep_detected = False
    yawn_detected = False
    distraction_detected = False

    if ear < ear_threshold:
        if sleep_start_time is None:
            sleep_start_time = current_time

        if current_time - sleep_start_time >= SLEEP_TIME:
            sleep_detected = True
    else:
        sleep_start_time = None

    if mar > MAR_THRESHOLD:
        if yawn_start_time is None:
            yawn_start_time = current_time

        if current_time - yawn_start_time >= YAWN_TIME:
            yawn_detected = True
    else:
        yawn_start_time = None

    if abs(adjusted_yaw) > YAW_THRESHOLD or abs(adjusted_pitch) > PITCH_THRESHOLD:
        if distraction_start_time is None:
            distraction_start_time = current_time

        if current_time - distraction_start_time >= DISTRACTION_TIME:
            distraction_detected = True
    else:
        distraction_start_time = None

    if sleep_detected:
        return "sleep"
    elif yawn_detected:
        return "yawn"
    elif distraction_detected:
        return "distraction"
    else:
        return "normal"

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
            yaw_values.append(yaw)
            pitch_values.append(pitch)

            cv2.putText(frame, "Calibrating... Look straight normally", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"Time: {elapsed_time:.1f}/{CALIBRATION_TIME}s", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if elapsed_time >= CALIBRATION_TIME:
                avg_ear = np.mean(ear_values)
                ear_threshold = avg_ear * EAR_RATIO

                yaw_baseline = np.mean(yaw_values)
                pitch_baseline = np.mean(pitch_values)

                print("Calibration complete")
                print(f"Average EAR: {avg_ear:.3f}")
                print(f"EAR_THRESHOLD: {ear_threshold:.3f}")
                print(f"YAW_BASELINE: {yaw_baseline:.3f}")
                print(f"PITCH_BASELINE: {pitch_baseline:.3f}")

        else:
            adjusted_yaw = yaw - yaw_baseline
            adjusted_pitch = pitch - pitch_baseline

            state = judge_state(ear, mar, adjusted_yaw, adjusted_pitch)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"EAR_TH: {ear_threshold:.2f}", (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"YAW: {adjusted_yaw:.1f}", (30, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.putText(frame, f"PITCH: {adjusted_pitch:.1f}", (30, 180),
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