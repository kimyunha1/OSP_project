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

    ear = (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))
    return ear

def calculate_mar(landmarks):
    left = landmarks[61]
    right = landmarks[291]
    upper = landmarks[13]
    lower = landmarks[14]

    mar = distance(upper, lower) / distance(left, right)
    return mar

def calculate_head_direction(landmarks):
    nose = landmarks[1]
    left_face = landmarks[234]
    right_face = landmarks[454]

    face_center_x = (left_face.x + right_face.x) / 2
    face_width = abs(right_face.x - left_face.x)

    if face_width == 0:
        return "front"

    nose_offset = (nose.x - face_center_x) / face_width

    if nose_offset > 0.12:
        return "left"
    elif nose_offset < -0.12:
        return "right"
    else:
        return "front"

def judge_state(ear, mar, head_direction):
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

    if head_direction != "front":
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

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        ear = calculate_ear(landmarks)
        mar = calculate_mar(landmarks)
        head_direction = calculate_head_direction(landmarks)

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
            state = judge_state(ear, mar, head_direction)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"EAR_TH: {ear_threshold:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"HEAD: {head_direction}", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(frame, f"STATE: {state}", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("EAR MAR Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()