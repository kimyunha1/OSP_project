import cv2
import mediapipe as mp
import math
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6

SLEEP_TIME = 2.0
YAWN_TIME = 1.0

sleep_start_time = None
yawn_start_time = None

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

def judge_state(ear, mar):
    global sleep_start_time, yawn_start_time

    current_time = time.time()
    state = "normal"

    if ear < EAR_THRESHOLD:
        if sleep_start_time is None:
            sleep_start_time = current_time

        sleep_duration = current_time - sleep_start_time

        if sleep_duration >= SLEEP_TIME:
            state = "sleep"
    else:
        sleep_start_time = None

    if mar > MAR_THRESHOLD:
        if yawn_start_time is None:
            yawn_start_time = current_time

        yawn_duration = current_time - yawn_start_time

        if yawn_duration >= YAWN_TIME:
            state = "yawn"
    else:
        yawn_start_time = None

    return state

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라를 열 수 없습니다.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        ear = calculate_ear(landmarks)
        mar = calculate_mar(landmarks)
        state = judge_state(ear, mar)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"STATE: {state}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("EAR MAR Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()