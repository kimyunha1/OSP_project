import cv2
import mediapipe as mp
import math
import time
import numpy as np
import os
import winsound
from PIL import ImageFont, ImageDraw, Image

import ear_mar_detection as emd
from ear_mar_detection import calculate_ear, calculate_mar, calculate_head_pose, DriverStateTracker
from detect_danger import load_model, run_yolo_detection
from alert import update_counters, get_stage, trigger_alert

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

CALIBRATION_TIME = 3.0
EAR_RATIO = 0.8
YAW_THRESHOLD = 20     
PITCH_THRESHOLD = 20  

ear_threshold = None
ear_values = []
yaw_values = []
pitch_values = []
yaw_baseline = None
pitch_baseline = None
calibration_start_time = time.time()

tracker = DriverStateTracker()
session = load_model("best.onnx")

font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else None

# 메인 루프
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
    mp_state = "normal"

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
                emd.ear_threshold = ear_threshold

                yaw_baseline = np.mean(yaw_values)
                pitch_baseline = np.mean(pitch_values)

                print("Calibration complete!")
                print(f"Average EAR: {avg_ear:.3f}")
                print(f"EAR_THRESHOLD: {ear_threshold:.3f}")
                print(f"YAW_BASELINE: {yaw_baseline:.3f}")
                print(f"PITCH_BASELINE: {pitch_baseline:.3f}")
        
        else:
            adjusted_yaw = yaw - yaw_baseline
            adjusted_pitch = pitch - pitch_baseline
            
            mp_state = tracker.judge_state(ear, mar, adjusted_yaw, adjusted_pitch,ear_threshold)

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

            cv2.putText(frame, f"STATE: {mp_state}", (30, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    # ---------------------------------------------------------
    # YOLOv5 스마트폰 탐지 
    # ---------------------------------------------------------
    yolo_state, frame = run_yolo_detection(session, frame)

    # 위험 등급 우선순위에 따른 최종 상태 결정 (스마트폰/졸음 > 시선이탈 > 하품)
    if yolo_state == "phone":
        final_state = "phone"
    elif mp_state == "sleep":
        final_state = "sleep"
    elif mp_state == "yawn":
        final_state = "yawn"
    elif mp_state == "distraction":
        final_state = "distraction"
    else:
        final_state = "normal"

    # 경고 시나리오
    update_counters(final_state)
    current_stage, _ = get_stage()
    text_to_show, text_color = trigger_alert(current_stage, final_state)

    # 최종 경고 문구 출력 (한글 깨짐 방지)
    if font:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        fill_color = (text_color[2], text_color[1], text_color[0])
        draw.text((30, height - 50), text_to_show, font=font, fill=fill_color)
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(frame, text_to_show, (30, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)

    
    cv2.imshow("Driver Monitoring Master System", frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()