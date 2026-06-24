import cv2
import numpy as np
import onnxruntime as ort
import os

from PIL import ImageFont, ImageDraw, Image

class_names = ["normal", "phone", "sleep", "yawn"]

def load_model(model_path="best.onnx"):
    # 1. ONNX 모델 파일 경로 설정
    if not os.path.exists(model_path):
        print(f"[에러] '{model_path}' 파일을 찾을 수 없습니다. OSP 폴더 안으로 이동 후 실행하세요.")
        return

    # 2. ONNX Runtime 세션 생성
    try:
        session = ort.InferenceSession(model_path)
        print("[정보] ONNX 모델 로드 성공!")
    except Exception as e:
        print(f"[에러] 모델 로드 실패: {e}")
        return None
    
    return session

def run_yolo_detection(session, frame):    
        h_orig, w_orig = frame.shape[:2]

        # 이미지 전처리 (640x640 크기 맞춤)
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # ONNX 추론
        outputs = session.run(None, {session.get_inputs()[0].name: img})
        predictions = outputs[0][0]

        best_conf = 0.0
        best_class = "normal"

        # 객체 검출 및 사각형 박스 치기
        for pred in predictions:
            obj_conf = float(pred[4])                    
            if obj_conf < 0.6:
                continue
            
            class_scores = pred[5:] * obj_conf  
            class_id = np.argmax(class_scores)

            current_class = class_names[class_id]
            final_conf = float(class_scores[class_id])

            if final_conf < 0.6:
                continue

            cx, cy, w_box, h_box = pred[:4]
            x1 = int((cx - w_box / 2) * w_orig / 640)
            y1 = int((cy - h_box / 2) * h_orig / 640)
            x2 = int((cx + w_box / 2) * w_orig / 640)
            y2 = int((cy + h_box / 2) * h_orig / 640)

            if current_class == "phone":
                box_color = (0, 0, 255)
            elif current_class == "sleep":
                box_color = (255, 255, 0)
            elif current_class == "yawn":
                box_color = (0, 165, 255)
            else:
                box_color = (0, 255, 0)

            if current_class != "normal":
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            if final_conf > best_conf:
                best_conf = final_conf
                best_class = current_class

        return best_class, frame

