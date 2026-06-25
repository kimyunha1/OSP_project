# 컴퓨터 비전 기반 실시간 운전자 이상 행동 감지 시스템 
## 1.  About the Project
스마트 기기 활용 증대 및 장거리 운전 환경 변화로 인해 발생하는 졸음운전 및 전방 주시 태만 등 운전자의 부주의를 방지하기 위한 시스템입니다. 기존 차량 중심에서 운전자 상태 중심 모니터링 체계로 전환하여 대형 인명 사고를 예방하고자 합니다. 컴퓨터 비전 및 인공지능 기술(YOLOv5)을 활용해 운전자의 상태를 실시간으로 분석하고 이상 행동을 감지합니다.

## 2. Setup and Installation
- 요구사항:

  - Python 3.x

  - Google Colab (GPU)

  - PyTorch, OpenCV 등

- 설치 및 세팅:

  - YOLOv5 레포지토리 클론 및 의존성 패키지 설치
  
```
    git clone https://github.com/ultralytics/yolov5

    cd yolov5

    pip install -r requirements.txt
```
  - 사전 학습 가중치 다운로드
 ``` 
    wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```
## 3. Usage and Execution Results

###  Execution (실행 방법)
통합 제어 마스터 시스템(`main.py`)을 구동하여 실시간 웹캠 영상을 기반으로 운전자의 상태를 상시 모니터링합니다.

# 통합 시스템 메인 루프 실행
python main.py

Core File Structure (주요 파일 구조)

이 시스템은 마스터 시스템을 중심으로 각 모듈이 독립적이고 유기적으로 작동하도록 설계되었습니다.

main.py: 시스템의 메인 허브 엔진으로, 실시간 영상 스트리밍(OpenCV)을 제어하고 상태 트래킹 인터페이스를 총괄합니다.

ear_mar_detection.py: MediaPipe Face Mesh를 활용하여 운전자의 눈(EAR) 및 입(MAR)의 상태, 안면 각도(Head Pose)를 산출해 냅니다.

detect_danger.py: ONNX Runtime 최적화 엔진 환경을 통해 주행 중 스마트폰 사용등의 위험 객체를 실시간으로 정밀 탐지합니다.

alert.py: 탐지된 위험 등급에 따라 시청각 경고(Pillow 한글 경고문, winsound 비프음 및 pyttsx3 음성 안내)를 비동기 백그라운드 스레드로 처리합니다.

◦ Execution Results (실행 결과 예시)
프로그램이 정상 구동된 웹캠 화면 


<img width="863" height="665" alt="image" src="https://github.com/user-attachments/assets/3cccbdae-9340-437c-afb6-9fe525a28bd2" />
<img width="861" height="673" alt="image" src="https://github.com/user-attachments/assets/fb0592ac-8c1d-4c63-b4e1-f937fb5342a1" />
<img width="861" height="635" alt="image" src="https://github.com/user-attachments/assets/5f5dcefa-06bc-44c1-b050-6a75b83462d6" />
<img width="811" height="649" alt="image" src="https://github.com/user-attachments/assets/7972ba40-a45f-45ec-ae78-dd832f7b74b7" />
<img width="1096" height="885" alt="image" src="https://github.com/user-attachments/assets/9d403817-0b14-4173-82a1-72baa09a3044" />







## 4. License

MIT License를 따릅니다.

## 5. How to Contribute

이 저장소를 Fork 합니다.

새로운 기능 브랜치를 생성합니다. (git checkout -b feature/NewFeature)

변경 사항을 커밋합니다. (git commit -m 'Add some NewFeature')

브랜치에 Push 합니다. (git push origin feature/NewFeature)

Pull Request(PR)를 생성해 주세요.
