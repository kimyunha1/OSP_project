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

업데이트 예정

## 4. License

MIT License를 따릅니다.

## 5. How to Contribute

이 저장소를 Fork 합니다.

새로운 기능 브랜치를 생성합니다. (git checkout -b feature/NewFeature)

변경 사항을 커밋합니다. (git commit -m 'Add some NewFeature')

브랜치에 Push 합니다. (git push origin feature/NewFeature)

Pull Request(PR)를 생성해 주세요.
