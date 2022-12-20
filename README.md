## yolov7-socket
---
#### 메인 코드
YOLOv7 detect 모델을 사용하시려면 detect.py 파일을 실행하시면 됩니다.
YOLOv7 + 소켓 통신을 같이 사용하시려면 서버에서는 ServerClass.py를, 클라이언트에서는 ClientClass.py를 실행하시면 됩니다.
---
#### 가중치 파일
https://drive.google.com/drive/folders/1x_PtKvTNjT3ewQwW7t3H0QHGzbIbk6hC?usp=sharing
키오스크용 객체 탐지 가중치 파일입니다.
    1. yolov7_w6_1280.pt
        1280 이미지 크기로 w6 모델을 학습시킨 가중치입니다.
    2. yolow_del-face_nomixup.py
        얼굴 클래스 없이 학습시킨 가중치입니다.