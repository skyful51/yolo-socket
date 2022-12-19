import socket
from threading import Thread
import pickle
import cv2
import numpy as np
import struct
import argparse
from datetime import datetime
import time

# 클라이언트 클래스
class Client:

    # 생성자 함수
    def __init__(self, host_addr, host_port, cam_src):
        self.HOST_ADDR = host_addr  # 연결할 서버의 주소
        self.HOST_PORT = host_port  # 연결할 서버의 포트 번호
        self.CAM_SRC = cam_src      # 카메라 소스(웹캠 인덱스 or IP 카메라 주소

        self.cap_frame = 0          # 멀티스레드 상에서 얻은 프레임 수 인덱스
        self.sent_frame = 0         # 소켓 통신으로 전송한 프레임 수 인덱스

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 인스턴스 내에서 사용할 소켓 객체

        for i in range(5):
            try:
                self.socket.connect((self.HOST_ADDR, self.HOST_PORT))
                self.is_connected = True
                break
            except Exception as e:
                self.is_connected = False
                print(e)

                if i >= 4:
                    exit()

        self.create_cap()

        ret, self.raw_frame = self.cap.read()
        self.enc_frame = cv2.imencode('.webp', self.raw_frame)
        print(self.raw_frame.shape)

        thread_1 = Thread(target=self.get_frame, args=([self.cap]), daemon=True)
        thread_1.start()
        
    # 비디오 캡처 객체 생성
    def create_cap(self):
        print('connecting to camera...')
        self.cap = cv2.VideoCapture(self.CAM_SRC)
        print('connected to camera')

    # 별도의 스레드를 생성해 이미지 프레임 생성
    def get_frame(self, cap):
        n = 0

        while cap.isOpened():
            n += 1
            cap.grab()

            # 멀티스레드로 받아오는 프레임을 4개에 하나만 사용
            if n == 4:
                success, im = cap.retrieve()
                im = cv2.resize(im, dsize=None, fx=0.3, fy=0.3)
                ret, self.enc_frame = cv2.imencode('.webp', im)
                ts = datetime.fromtimestamp(time.time())
                print(f'captured {self.cap_frame} th frame at {ts}')
                self.cap_frame += 1
                n = 0

    # 캡처 객체 해제
    def release_cap(self, cap):
        cap.release()
        cv2.destroyAllWindows()

    # 이미지를 바이너리로 변환 후 서버로 전송
    def send_to_server(self):
        try:
            self.pickle_frame = pickle.dumps(self.enc_frame)  # 이미지 프레임 바이너리화
            self.msg_size = struct.pack("L", len(self.pickle_frame))
            self.socket.sendall(self.msg_size + self.pickle_frame)
        except Exception as e:
            pass
    
    # 서버로부터 데이터 수신
    def receive_from_server(self):
        self.received_data = self.socket.recv(1024)
    
    def close_socket(self):
        self.socket.close()

def main_function():

    # hikvision address: rtsp://admin:hikvision123@192.168.11.69:554/ISAPI/streaming/channels/101
    client = Client(host_addr=opt.host_id, host_port=opt.host_port, cam_src=opt.src)

    try:
        while True:
            client.send_to_server()
            client.receive_from_server()
    except Exception as e:
        print(e)
        client.close_socket()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--host-id', type=str, required=True, help='host id or ip information.')
    parser.add_argument('--host-port', type=int, default=9999, help='host port number. default to 9999.')
    parser.add_argument('--src', required=True, help='camera id(webcam) or ip(ip cam) information.')
    opt = parser.parse_args()
    print(opt)

    main_function()