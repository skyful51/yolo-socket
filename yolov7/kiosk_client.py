import socket
import argparse
import time
import cv2
import numpy as np
import pickle
from threading import Thread
import queue
import random

# q = queue.Queue()



# def receive_function():
#     cap = cv2.VideoCapture('rtsp://admin:hikvision123@192.168.11.69:554/ISAPI/streaming/channels/101')
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, dsize=None, fx=0.25, fy=0.25)
#     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#     q.put(frame)
#     while ret:
#         ret, frame = cap.read()
#         q.put(frame)

# def display_function():
#     while True:
#         if q.empty() != True:
#             frame = q.get()
#             cv2.imshow("frame", frame)
#         if cv2.waitKey(1) == ord('q'):
#             break

cls_name = ['Face', 'Face_Shield', 'Fire_Gloves', 'Gas_Mask_Full', 'Gas_Mask_Half', 'Glasses', 'Hazmat_Coat', 'Lab_Coat', 'Latex_Gloves', 'Mask', 'No_Gloves', 'Normal_Shoes', 'Safety_Glasses', 'Safety_Hat', 'Sandal', 'Work_Gloves']
color_index = [(255,0,0), (255,140,0), (255,215,0), (255,255,0), (127,255,0), (0,128,0), (0,255,255), (100,149,237), (25,25,112), (138,43,226), (186,85,211), (255,0,255), (250,250,210), (230,230,250), (0,0,0), (169,169,169), (205,133,63)]

def my_function(cls_matrix, img):
    for i in range(len(cls_matrix)):
        list_a = list(np.where(cls_matrix[i] > 0)[0])

        if len(list_a) / len(cls_matrix[0]) > 0.8:
            cv2.putText(img, f"{cls_name[i]} Detected!", (10, 30 * (i+1)), cv2.LINE_AA, 1, (0,0,255), 2)
            
class LoadVideoStream:
    def __init__(self, cam_source):

        self.video_frame = None

        # 비디오 캡처 객체 생성
        print("trying to connect to camera...")
        self.cap = cv2.VideoCapture(cam_source)
        print("connected to camera!")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) % 100

        # 첫번째 프레임 받아오는 것을 보장
        _, self.video_frame = self.cap.read()
        self.video_frame = cv2.rotate(self.video_frame, cv2.ROTATE_90_CLOCKWISE)

        # 비디오 프레임 받아오는 스레드 시작
        thread = Thread(target=self.update, args=([self.cap]), daemon=True)
        thread.start()

    # 멀티스레드 상에서 영상 스트림을 업데이트 하는 메서드
    def update(self, cap):
        n = 0

        while cap.isOpened():
            n += 1
            cap.grab()

            if n == 1: # 4번째 프레임마다 읽어온다
                success, im = cap.retrieve()
                # im = cv2.resize(im, dsize=None, fx=0.5, fy=0.5)
                im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                self.h, self.w, _ = im.shape
                self.video_frame = im
                n = 0
                # time.sleep(1 / self.fps)

    def release(self, cap):
        cap.release()
        cv2.destroyAllWindows()

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    #color = [random.randint(0, 255) for _ in range(3)] ,color = (0, 255, 0) 
    color = color_index[int(cls_name.index(label))]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main_function():
    cls_matrix = np.zeros((17,80))

    # argument 정보 반영
    HOST_ADDRESS, HOST_PORT, CAM_ADDRESS = \
        opt.host_id, opt.host_port, f'rtsp://admin:hikvision123@{opt.src}:554/ISAPI/streaming/channels/101'
    
    # # 비디오 캡처 객체
    print("connecting ipcamera...")
    # cap = cv2.VideoCapture(CAM_ADDRESS)
    print("ipcamera connected!")

    # 클라이언트 소켓 객체
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_connected = False

    # 서버 연결 시도 (5번 실패 시 프로그램 종료)
    for i in range(5):
        try:
            client_socket.connect((HOST_ADDRESS, HOST_PORT))
            is_connected = True
            break
        except Exception as e:
            is_connected = False
            print(f'Error occured while connecting to Server. Trying again...{i + 1}', end='\r')
            time.sleep(1)

    # 프레임 받아오는 스레드 시작
    stream = LoadVideoStream(CAM_ADDRESS)

    # p1 = threading.Thread(target=receive_function)
    # p2 = threading.Thread(target=display_function)
    # p1.start()
    # p2.start()

    if is_connected:

        while True:
            
            # 비디오 캡처 객체로부터 프레임 읽어오기
            # ret, frame = cap.read()

            # if not ret:
            #     continue

            # 서버로부터 종료 신호를 수신하기 전까지 무한 루프
            cls_in_frame = [0 for i in range(17)]
            data = client_socket.recv(1024)
            try:
                cmd = pickle.loads(data)
            except Exception as e:
                continue
            data = None
                
            # 서버로부터 수신한 객체 탐지 결과(좌표 정보) 출력
            # 나중에 bbox 그리는 코드로 대체할 것
            print(f'Received: {cmd}')
            print((stream.w, stream.h))

            try:
                for d in cmd:
                    # cv2.rectangle(stream.video_frame, (d[1], d[2]), (d[3], d[4]), (0, 255, 0), 1)
                    # cv2.rectangle(stream.video_frame, (), (), (), -1)               
                    plot_one_box(d[1:], stream.video_frame, label=d[0])
                    #plot_one_box(d[1:], stream.video_frame, label=d[0])
                    cls_in_frame[cls_name.index(d[0])] = 1
            except TypeError:
                print("nothing detected")
            
            cmd = None
            cls_matrix = np.delete(cls_matrix, 0, axis=1)
            cls_matrix = np.c_[cls_matrix, cls_in_frame]
            my_function(cls_matrix, stream.video_frame)

            # 클라이언트 모니터에 영상 재생
            cv2.imshow('frame', stream.video_frame)

            if cv2.waitKey(1) == ord('s'):
                break

        # 비디오 캡처 객체 해제
        stream.release(stream.cap)

    # 소켓 통신 종료
    client_socket.close()

if __name__ == "__main__":
    
    # 필요한 argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--host-id', type=str, required=True, help='server ip address')
    parser.add_argument('--server-port', type=int, required=True, help='server port number')
    parser.add_argument('--src', type=str, required=True, help='ipcam address. please type IPv4 address only.')
    opt = parser.parse_args()

    # 메인 함수 시작
    main_function()