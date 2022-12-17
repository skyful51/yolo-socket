import socket
from threading import Thread
import pickle
import cv2
import numpy as np
import struct
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
from numpy import random
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets2 import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# 서버측 클래스
class Server:

    # 생성자
    def __init__(self, host_addr, host_port):
        self.HOST_ADDR = host_addr  # 자신의 주소
        self.HOST_PORT = host_port  # 개방할 포트 번호

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.HOST_ADDR, self.HOST_PORT))

        self.listen_to_client()

        self.data_received = None
        self.img = None

        # argparse
        self.source, self.weights, self.view_img, self.save_txt, self.imgsz, self.trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        self.save_img = not opt.nosave and not self.source.endswith('.txt')  # save inference images
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

        self.device = select_device(opt.device)

    # 클라이언트로부터 연결 대기
    def listen_to_client(self):
        print('listening connection')
        self.socket.listen()
        self.client_socket, self.client_addr = self.socket.accept()
        print('connected by: ', self.client_addr)

    # 클라이언트로부터 이미지 패킷 수신
    def receive_from_client(self):
        data = b''
        palyload_size = struct.calcsize("L")

        try:
            while len(data) < palyload_size:
                data += self.client_socket.recv(4096)

            packed_mgs_size = data[:palyload_size]
            data = data[palyload_size:]
            msg_size = struct.unpack("L", packed_mgs_size)[0]

            while len(data) < msg_size:
                data += self.client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            self.img = pickle.loads(frame_data)
        except Exception as e:
            print(e)

    # 클라이언트로 수신 완료 에코
    def send_to_client(self):
        self.client_socket.send('good'.encode())

    # 이미지 표출
    def show_frame(self):
        try:
            cv2.imshow('server window', self.img)
        except Exception as e:
            print(e)

    # 소켓 통신 종료    
    def close_socket(self):
        self.socket.close()

    # yolov7 Inference 관련 코드
    def yolo_init(self):

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        if self.half:
            self.model.half()

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

    # Run inference
    def yolo_inference(self):
        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, frame=self.img)

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0    # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():
                self.pred = self.model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            self.pred = non_max_suppression(self.pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if self.classify:
                self.pred = apply_classifier(self.pred, self.modelc, img, im0s)

            # Process detections
            for i, det in enumerate(self.pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                    #     if save_txt:  # Write to file
                    #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #         with open(txt_path + '.txt', 'a') as f:
                    #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.view_img:  # Add bbox to image
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if self.view_img:
                    cv2.imshow("inference window", im0)
                    cv2.waitKey(1)

        # t0 = time.time()
        # img = self.img.copy()
        # img = torch.from_numpy(img).to(self.device)
        # img = img.half() if self.half else img.float()   # uint8 to fp16/32
        # img /= 255.0 # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)

        # # Warmup
        # if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
        #     self.old_img_b = img.shape[0]
        #     self.old_img_h = img.shape[2]
        #     self.old_img_w = img.shape[3]
        #     for i in range(3):
        #         self.model(img, augment=opt.augment)[0]

        # # Inference
        # t1 = time_synchronized()
        # with torch.no_grad():
        #     self.pred = self.model(img, augment=opt.augment)[0]
        # t2 = time_synchronized()

        # # Apply NMS
        # # self.pred = non_max_suppression(self.pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # # t3 = time_synchronized()

        # # Apply Classifier
        # # if self.classify:
        # #     self.pred = apply_classifier(self.pred, self.modelc, self.img, im0)

        # # Process detections
        # for i, det in enumerate(self.pred):
        #     print(det)
        
        
def main_function():
    server = Server(opt.host_id, opt.host_port)
    server.yolo_init()

    while True:
        server.receive_from_client()
        # server.show_frame()
        server.yolo_inference()
        server.send_to_client()

        if cv2.waitKey(1) == ord('q'):
            break

    server.close_socket()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/ima', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--host-id', type=str, required=True, help='host id or ip information.')
    parser.add_argument('--host-port', type=int, default=9999, help='host port number. default to 9999.')
    opt = parser.parse_args()
    print(opt)

    main_function()