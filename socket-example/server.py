import socket
import pickle
import cv2
import numpy as np
import struct

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils2 import select_device, load_classifier, time_synchronized, TracedModel

HOST_ADDR = '127.0.0.1'
HOST_PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_ADDR, HOST_PORT))
server_socket.listen()
client_socket, client_addr = server_socket.accept()
print(client_addr)

data = b''
palyload_size = struct.calcsize("L")

while True:

    while len(data) < palyload_size:
        data += client_socket.recv(4096)

    packed_mgs_size = data[:palyload_size]
    data = data[palyload_size:]
    msg_size = struct.unpack("L", packed_mgs_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data)
    print(type(frame))
    cv2.imshow("server window", frame)
    cv2.waitKey(1)
    client_socket.send('hi'.encode())

# import socket

# server_socket = socket.socket()
# server_socket.bind(('127.0.0.1', 8820))

# server_socket.listen(1)

# (client_socket, client_address) = server_socket.accept()

# client_data = client_socket.recv(1024)
# print("Received: %s" % client_data.decode())
# client_socket.send('Hello!'.encode())

# client_socket.close()
# server_socket.close()