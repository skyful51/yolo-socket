import socket
import pickle
import cv2
import numpy as np
import struct

HOST_ADDR = '127.0.0.1'
HOST_PORT = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST_ADDR, HOST_PORT))
print(HOST_ADDR)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, dsize=None, fx=0.6, fy=0.6)
    pickled_img = pickle.dumps(frame)
    msg_size = struct.pack("L", len(pickled_img))
    client_socket.sendall(msg_size + pickled_img)
    data = client_socket.recv(1024)
    print(f'{HOST_ADDR}: {data.decode()}')

# import socket
# my_socket = socket.socket()
# my_socket.connect(('127.0.0.1', 8820))
# message = input('Enter some data: ')
# my_socket.send(message.encode())
# response_data = my_socket.recv(1024)
# print("Received: %s" % response_data.decode())
# my_socket.close()