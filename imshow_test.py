import cv2
import numpy as np

cap = cv2.VideoCapture(0)
print('h')

while cap.isOpened():

    ret, frame = cap.read()
    print('hh')

    if not ret:
        continue

    cv2.imshow("window", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()