import cv2
import numpy as np
from KalmanFilter import KalmanFilter
from Detector import detect

dt = 0.1
u_x = 1
u_y = 1
std_acc = 1
x_std_meas = 0.1
y_std_meas = 0.1
kf = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

cap = cv2.VideoCapture('2D_KalMan-Filter_TP1/video/randomball.avi')

trajectory = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    centers = detect(frame)

    if centers:
        detected_center = centers[0]
        predicted = kf.predict()
        estimated = kf.update(detected_center)
        cv2.circle(frame, (int(detected_center[0][0]), int(detected_center[1][0])), 5, (0, 255, 0), -1)
        cv2.rectangle(frame, (int(predicted[0] - 10), int(predicted[1] - 10)),
                      (int(predicted[0] + 10), int(predicted[1] + 10)), (255, 0, 0), 2)
        cv2.rectangle(frame, (int(estimated[0] - 10), int(estimated[1] - 10)),
                      (int(estimated[0] + 10), int(estimated[1] + 10)), (0, 0, 255), 2)
        trajectory.append((int(estimated[0]), int(estimated[1])))
    for point in trajectory:
        cv2.circle(frame, point, 2, (0, 255, 255), -1)
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()