from time import time
import math
import cv2 as cv
import handLandmarker as hlm
import numpy as np
from pycaw.pycaw import AudioUtilities

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
volumeLevel = 0
previous_time = time()
fps = 0
wcam, hcam = 640, 480
cap = cv.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
handlandmarker  = hlm.HandLandmarker()
while True:
        current_time = time()
        new_fps = 1 / (current_time - previous_time)
        previous_time = current_time
        fps = 0.9 * fps + 0.1 * new_fps
        success, frame = cap.read()
        if not success:
            break
        result = handlandmarker.detect_hand(frame)
        landmarks = handlandmarker.landmarks_positions(result,frame)
        if len(landmarks) !=0:
            x4,y4 = landmarks[4][1],landmarks[4][2]
            x8,y8 = landmarks[8][1],landmarks[8][2]
            cv.circle(frame, (x4, y4), 5, (255, 0, 255), cv.FILLED)
            cv.circle(frame, (x8, y8), 5, (255, 0, 255), cv.FILLED)
            cv.line(frame, (x4, y4), (x8, y8), (0, 255, 0), 2)
            distance  = math.hypot(x8-x4,y8-y4)
            volumeLevel = np.interp(distance, [20, 200], [0.0, 1.0])
            print(volumeLevel)
            volume.SetMasterVolumeLevelScalar(volumeLevel, None)
        cv.rectangle(frame , (50, 150), (80, 400), (255, 0, 0), 3)
        volumeBar = np.interp(volumeLevel, [0.0, 1.0], [400, 150])
        cv.putText(frame, f"{int(volumeLevel*100)} %", (50, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv.rectangle(frame , (50, 400),(80, int(volumeBar)), (255, 0, 0), cv.FILLED)
        cv.putText(frame, f"{str(int(fps))} FPS", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv.imshow("Camera", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv.destroyAllWindows()
        