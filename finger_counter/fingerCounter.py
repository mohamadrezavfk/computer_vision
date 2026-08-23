import time
import cv2 as cv
import handLandmarker as hlm
import os
cap = cv.VideoCapture(0)
previous_time = 0
current_time = 0
fps = 0

handLandmarker = hlm.HandLandmarker()
finger_tips = [4, 8, 12, 16, 20]
overload_img = []
for img in os.listdir(r"fingerCountPhotos"):
    img_path = os.path.join("fingerCountPhotos", img)
    image = cv.imread(img_path)
    overload_img.append(image)
    
while True:
    current_time = time.time()
    new_fps = 1 / (current_time - previous_time)
    previous_time = current_time
    fps = 0.9 * fps + 0.1 * new_fps
    success, frame = cap.read()
    if not success:
        break
    result = handLandmarker.detect_hand(frame)
    landmarks = handLandmarker.landmarks_positions(result,frame)
    numberOfFingers = 0
    if len(landmarks) != 0:
        fingers_status = []
        if landmarks[finger_tips[0]][1] > landmarks[finger_tips[0]-1][1]:
            fingers_status.append(1)
        else : 
            fingers_status.append(0)
        for finger in range(1,5):
            if landmarks[finger_tips[finger]][2] < landmarks[finger_tips[finger]-2][2]:
                fingers_status.append(1)
            else : 
                fingers_status.append(0)
        numberOfFingers = fingers_status.count(1)
        print(fingers_status)


        frame[0:300,0:200] = overload_img[numberOfFingers-1]
        cv.putText(frame,f'Finger Count: {str(numberOfFingers)}',(10,350),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
    cv.putText(frame, f"{str(int(fps))} FPS", (520, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv.imshow("Camera", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()
    



