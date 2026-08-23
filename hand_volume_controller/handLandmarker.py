import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



class HandLandmarker:

    def __init__(self, model_path=r"hand_landmarker.task", num_hands=2): #put the 'hand_landmarker.task' file path here
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options,num_hands=num_hands)
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def detect_hand(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
        result = self.detector.detect(mp_image)
        return result
    
    def draw_landmarks(self, frame, result):
         if result.hand_landmarks:
              for hand in result.hand_landmarks:
                  height , width, _ = frame.shape
                  for i,landmark in enumerate(hand):
                      x = int(landmark.x * width)
                      y = int(landmark.y * height)
                      z = int(landmark.z * width) 
                      cv.circle(frame,(x,y),4,(255,0,0),-1)
                      cv.putText(frame,str(i),(x+5,y-5),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

    def draw_connections(self, frame, result):
        HAND_CONNECTIONS = [
            (0,1), (1,2), (2,3), (3,4),

            (0,5), (5,6), (6,7), (7,8),

            (5,9), (9,10), (10,11), (11,12),

            (9,13), (13,14), (14,15), (15,16),

            (13,17), (17,18), (18,19), (19,20),

            (0,17)
        ]
        if result.hand_landmarks:
            height , width, _ = frame.shape
            for hand in result.hand_landmarks:
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start = hand[start_idx]
                    end = hand[end_idx]
                    x1 = int(start.x * width)
                    y1 = int(start.y * height)
                    x2 = int(end.x * width)
                    y2 = int(end.y * height)
                    cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def landmarks_positions(self, result , frame):
         landmarks = []
         if result.hand_landmarks:
              height , width, _ = frame.shape              
              for hand in result.hand_landmarks:
                  for i,landmark in enumerate(hand):
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        z = int(landmark.z * width) 
                        landmarks.append([i,x,y,z])
         return landmarks

    def get_handendness(self, result):
         handedness_info = []
         if result.handedness:
              for handedness in result.handedness:
                  handedness_info.append((handedness[0].category_name, handedness[0].score))
         return handedness_info

             
if __name__ == "__main__":
    import time
    cap = cv.VideoCapture(0)
    previous_time = 0
    current_time = 0
    fps = 0
    h = HandLandmarker()
    while True:
        current_time = time.time()
        new_fps = 1 / (current_time - previous_time)
        previous_time = current_time
        fps = 0.9 * fps + 0.1 * new_fps
        success, frame = cap.read()
        if not success:
            break
        result = h.detect_hand(frame)
        h.draw_landmarks(frame,result)
        if len(result.hand_landmarks)!=0:
            print(result.hand_landmarks[0],'\n\n')
        cv.putText(frame, f"{str(int(fps))} FPS", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv.imshow("Camera", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv.destroyAllWindows()
    