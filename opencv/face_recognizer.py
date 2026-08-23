import cv2 as cv
import os


people = []
for D in os.listdir(''):#put your persons folder path here
    people.append(D)

# load trainer.yml we've create.
face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.read('trainer.yml')

# read the img that want to recognize.
img = cv.imread('') #put your img path here
img  = cv.resize(img,(300,400))
gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# img's face detection and run recognition predict.
haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
detected_face_Rect_coordinate = haarcascade.detectMultiScale(gray_img,1.1,5) #change these parameters if face detect was wrong.
for (x,y,w,h) in detected_face_Rect_coordinate:
    cropped_face = gray_img[y:y+h,x:x+w]
    label , confidence = face_recognizer.predict(cropped_face)
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    cv.putText(img,f'{people[label]} with confidence {confidence:.2f}',(10,20),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)

cv.imshow('prediction',img)

cv.waitKey(0)