import cv2 as cv

haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    detected_face = haarcascade.detectMultiScale(gray_frame,1.1,3)
    for (x,y,w,h) in detected_face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
    cv.imshow('webcam',frame)
    if cv.waitKey(1) and 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()