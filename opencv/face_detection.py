import cv2 as cv

img = cv.imread('') #put your img here
img = cv.resize(img,(300,400))
gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# CascadeClassifier is a class that read a pretrained data from xml file.
haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# detectMultiScale return a list of detected faces . each item in the list contains a face coordinates (x,y,w,h)
detected_faces = haarcascade.detectMultiScale(gray_img,1.1,3)

print(f'number of faces found : {len(detected_faces)}')

for i,(x,y,w,h) in enumerate(detected_faces):
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    cv.putText(img,f'face number {i+1}',(x,y-5),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

cv.imshow('detected faces',img)
cv.waitKey(0)