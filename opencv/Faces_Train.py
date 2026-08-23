
import os
import cv2 as cv
import numpy as np

# create a list of directoies(people's name) name.
people = []
for D in os.listdir(''):#put your persons folder path here
    people.append(D)

DIR = '' #put your persons folder path here
haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

features = []
labels = []

def Create_Train():
    for person in people:
        person_path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path,img)
            img_array = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
            detected_face_Rect_coordinate = haarcascade.detectMultiScale(img_array,1.1,3)
            for (x,y,w,h) in detected_face_Rect_coordinate:
                cropped_face = img_array[y:y+h,x:x+w]
                features.append(cropped_face)
                labels.append(label)
    
    print(f'number of detected features: {len(features)} ')
    print(f' number of detected labels: {len(labels)}')
    if len(features) != len(labels):
        raise ValueError('somtiong went wrong! the number of features and labels must be the same!')

Create_Train()

face_recognizer = cv.face.LBPHFaceRecognizer.create()
labels_np_arr = np.array(labels)
face_recognizer.train(features,labels_np_arr)
face_recognizer.save('trainer.yml')
print('Training done...')
            




