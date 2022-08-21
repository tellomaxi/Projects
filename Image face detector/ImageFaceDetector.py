#Script - Face detector from pictures on internet

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


image = cv2.imread('/Users/maximilianotello/Downloads/prueba6.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,    #It is the image where the face detector will act.
    scaleFactor = 1.1,     #It depends on the size of the photo. Specifies how much the image is to be reduced. If we give a very high value, fewer faces are detected. If we give a very low value, it will detect many faces.
    minNeighbors=5,        #Specifies the minimum number of bounding boxes or neighbors that a face must have for it to be detected as such.
    minSize=(30,30),       #Indicates the minimum possible size of the object.
    maxSize=(200,200))     #Indicates the maximum possible size of the object.

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,2),2)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()