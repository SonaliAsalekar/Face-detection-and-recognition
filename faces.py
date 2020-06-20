import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('C:\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
eye_cascade = cv2.CascadeClassifier('C:\Python\Python37\Lib\site-packages\cv2\data\haarcascade_eye.xml')

recognizer.read("trainner.yml")
labels = {"Person_Name":1}

with open("lablels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


#eye_cascade = cv2.CascadeClassifier('C:\Python\Python37\Lib\site-packages\cv2\data\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
    #    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] 
        id_,conf = recognizer.predict(roi_gray)
        if conf>=45:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        img_item = "my_image.png"
        cv2.imwrite(img_item,roi_gray)
        color = (255,0,0)
        stroke = 2
        cv2.rectangle(img,(x,y),(x+w,y+h),color,stroke)

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        
    #    eyes = eye_cascade.detectMultiScale(roi_gray)

    #    for (ex,ey,ew,eh) in eyes:
     #       cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

  #  cv2.imshow('img',img)
   # k = cv2.waitKey(30) & 0xff
    

cap.release()
cv2.destroyAllWindows()