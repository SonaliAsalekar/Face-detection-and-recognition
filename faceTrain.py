import os
import cv2
import pickle
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"dataset2/")
#image_dir = os.path.join(BASE_DIR,"Atul11")

face_cascade = cv2.CascadeClassifier('C:\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("trainner.yml")
labels = {"Person_Name":1}

with open("lablels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
current_id = 0
label_ids = {}
Y_labels = []
X_train = []

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" " , "-").lower()
        #    print(label,path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id+=1
            id_ = label_ids[label]
        #    print(label_ids)
        

        #    Y_labels.append(label)
        #    X_train.append(path)
            pil_image = Image.open(path).convert("L")
            size = (550,550)
            image_array = np.array(pil_image,"uint8")
        #    print(image_array)
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors = 5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                X_train.append(roi)
                Y_labels.append(id_)
#print(Y_labels)
#print(X_train)

with open("lablels.pickle",'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(X_train,np.array(Y_labels))
recognizer.save("trainner.yml")