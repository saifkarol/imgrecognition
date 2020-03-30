import os
from PIL import Image
import numpy as np
import cv2
import pickle

face_detection = cv2.CascadeClassifier(r'C:\Users\User\PycharmProjects\attendance\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "faces")

current_id = 0
label_ids = {}
y_labels = []
training_x = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith(".png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id = current_id + 1

            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")


            cv2.imshow("cd", image_array)
            cv2.waitKey()

            faces = face_detection.detectMultiScale(image_array)

            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                training_x.append(roi)
                y_labels.append(id_)


with open("Labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(training_x, np.array(y_labels))
recognizer.save("trainer.yml")
