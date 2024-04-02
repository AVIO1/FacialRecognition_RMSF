import cv2
import numpy as np
from PIL import Image
import os
import requests as req
import json
from datetime import datetime
import base64
import time

service_id = 54321

class ImageRecogn:
    def __init__(self, path='dataset'):
        self.path = 'dataset'
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.names = {0: 'none'}
        self._historic = {}

    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []
        names = []

        for image_path in image_paths:
            try:
                pil_img = Image.open(image_path).convert('L')
                img_numpy = np.array(pil_img, 'uint8')
                id = int(os.path.split(image_path)[1].split(".")[1])
                name = str(os.path.split(image_path)[1].split(".")[2])

                faces = self.detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
                    names.append(name)
            except Exception as e:
                print('*Alert* Fail to read', image_path)
                print(e)
        return face_samples, ids, names

    def fit(self):
        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids, names = self.get_images_and_labels(self.path)
        self.recognizer.train(faces, np.array(ids))

        for i in range(1, len(np.unique(ids)) + 1):
            index = ids.index(i)
            self.recognizer.setLabelInfo(i, str(names[index]))

        self.recognizer.write('trainer/trainer.yml')
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    def get_labels(self, max_labels=100):
        print("Getting labels %i \n" % max_labels)
        for i in range(1, max_labels):
            retval = self.recognizer.getLabelInfo(i)
            if not retval:
                print('*Alert* empty labels')
                break
            else:
                self.names[i] = retval

    def classify(self, id, confidence=101):
        if confidence < 65:
            label = self.names[id]
        elif 65 < confidence < 101:
            label = str(self.names[id]) + " no match"
        else:
            label = "unknown"
        return label

    def predict(self):
        self.recognizer.read('trainer/trainer.yml')
        font = cv2.FONT_HERSHEY_SIMPLEX
        delta = 0
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)
        min_w = 0.1 * cam.get(3)
        min_h = 0.1 * cam.get(4)

        while True:
            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(min_w), int(min_h)),
            )

            for (x, y, w, h) in faces:
                label = None
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                label = self.classify(id, confidence)
                time2 = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                time_com = datetime.now()

                if label in self._historic and (time_com - self._historic[label]).total_seconds() > 5:
                    try:
                        print("Preparing to send:", label)
                        self._historic[label] = time_com
                        _, imdata = cv2.imencode('.JPG', img)
                        jpac = json.dumps({"image": base64.b64encode(imdata).decode('utf-8'), "time": time2,
                                           "token": 12345, "name": label})
                        req.put("https://rmsf-smartlock.ew.r.appspot.com/add/54321",
                                headers={'Content-type': 'application/json'}, json=jpac)
                    except Exception as e:
                        print(e)

                if not self._historic:
                    try:
                        print("Preparing to send 2:", label)
                        self._historic[label] = time_com
                        _, imdata = cv2.imencode('.JPG', img)
                        jpac = json.dumps({"image": base64.b64encode(imdata).decode('utf-8'), "time": time2,
                                           "token": 12345, "name": label})
                        req.put("https://rmsf-smartlock.ew.r.appspot.com/add/54321",
                                headers={'Content-type': 'application/json'}, json=jpac)
                    except Exception as e:
                        print(e)

                try:
                    door = req.get("https://rmsf-smartlock.ew.r.appspot.com/door/54321").json()
                    print("DOOR:", door)
                    if door.get("door") == 1:
                        pass  # Turn on the LED
                    elif door.get("door") == 0:
                        pass  # Turn off the LED
                except Exception as e:
                    print(e)

                cv2.putText(img, label, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                confidence = "  {0}%".format(confidence)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = ImageRecogn()
    recognizer.fit()
    recognizer.predict()
