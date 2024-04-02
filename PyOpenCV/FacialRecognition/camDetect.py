import cv2
import os

class Detection:
    def __init__(self, model='haarcascade_frontalface_default.xml'):
        self.model = model
        self.image = {"color": 'grey', "scale": 1.3, "min_neighbors": 5, "local": "./dataset/"}
        self.classes = {}
        self.nclasses = None

    def parse_data(self):
        if not os.path.exists("dataset/"):
            print("Dataset folder does not exist.\n")
            exit(0)

        for data in os.listdir("dataset/"):
            filename = data.split(".")
            try:
                self.classes[filename[1]]['size'] += 1
            except:
                try:
                    self.classes[filename[1]] = {'size': 1, 'name': filename[2]}
                except:
                    pass

        self.nclasses = len(self.classes)

    def update_data(self, face_id, face_name):
        if not self.classes:
            self.parse_data()

        try:
            if self.classes[str(face_id)]['name'] == face_name:
                self.capture(face_name, str(face_id))
            else:
                print("*ERROR* This ID and label name do not match.")
                print(self.classes[str(face_id)]['name'], "diff >", face_name)
        except KeyError:
            print("*ERROR* This ID does not exist.")

    def capture(self, face_name, face_id=None, nclips=60):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # Set video width
        cam.set(4, 480)  # Set video height
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        if face_id is None:
            print("New training label")
            face_id = self.nclasses + 1
            for face_id in range(self.nclasses + 1, 100, 1):
                if str(face_id) not in self.classes:
                    break
            if self.nclasses > 99:
                print("Dataset indexes number was exceeded\n")
                exit(0)
            count = 0
        else:
            try:
                print("Updating User." + str(face_id) + '.' + str(face_name))
                count = self.classes[str(face_id)]["size"] + 1
                print(count, "clips")
            except KeyError:
                print("*ERROR* Check the input arguments; they may be incorrect.")
                exit(0)

        string_name = "User." + str(face_id) + '.' + str(face_name)
        print("\n [INFO] Initializing face capture. Look at the camera and wait...")

        i = count
        while True:
            ret, img = cam.read()
            img = cv2.flip(img, 1)  # Flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                i += 1
                wstatus = cv2.imwrite("dataset/User." + str(face_id) + '.' + str(face_name) + '.' + str(i) + ".jpg",
                                      gray[y:y + h, x:x + w])

                if not wstatus:
                    print("Issue found at saving data")

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' to exit video
            if k == 27:
                break
            elif i >= count + nclips:
                break

        print("\n [INFO] Exiting program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

# Example
if __name__ == "__main__":
    detector = Detection()
    detector.parse_data()
    detector.capture("Bruno", 5, nclips=60)
