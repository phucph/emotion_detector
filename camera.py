from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2


# print(model.summary())
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # (grabbed, frame) = self.video.read()
        _, frame = self.video.read()
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        model = load_model('checkpoints/epoch_75.hdf5')
        EMOTIONS = ["angry", "scared", "happy", "sad", "surprised",
                    "neutral"]
        
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # we can draw on it
        rects = detector.detectMultiScale(gray, scaleFactor=1.3,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        # ensure at least one face was found before continuing
        if len(rects) > 0:
            # determine the largest face area
            rect = sorted(rects, reverse=True,
                          key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = rect

            # extract the face ROI from the image, then pre-process
            # it for the network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            # label
            preds = model.predict(roi)[0]
            label = EMOTIONS[preds.argmax()]

            # loop over the labels + probabilities and draw them
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
                w = int(prob * 300)
                cv2.rectangle(frame, (5, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(frame, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 0, 0), 2)
                cv2.putText(frame, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV

        grabbed, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
