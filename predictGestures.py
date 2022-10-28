import mediapipe as mp
import pandas as pd
import pickle
import os
import cv2
import numpy as np

cd = os.getcwd()

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True,max_num_hands=1, min_detection_confidence=0.5)

with open('svm.pkl', 'rb') as f:
    model = pickle.load(f)

def getData(frame):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = hands.process(imgRGB)

    if output.multi_hand_landmarks:
        data = output.multi_hand_landmarks[0]
        lms = []
        for lm in data.landmark:
            lms.extend([lm.x, lm.y, lm.z])
        return lms
    return None


def main(cam):

    cap = cv2.VideoCapture(cam)
    while True:
        _, frame = cap.read()
        data = getData(frame)
        if data:
            data = np.array(data)
            pred = model.predict(data.reshape(-1,63))
            print(pred)
        else:
            print("None")
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
            

    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    cam = int(input("Enter Camera No:"))
    main(cam)