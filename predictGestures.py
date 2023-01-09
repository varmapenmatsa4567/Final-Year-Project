import mediapipe as mp
import pandas as pd
import pickle
import os
import cv2
import numpy as np
import sklearn
import pyautogui as pg
import screen_brightness_control as brc
import time

cd = os.getcwd()

mpHands = mp.solutions.hands
hands = mpHands.Hands()

with open('svm.pkl', 'rb') as f:
    model = pickle.load(f)

def getData(frame):
    img = frame
    if img is not None:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output = hands.process(imgRGB)

        if output.multi_hand_landmarks:
            data = output.multi_hand_landmarks[0]
            lms = []
            for lm in data.landmark:
                lms.extend([lm.x, lm.y, lm.z])
            return lms
    return None


tim = 0.5


def main(cam):

    cap = cv2.VideoCapture(cam)
    k = 1
    while True:
        _, frame = cap.read()
        data = getData(frame)
        if data:
            k = 0
            data = np.array(data)
            pred = model.predict(data.reshape(-1,63))[0]
            val = max(model.predict_proba(data.reshape(-1,63))[0])
            if val > 0.8:
                print(val, pred)
                if pred == 'left':
                    pg.press("left")
                    time.sleep(tim)
                elif pred == 'right':
                    pg.press('right')
                    time.sleep(tim)
                elif pred == 'full':
                    pg.press('f5')
                    time.sleep(tim)
                elif pred == 'stop':
                    pg.press("esc")
                    time.sleep(tim)
            else:
                print(pred)
                
        # else:
        #     k = 1
            # print("None")
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
            

    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    cam = int(input("Enter Camera No:"))
    main(cam)