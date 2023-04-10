import mediapipe as mp
import pandas as pd
import numpy as np
import os
import cv2 

cd = os.getcwd()

mpHands = mp.solutions.hands
hands = mpHands.Hands()

def getData(imgPath):
    img = cv2.imread(imgPath)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = hands.process(imgRGB)
    if output.multi_hand_landmarks:

        data = output.multi_hand_landmarks[0]
        return data.landmark
    return None

def main():
    path = os.path.join(cd,'images')
    file = open('dataset.csv','a')
    for gesture in os.listdir(path):
        # print(gesture)
        gesturePath = path+'/'+gesture
        c = 0
        for img in os.listdir(gesturePath):
            filePath = gesturePath + '/' + img
            lms = getData(filePath)
            if lms:
                for id,lm in enumerate(lms):
                    file.write(str(lm.x))
                    file.write(",")
                    file.write(str(lm.y))
                    file.write(",")
                    file.write(str(lm.z))
                    file.write(",")
                file.write(gesture)
                file.write("\n")
                c += 1
                
        print(c)
    file.close()

if __name__ == "__main__":
    main()