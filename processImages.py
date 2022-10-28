import mediapipe as mp
import pandas as pd
import numpy as np
import os
import cv2 

cd = os.getcwd()

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True,max_num_hands=1, min_detection_confidence=0.5)

def getData(imgPath):
    img = cv2.imread(imgPath)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = hands.process(imgRGB)

    data = output.multi_hand_landmarks[0]
    return data.landmark
    

def storeData():
    path = os.path.join(cd,'images')
    file = open('dataset.csv','a')
    for gesture in os.listdir(path):
        print(gesture)
        gesturePath = path+'/'+gesture
        for img in os.listdir(gesturePath):
            filePath = gesturePath + '/' + img
            lms = getData(filePath)
            for id,lm in enumerate(lms):
                file.write(str(lm.x))
                file.write(",")
                file.write(str(lm.y))
                file.write(",")
                file.write(str(lm.z))
                file.write(",")
            file.write(gesture)
            file.write("\n")
            print(img)
    file.close()

if __name__ == "__main__":
    storeData()