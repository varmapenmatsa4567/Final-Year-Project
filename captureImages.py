import numpy as np
import cv2
import pandas as pd
import os 
import mediapipe as mp

# To get Current Directory
cd = os.getcwd()

# Initialization for Hands Module
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# To check whether a hand is present or not in image
def isHand(frame):
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output = hands.process(imgRGB)
    if output.multi_hand_landmarks:
        return True
    return False

# To Store images with hands
def storeImages(gesture="hello", cam=1):
    # To store images in specific gesture folder
    dataset = os.path.join(cd,'images')
    classFolder = os.path.join(dataset,gesture)
    os.mkdir(classFolder)

    # Reading webcam feed
    cap = cv2.VideoCapture(cam)
    i = 0
    while True:
        _, frame = cap.read()
        i += 1
        if i%3==0:
            if isHand(frame):
                print(i//3)
                cv2.imwrite(classFolder+'/'+str(i//3)+'.png',frame)
            else:
                i -= 3
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) == ord('q') or i > 600:
            break
    cap.release()
    cv2.destroyAllWindows()

# Initialize the camera and gesture for image Capturing
def main():
    cam = int(input("Enter Camera No:"))
    n = int(input("Enter no of Gestures:"))
    for i in range(n):
        gesture = input("Enter gesture name:")
        storeImages(gesture, cam)

# main()

if __name__ == "__main__":
    main()