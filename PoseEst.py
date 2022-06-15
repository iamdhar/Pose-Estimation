import cv2
import mediapipe as mp
import time
import numpy as np

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)
pTime = 0
while True:
    success, img = cap.read()
    flipped = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(flipped,results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = flipped.shape()

            cx, cy = int(lm.x * w) , int(lm.y * h)
            cv2.circle(flipped, cx, cy, 5, (255,0,0), cv2.FILLED)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(flipped, str(int(fps)), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                         (255,0,0), 3)

    cv2.imshow("Flipped Image", flipped)
    cv2.waitKey(1)
