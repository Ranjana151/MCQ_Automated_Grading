import cv2
import numpy as np

def rectContour(contours):
    rectContours = []
    for i in contours:
        area = cv2.contourArea(i)
        #print(area)
        if area > 50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            #print("Corner points", len(approx))
            if len(approx) == 4:
                rectContours.append(i)
    #print(rectContours)
    rectCont = sorted(rectContours, key=cv2.contourArea, reverse=True)
    return rectCont

def getCorner(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    return approx

def reorder(mypoints):
    mypoints = mypoints.reshape((4, 2))
    mypointsNew = np.zeros((4, 1, 2), np.int32)
    add = mypoints.sum(1)
    mypointsNew[0] = mypoints[np.argmin(add)]
    mypointsNew[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints, 1)

    mypointsNew[1] = mypoints[np.argmin(diff)]
    mypointsNew[2] = mypoints[np.argmax(diff)]

    return mypointsNew

def splitbox(img):
    boxes = []
    rows = np.vsplit(img, 5)
    for r in rows:
        col = np.hsplit(r, 5)
        for box in col:
            boxes.append(box)
    return boxes

def displayAnswer(img, answerIndex, grading, rightAnswer, questions, choices):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)

    for x in range(0, questions):
        ans = answerIndex[x]
        cX = (ans*secW) + secW//2
        cY = (x*secH) + secH//2

        if grading[x] == 1:
            myColor = (0, 255, 0)
        else:
            correctAnswer = rightAnswer[x]
            cXRight = (correctAnswer * secW) + secW // 2
            cYRight = (x * secH) + secH // 2
            myColor = (0, 0, 255)
            cv2.circle(img, (cXRight, cYRight), 20, (0, 255, 0), cv2.FILLED)


        cv2.circle(img, (cX, cY), 20, myColor, cv2.FILLED)
    return img


