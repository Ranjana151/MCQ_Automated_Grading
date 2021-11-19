import cv2
import numpy as np
import utils
#import pytesseract
wImg = 700
hImg = 700
no_question = 5
no_choices = 5
count = 0
rightAnswerIndex = [2, 3, 1, 0, 4]

# When want to save name from mcq image
"""pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

img = cv2.imread("ocr.png")

data = pytesseract.image_to_string(img)
##print(data)
need_data = data.split('\n')[1:]
#print(need_data)
name = ""
for i in need_data:
    if i.startswith('Name'):
        name = i
#print(name)
Names = name.split(":")[1]
#print(Names)"""


img = cv2.imread("1.jpg")
imgResize = cv2.resize(img, (250, 350))
imgFinal = imgResize.copy()
imgFinalGrade = imgResize.copy()
imgContours = imgResize.copy()
imgBiggestRect = imgResize.copy()
imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur, 50, 100)

# Finding all Contours

contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)

# Finding rectangle having biggest area and second biggest  area

retContours = utils.rectContour(contours)

# Finding Biggest Rectangle , and second biggest rectangle corner point

biggestRect = utils.getCorner(retContours[0])
gradeRect = utils.getCorner(retContours[1])
#print(gradeRect)
#print(biggestRect, gradeRect)

# Draw biggest rectangle and second biggest corner points
if biggestRect.size != 0 and gradeRect.size != 0:
    cv2.drawContours(imgBiggestRect, biggestRect, -1, (255, 0, 0), 10)
    cv2.drawContours(imgBiggestRect, gradeRect, -1, (255, 0, 0), 10)

#print(biggestRect.shape)
    biggestContour = utils.reorder(biggestRect)
    gradePoints = utils.reorder(gradeRect)

    #Warp perspective for biggest Rectangle

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [250, 0], [0, 350], [250, 350]])

    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgScanAnswer = cv2.warpPerspective(imgResize, matrix, (250, 350))
    #cv2.imshow("Image Scanned", imgScanAnswer)

    # Warp perspective for second biggest Rectangle
    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [250, 0], [0, 350], [250, 350]])

    matrix = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgScanGrade = cv2.warpPerspective(imgResize, matrix, (250, 350))

    #Applying Threshold
    imgScanGradeGray = cv2.cvtColor(imgScanAnswer, cv2.COLOR_BGR2GRAY)
    imgScanGradeThresh = cv2.threshold(imgScanGradeGray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("Threshold Image", imgScanGradeThresh)
    #cv2.imshow(" Grade Image Scanned", imgScanGrade)

    boxes = utils.splitbox(imgScanGradeThresh)

    # Getting marking boxes

    myBoxPixel = np.zeros((no_question, no_choices))
    r = 0
    c = 0
    PixelValue = []
    for i in boxes:
        boxPixel = cv2.countNonZero(i)
        myBoxPixel[r][c] =boxPixel
        c += 1
        if c == no_choices:
            r += 1
            c = 0

    #print(myBoxPixel)

    # Index of answer by user

    answerIndex = []

    for x in range(0, no_question):
        arr = myBoxPixel[x]
        indexValue = np.where(arr == np.amax(arr))
        answerIndex.append(indexValue[0][0])
    #print(answerIndex)

    # Calculating score
    grading = []

    for s in range(0, no_question):
        if answerIndex[s] == rightAnswerIndex[s]:
            grading.append(1)
        else:
            grading.append(0)
    #print(grading)
    score = int((sum(grading)/no_question)*100)
    score = str(score)
    #print(score)

# Display answer in an image
    imgResult = imgScanAnswer.copy()
    imgResult = utils.displayAnswer(imgResult, answerIndex, grading, rightAnswerIndex,
                                    no_question, no_choices)
    #cv2.imshow("Answer Image", imgResult)

    imgRawDrawing = np.zeros_like(imgScanAnswer)
    imgRawDrawing = utils.displayAnswer(imgRawDrawing, answerIndex, grading, rightAnswerIndex,
                                        no_question, no_choices)
    #cv2.imshow("Black", imgRawDrawing)
    inverseMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWrap = cv2.warpPerspective(imgRawDrawing, inverseMatrix, (250, 350))

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWrap, 1, 0)
    #cv2.imshow("Final Image", imgFinal)


    # for grade Image
    imgGradeScan = imgScanGrade.copy()
    imgRawDrawingG = np.zeros_like(imgGradeScan)
    cv2.putText(imgRawDrawingG, str(int(score))+"%", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255),
                8)
    #cv2.imshow("Final Image", imgRawDrawingG)


    inverseMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
    imgInvWrapG = cv2.warpPerspective(imgRawDrawingG, inverseMatrixG, (250, 350))

    imgFinalG = cv2.addWeighted(imgFinal, 1, imgInvWrapG, 1, 0)
    cv2.imshow("Result", imgFinalG)
    # Save image when s key is pressed
    if cv2.waitKey(0) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", imgFinalG)
        cv2.waitKey(300)
        count += 1
    #Saving grade with name
    """with open("grade.csv", "r+") as f:
        f.writelines(f'/n{Names},{score}')"""
cv2.imshow("Original Image", imgResize)
cv2.imshow("Corner point Image", imgBiggestRect)
cv2.imshow("Edge Image", imgCanny)
cv2.imshow("Contour Image", imgContours)

cv2.imshow("Blur Image", imgBlur)
cv2.waitKey(0)



