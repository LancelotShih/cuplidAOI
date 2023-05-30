# this code is meant to process the cup lids and clip only the relevant sections of the cup
# please adjust the weights of the threshold accordingly

import numpy as np 
import cv2
import os

# imagesG = os.listdir(r'C:\Users\user\Documents\GitHub\cuplidAOI\processedCupImages\goodProc')
imagesB = os.listdir(r'C:\Users\user\Documents\GitHub\cuplidAOI\processedCupImages\badProc')
template = cv2.imread(r"processedCupImages\warningTemplate.png", 0)
h, w = template.shape

count = 0

# for i in imagesG:
#     img = cv2.imread(r"C:\Users\user\Documents\GitHub\cuplidAOI\processedCupImages\goodProc\\" + i)
#     # img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (7,7), 0)

#     (T, threshInv) = cv2.threshold(blurred,130,255, cv2.THRESH_BINARY)
#     cv2.imshow("detected?", threshInv)
#     masked = cv2.bitwise_and(img, img, mask = threshInv)
#     cv2.imshow("clipped", masked)
#     # status = cv2.imwrite("cupInspectAOI\goodProc\procimg" + str(count) + ".png", masked)
#     # print("written to disk", status)
    

#     # template match the warning
#     print("Finding label...")
#     img2 = gray.copy()
#     result = cv2.matchTemplate(img2, template, cv2.TM_CCORR_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     location = max_loc
#     bottomRight = (location[0] + w, location[1] + h)
#     print(location)
#     img2 = img2[location[1]: location[1] + h , location[0]: location[0] + w]
#     cv2.rectangle(img2, location, bottomRight, 255, 5)
#     cv2.imshow('Match', img2)
#     cv2.imwrite("processedCupImages\goodCrop\gCrop" + str(count) + ".png", img2)
#     count+=1
#     cv2.waitKey(0)

for i in imagesB:
    img = cv2.imread(r"C:\Users\user\Documents\GitHub\cuplidAOI\processedCupImages\badProc\\" + i)
    # img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)

    (T, threshInv) = cv2.threshold(blurred,130,255, cv2.THRESH_BINARY)
    cv2.imshow("detected?", threshInv)
    masked = cv2.bitwise_and(img, img, mask = threshInv)
    cv2.imshow("clipped", masked)
    status = cv2.imwrite(r"processedCupImages\badCrop\bCrop" + str(count) + ".png", masked)
    print("written to disk", status)

# template matching 
    print("Finding label...")
    img2 = gray.copy()
    result = cv2.matchTemplate(img2, template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    location = max_loc
    bottomRight = (location[0] + w, location[1] + h)
    print(location)
    cv2.rectangle(img2, location, bottomRight, 255, 5)
    cv2.imshow('Match', img2)
    img2 = img2[location[1]: location[1] + h , location[0]: location[0] + w]
    cv2.rectangle(img2, location, bottomRight, 255, 5)
    cv2.imshow('Match', img2)
    cv2.imwrite(r"processedCupImages\badCrop\bCrop" + str(count) + ".png", img2)

    count+=1
    cv2.waitKey(0)