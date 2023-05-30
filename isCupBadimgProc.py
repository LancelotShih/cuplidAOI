# this code is meant to process the cup lids and clip only the relevant sections of the cup
# please adjust the weights of the threshold accordingly

import numpy as np 
import cv2
import os

imagesG = os.listdir(r'cupInspectAOI\good')
# imagesB = os.listdir(r'cupInspectAOI\bad')

count = 0

for i in imagesG:
    img = cv2.imread(r"cupInspectAOI\good\\" + i)
    img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)

    (T, threshInv) = cv2.threshold(blurred,130,255, cv2.THRESH_BINARY)
    cv2.imshow("detected?", threshInv)
    masked = cv2.bitwise_and(img, img, mask = threshInv)
    cv2.imshow("clipped", masked)
    status = cv2.imwrite("cupInspectAOI\goodProc\procimg" + str(count) + ".png", masked)
    print("written to disk", status)
    count+=1
    cv2.waitKey(0)

# for i in imagesB:
#     img = cv2.imread(r"cupInspectAOI\bad\\" + i)
#     img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (7,7), 0)

#     (T, threshInv) = cv2.threshold(blurred,130,255, cv2.THRESH_BINARY)
#     cv2.imshow("detected?", threshInv)
#     masked = cv2.bitwise_and(img, img, mask = threshInv)
#     cv2.imshow("clipped", masked)
#     status = cv2.imwrite(r"cupInspectAOI\badProc\procimg" + str(count) + ".png", masked)
#     print("written to disk", status)
#     count+=1
#     cv2.waitKey(0)