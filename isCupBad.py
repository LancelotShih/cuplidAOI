import numpy as np
import cv2
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('ggplot')

capture = cv2.VideoCapture(1)
template = cv2.imread('cupInspectAOI\WIN_20230526_14_05_28_Pro.jpg', 0)
template = cv2.resize(template, (0,0), fx = 0.5, fy = 0.5)
h,w = template.shape[:2]

# camera set up for pretty image
capture.set(cv2.CAP_PROP_AUTOFOCUS,255) # autofocuses camera to give better view
capture.set(cv2.CAP_PROP_FRAME_WIDTH ,1920) # sets the width of image
capture.set(cv2.CAP_PROP_FRAME_HEIGHT ,1200) # sets the height of image

while True:
    ret, frame = capture.read()
    framecpy = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    crop = cv2.matchTemplate(frame, template, cv2.TM_CCORR)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(crop)
    location = max_loc

    bottom_right = (location[0] + w, location[1] + h)
    cv2.rectangle(framecpy, location, bottom_right, 255, 5)

    cv2.imshow("feed", framecpy)
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
