import numpy as np
import cv2
import urllib.request

# URL = "http://192.168.0.101:8080/shot.jpg"
video = cv2.VideoCapture(0)

while True:
    # imgResp = urllib.request.urlopen(URL)
    # imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

    # img = cv2.imdecode(imgNp, -1)
    ret, img = video.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (640, 480))

    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([90, 60, 0])
    upper_blue = np.array([121, 255, 255])
    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # define range for red color in HSV
    lower_red = np.array([0, 50, 120])
    upper_red = np.array([10, 255, 255])
    # Threshold the HSV image to get only blue colors
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # define range for green color in HSV
    lower_green = np.array([40, 70, 80])
    upper_green = np.array([70, 255, 255])
    # Threshold the HSV image to get only blue colors
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # define range for yellow color in HSV
    lower_yellow = np.array([25, 70, 120])
    upper_yellow = np.array([30, 255, 255])
    # Threshold the HSV image to get only blue colors
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # define range for white color in HSV
    # lower_white = np.array([0, 0, 150])
    # upper_white = np.array([180, 50, 255])
    # Threshold the HSV image to get only blue colors
    # white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # define range for black color in HSV
    # lower_black = np.array([0, 0, 0])
    # upper_black = np.array([180, 255, 50])
    # Threshold the HSV image to get only blue colors
    # black_mask = cv2.inRange(hsv_frame, lower_black, upper_black)

    # define range for orange color in HSV
    # lower_orange = np.array([5, 50, 50])
    # upper_orange = np.array([15, 255, 255])
    # Threshold the HSV image to get only blue colors
    # orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    # define range for pink color in HSV
    # lower_pink = np.array([160, 50, 50])
    # upper_pink = np.array([180, 255, 255])
    # Threshold the HSV image to get only blue colors
    # pink_mask = cv2.inRange(hsv_frame, lower_pink, upper_pink)

    # define range for purple color in HSV
    # lower_purple = np.array([140, 50, 50])
    # upper_purple = np.array([160, 255, 255])
    # Threshold the HSV image to get only blue colors
    # purple_mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)

    # define range for violet color in HSV
    # lower_violet = np.array([140, 50, 50])
    # upper_violet = np.array([160, 255, 255])
    # Threshold the HSV image to get only blue colors
    # violet_mask = cv2.inRange(hsv_frame, lower_violet, upper_violet)

    # define range for gray color in HSV
    # lower_gray = np.array([0, 0, 0])
    # upper_gray = np.array([180, 255, 50])
    # Threshold the HSV image to get only blue colors
    # gray_mask = cv2.inRange(hsv_frame, lower_gray, upper_gray)

    # Morphological Transform, Dilation for each color and bitwise_and operator between imageFrame and mask determines to detect only that particular color
    kernel = np.ones((5, 5), "uint8")

    # for red color
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(img, img, mask=red_mask)

    # for blue color
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(img, img, mask=blue_mask)

    # for green color
    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(img, img, mask=green_mask)

    # for yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    res_yellow = cv2.bitwise_and(img, img, mask=yellow_mask)

    # for white color
    # white_mask = cv2.dilate(white_mask, kernel)
    # res_white = cv2.bitwise_and(img, img, mask=white_mask)

    # for black color
    # black_mask = cv2.dilate(black_mask, kernel)
    # res_black = cv2.bitwise_and(img, img, mask=black_mask)

    # for orange color
    # orange_mask = cv2.dilate(orange_mask, kernel)
    # res_orange = cv2.bitwise_and(img, img, mask=orange_mask)

    # for pink color
    # pink_mask = cv2.dilate(pink_mask, kernel)
    # res_pink = cv2.bitwise_and(img, img, mask=pink_mask)

    # for purple color
    # purple_mask = cv2.dilate(purple_mask, kernel)
    # res_purple = cv2.bitwise_and(img, img, mask=purple_mask)

    # for violet color
    # violet_mask = cv2.dilate(violet_mask, kernel)
    # res_violet = cv2.bitwise_and(img, img, mask=violet_mask)

    # for gray color
    # gray_mask = cv2.dilate(gray_mask, kernel)
    # res_gray = cv2.bitwise_and(img, img, mask=gray_mask)

    MAX_AREA = 500
    # creating a contour to track red color
    red_contours, red_hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(red_contours):
        area = cv2.contourArea(contour)
        if (area > MAX_AREA):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "RED", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

    # creating a contour to track blue color
    blue_contours, blue_hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(blue_contours):
        area = cv2.contourArea(contour)
        if (area > MAX_AREA):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "BLUE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))

    # creating a contour to track green color
    green_contours, green_hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(green_contours):
        area = cv2.contourArea(contour)
        if (area > MAX_AREA):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "GREEN", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

    # creating a contour to track yellow color
    yellow_contours, yellow_hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(yellow_contours):
        area = cv2.contourArea(contour)
        if (area > MAX_AREA):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, "YELLOW", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))

    # creating a contour to track white color
    # white_contours, white_hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for pic, contour in enumerate(white_contours):
    #     area = cv2.contourArea(contour)
    #     if (area > MAX_AREA):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    #         cv2.putText(img, "WHITE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

    # creating a contour to track black color
    # black_contours, black_hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for pic, contour in enumerate(black_contours):
    #     area = cv2.contourArea(contour)
    #     if (area > MAX_AREA):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    #         cv2.putText(img, "BLACK", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))

    # creating a contour to track orange color
    # orange_contours, orange_hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for pic, contour in enumerate(orange_contours):
    #     area = cv2.contourArea(contour)
    #     if (area > MAX_AREA):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
    #         cv2.putText(img, "ORANGE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255))

    # creating a contour to track pink color
    # pink_contours, pink_hierarchy = cv2.findContours(pink_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for pic, contour in enumerate(pink_contours):
    #     area = cv2.contourArea(contour)
    #     if (area > MAX_AREA):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 105, 180), 2)
    #         cv2.putText(img, "PINK", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 105, 180))

    # creating a contour to track purple color
    # purple_contours, purple_hierarchy = cv2.findContours(purple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for pic, contour in enumerate(purple_contours):
    #     area = cv2.contourArea(contour)
    #     if (area > MAX_AREA):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 128), 2)
    #         cv2.putText(img, "PURPLE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128))

    # creating a contour to track violet color
    # violet_contours, violet_hierarchy = cv2.findContours(violet_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for pic, contour in enumerate(violet_contours):
    #     area = cv2.contourArea(contour)
    #     if (area > MAX_AREA):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (148, 0, 211), 2)
    #         cv2.putText(img, "VIOLET", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (148, 0, 211))

    # creating a contour to track gray color
    # gray_contours, gray_hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for pic, contour in enumerate(gray_contours):
    #     area = cv2.contourArea(contour)
    #     if (area > MAX_AREA):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (127, 127, 127), 2)
    #         cv2.putText(img, "GRAY", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 127, 127))

    # show image window
    cv2.imshow("Color Tracking", img)

    # terminate program by pressing escape key
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        video.release()
        cv2.destroyAllWindows()
        break
