# import cv2
#
# image = cv2.imread('./images/TSHIRT_5.png')
# lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# blurred = cv2.GaussianBlur(lab_image, (5,5), None)
#
# # In inRange command, Anything between those numbers will become white, everything else becomes black.
# # So we take this input image, we have it lower range and upper range.
#
# lower_range_in_lab = (0, 85, 115)
# upper_range_in_lab = (255, 115, 135)
#
# binary = cv2.inRange(blurred, lower_range_in_lab, upper_range_in_lab)
# cv2.imshow('output', binary)
# cv2.waitKey(-1)
#
# cnts, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #print(len(cnts))
# biggest_contour = max(cnts, key=cv2.contourArea)
# biggest_contour_area = cv2.contourArea(biggest_contour)
#
# print('area :', biggest_contour_area)
#
# if biggest_contour_area > 100:
#     print('Found Employee')
#     cv2.putText(image, "Employee Found", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#     cv2.drawContours(image, [biggest_contour], 0, (255,0,0), -1)
# else:
#     print('Employee Not Found')
#
# cv2.imshow('output', image)
# cv2.waitKey(-1)
import cv2
import numpy as np
def employee_finder( image ):

    # image = cv2.imread('./TSHIRT_5.png')
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    blurred = cv2.GaussianBlur(lab_image, (5,5), None)
    #
    # In inRange command, Anything between those numbers will become white, everything else becomes black.
    # So we take this input image, we have it lower range and upper range.100

    lower_range_in_lab = (0, 123, 120)
    upper_range_in_lab = (255, 138, 127)
    M = np.ones((11, 11))
    thresh = cv2.erode(blurred, M, iterations=2)  # erode twice
    thresh = cv2.dilate(thresh, M, iterations=2)
    binary = cv2.inRange(thresh, lower_range_in_lab, upper_range_in_lab)
    # cv2.imshow('output', binary)
    # cv2.waitKey(-1)

    cnts, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(cnts))
    if (len(cnts) == 0):
        cv2.putText(image, "Employee Not Found", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        return image

    biggest_contour = max(cnts, key=cv2.contourArea)
    biggest_contour_area = cv2.contourArea(biggest_contour)

    print('area :', biggest_contour_area)

    if biggest_contour_area > 100000:
        print('Found Employee')
        cv2.putText(image, "Employee Found", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.drawContours(image, [biggest_contour], 0, (255, 0, 0), -1)
    else:
        cv2.putText(image, "Employee Not Found", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        print('Employee Not Found')
    return image
    # cv2.imshow('output', image)
    # cv2.waitKey(-1)
