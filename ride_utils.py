import cv2
import math
import numpy as np
import threading

#Les boucles qui ralentissent
#Les zones mals définient


def make_line(thresh):
    """We make line for detect more than one area
    with border, on eyelashes is paste to the border"""

    cv2.line(thresh, (0, 0), (0, thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0, 0), (thresh.shape[1], 0), (255, 255, 255), 2)
    cv2.line(thresh, (thresh.shape[1], 0), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0,  thresh.shape[0]), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)

    return thresh


def recuperate_coordinates(points, adding, landmarks, frame, mode):


    if mode == "middle":
        point  = points[-1]
        points = points[:-1]

    area_landmarks = [(landmarks.part(pts[0]).x + add[0],
                       landmarks.part(pts[1]).y + add[1])

                      for pts, add in zip(points, adding)]

    if mode == "middle":
        midle_point = landmarks.part(point[0]).x + landmarks.part(point[1]).x
        midle_point = int(midle_point / 2)
        area_landmarks += [(midle_point, landmarks.part(point[0]).y)]


    convexHull = cv2.convexHull(np.array(area_landmarks))
    #cv2.drawContours(frame, [convexHull], -1, (0, 0, 255), 1)    

    return convexHull


def masks_from_convex(convexPoints, threshold, frame):

    height_frame, width_frame = frame.shape[:2]
    black_frame = np.zeros((height_frame, width_frame), np.uint8)
    mask = np.full((height_frame, width_frame), 255, np.uint8)
    cv2.fillPoly(mask, [convexPoints], (0, 0, 255))

    mask_threhsold = cv2.bitwise_not(black_frame, threshold.copy(), mask=mask)

    box_crop = cv2.boundingRect(convexPoints)
    x ,y, w, h = box_crop

    crop_threhsold = mask_threhsold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)

    crop_frame     = frame[y:y+h, x:x+w]

    return crop_threhsold, crop_frame, box_crop


def masks_from_box(convexPoints, threshold, frame):

    box_crop = cv2.boundingRect(np.array(convexPoints))
    x ,y, w, h = box_crop

    crop_threhsold = threshold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)
 
    crop_frame      = frame[y:y+h, x:x+w]

    return crop_threhsold, crop_frame, box_crop



def extremums(c):

    xe = tuple(c[c[:, :, 0].argmin()][0])  #left
    ye = tuple(c[c[:, :, 1].argmin()][0])  #right
    we = tuple(c[c[:, :, 0].argmax()][0])
    he = tuple(c[c[:, :, 1].argmax()][0])  #bottom

    return xe, ye, we, he


def identification_wrinkle(crop_threhsold, crop_frame, crop_box, head_box,
                           maxContour, minContour, maxLength, minLength):

    x ,y, w, h          = crop_box
    _, _, width_head, _ = head_box

    wrinkle = 0
    wrinkle_list = []

    contours, _ = cv2.findContours(crop_threhsold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:

        xe, ye, we, he = extremums(cnt)

        largeur  = we[0] - xe[0]
        longueur = he[1] - ye[1]

        max_contour = (w * h)* maxContour
        min_contour = int((w * h) * minContour)

        if min_contour < cv2.contourArea(cnt) < max_contour:

            min_length = int(h * minLength)

            if largeur > longueur:
                wrinkle_list.append((we, xe))

            elif longueur > largeur and longueur < min_length:
                wrinkle_list.append((he, ye))

            wrinkle += 1


    if len(wrinkle_list) >= 4:
        [cv2.line(crop_frame, i[0], i[1], (0, 0, 255), 1) for i in wrinkle_list]









































