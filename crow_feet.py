import cv2
import numpy as np
import threading
import math

def extremums(c):

    xe = tuple(c[c[:, :, 0].argmin()][0])  #left
    ye = tuple(c[c[:, :, 1].argmin()][0])  #right
    we = tuple(c[c[:, :, 0].argmax()][0])
    he = tuple(c[c[:, :, 1].argmax()][0])  #bottom

    return xe, ye, we, he


def make_line(thresh):
    """We make line for detect more than one area
    with border, on eyelashes is paste to the border"""

    cv2.line(thresh, (0, 0), (0, thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0, 0), (thresh.shape[1], 0), (255, 255, 255), 2)
    cv2.line(thresh, (thresh.shape[1], 0), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0,  thresh.shape[0]), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)

    return thresh



def recuperate_points(landmarks_head, list_land):


    area_landmarks1 = (landmarks_head.part(list_land[0]).x, landmarks_head.part(list_land[2]).y)
    area_landmarks2 = (landmarks_head.part(list_land[1]).x, landmarks_head.part(list_land[2]).y)
    area_landmarks3 = (int((landmarks_head.part(list_land[3]).x + landmarks_head.part(list_land[1]).x) / 2),
                           landmarks_head.part(list_land[3]).y)

    area_landmarks4 = (landmarks_head.part(list_land[4]).x, landmarks_head.part(list_land[5]).y)

    area_landmarks = [area_landmarks1, area_landmarks2, area_landmarks4, area_landmarks3]



    return area_landmarks


def masks(area_landmarks, threshold, frame_head):

    convexHull = cv2.convexHull(np.array(area_landmarks))
    #cv2.drawContours(frame_head, [convexHull], -1, (255, 0, 0), 1)

    height_frame, width_frame = frame_head.shape[:2]
    black_frame = np.zeros((height_frame, width_frame), np.uint8)
    mask = np.full((height_frame, width_frame), 255, np.uint8)
    cv2.fillPoly(mask, [convexHull], (0, 0, 255))

    mask_threhsold = cv2.bitwise_not(black_frame, threshold.copy(), mask=mask)


    box = cv2.boundingRect(convexHull)
    x ,y, w, h = box

    crop_threhsold = mask_threhsold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)

    crop_frame     = frame_head[y:y+h, x:x+w]

    return crop_frame, crop_threhsold, box


def identify_wrinkle(crop_threhsold, crop_frame, box):

    x ,y, w, h = box

    wrinkle = 0
    wrinkle_list = []
    contours, _ = cv2.findContours(crop_threhsold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:

        xe, ye, we, he = extremums(cnt)

        largeur  = we[0] - xe[0]
        longueur = he[1] - ye[1]

        max_contour = (w * h)* 0.8 #285 de 357
        min_contour = int((w * h) * 0.003) #1 de 465

        if min_contour < cv2.contourArea(cnt) < max_contour:

            min_length = int(h * 0.28) #10 de 37

            if largeur > longueur:
                wrinkle_list.append((we, xe))

            elif longueur > largeur and longueur < min_length:
                wrinkle_list.append((he, ye))

            wrinkle += 1


    if len(wrinkle_list) >= 4:
        [cv2.line(crop_frame, i[0], i[1], (0, 0, 255), 1) for i in wrinkle_list]
        #print("sourire / rire / frustration et de nervosité")


def crow_feet(frame_head, landmarks_head, gray, threshold, head_box_head, list_land):

    area_landmarks = recuperate_points(landmarks_head, list_land)
    crop_frame, crop_threhsold, box = masks(area_landmarks, threshold, frame_head)
    identify_wrinkle(crop_threhsold, crop_frame, box)
