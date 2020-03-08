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


def recuperate_landmarks(landmarks_head, list_land, head_box_head):

    _, _, _, height_head = head_box_head


    add_height = int(height_head * 0.1) #8 de 85

    #Deduce or add pixel for pass the head side contour.
    area_landmarks1 = (landmarks_head.part(list_land[0]).x,
                       landmarks_head.part(list_land[1]).y + add_height)

    area_landmarks2 = (landmarks_head.part(list_land[2]).x,
                       landmarks_head.part(list_land[3]).y + add_height)

    area_landmarks3 = (landmarks_head.part(list_land[4]).x,
                       landmarks_head.part(list_land[5]).y + (2 * + add_height))

    area_landmarks4 = (landmarks_head.part(list_land[6]).x,
                       landmarks_head.part(list_land[7]).y + (2 * + add_height))

    area_landmarks = [area_landmarks1, area_landmarks2, area_landmarks3, area_landmarks4]

    box = cv2.boundingRect(np.array(area_landmarks))

    return area_landmarks, box


def masks(box, threshold, frame_head):

    x ,y, w, h = box

    #cv2.imshow("dazqda", threshold)
    #cv2.rectangle(frame_head, (x, y), (x+w, y+h), (0, 0, 255), 1)

    crop_threhsold = threshold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)

    crop_frame     = frame_head[y:y+h, x:x+w]

    return crop_frame, crop_threhsold

def identify_wrinkle(crop_threhsold, crop_frame, box, mode):

    x ,y, w, h = box

    contours, _ = cv2.findContours(crop_threhsold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        max_contour = (w * h)* 0.8        #285 de 357
        min_contour = int((w * h) * 0.05) #23 de 465 


        xe, ye, we, he = extremums(cnt)

        if min_contour < cv2.contourArea(cnt) < max_contour:
            if mode == "left":
                cv2.line(crop_frame, (we[0], ye[1]), (xe[0], he[1]), (0, 255, 0), 1)
            elif mode == "right":
                cv2.line(crop_frame, (xe[0], ye[1]), (we[0], he[1]), (0, 255, 255), 1) 


def under_eyes(frame_head, landmarks_head, gray, threshold, head_box_head, list_land, mode):

    area_landmarks, box = recuperate_landmarks(landmarks_head, list_land, head_box_head)
    crop_frame, crop_threhsold = masks(box, threshold, frame_head)
    identify_wrinkle(crop_threhsold, crop_frame, box, mode)



