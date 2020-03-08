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



def recuperate_landmarks(landmarks_head, head_box_head):



    _, _, width_head, height_head = head_box_head

    adding_height = int(height_head * 0.09) #5 de 74
    adding_width  = int(width_head * 0.015)  #1 de 90
    
    area_landmarks1 = (landmarks_head.part(21).x - adding_width,
                       landmarks_head.part(21).y - adding_height)

    area_landmarks2 = (landmarks_head.part(22).x + adding_width,
                       landmarks_head.part(22).y - adding_height)

    area_landmarks3 = (landmarks_head.part(27).x,
                       landmarks_head.part(27).y - adding_height)

    area_landmarks = [area_landmarks1, area_landmarks2, area_landmarks3]

    return area_landmarks






def masks(area_landmarks, threshold, frame_head):


    #Make a box of the region.
    box_crop = cv2.boundingRect(np.array(area_landmarks))
    x ,y, w, h = box_crop

    #cv2.rectangle(frame_head, (x, y), (x+w, y+h), (0, 0, 255), 1)

    crop_threhsold = threshold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)
 
    crop_frame      = frame_head[y:y+h, x:x+w]


    return crop_threhsold, crop_frame, box_crop


def localisation_wrinkle(crop_threhsold, crop_frame, box_crop):

    x ,y, w, h = box_crop

    contours, _ = cv2.findContours(crop_threhsold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
 
    wrinkle = 0
    wrinkle_list = []
    for c in contours:

        max_contour = int((w * h) * 0.5)
        min_contour = int((w * h) * 0.075)
        if min_contour < cv2.contourArea(c) < max_contour:

            xe, ye, we, he = extremums(c)
            largeur  = we[0] - xe[0]
            longueur = he[1] - ye[1]

            if longueur > largeur:
                wrinkle += 1
                wrinkle_list.append((he, ye))
   
    if wrinkle == 2:
        [cv2.line(crop_frame, i[0], i[1], (0, 0, 255), 1) for i in wrinkle_list]


   
def wrinkle_lion(frame_head, landmarks_head, gray, threshold, head_box_head):


    area_landmarks = recuperate_landmarks(landmarks_head, head_box_head)
    crop_threhsold, crop_frame, box_crop = masks(area_landmarks, threshold, frame_head)

    localisation_wrinkle(crop_threhsold, crop_frame, box_crop)

    





