import cv2
import numpy as np
import threading
import math

def make_line(thresh):
    """We make line for detect more than one area
    with border, on eyelashes is paste to the border"""

    cv2.line(thresh, (0, 0), (0, thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0, 0), (thresh.shape[1], 0), (255, 255, 255), 2)
    cv2.line(thresh, (thresh.shape[1], 0), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)
    cv2.line(thresh, (0,  thresh.shape[0]), (thresh.shape[1], thresh.shape[0]), (255, 255, 255), 2)

    return thresh


def extremums(c):

    xe = tuple(c[c[:, :, 0].argmin()][0])  #left
    ye = tuple(c[c[:, :, 1].argmin()][0])  #right
    we = tuple(c[c[:, :, 0].argmax()][0])
    he = tuple(c[c[:, :, 1].argmax()][0])  #bottom

    return xe, ye, we, he

def recuperate_points(landmarks_head, nb_list):
    """Recuperate landmarks coordinates"""

    #Deduce or add pixel for pass the head side contour.
    area_landmarks1 = (landmarks_head.part(nb_list[0]).x + nb_list[3],
                       landmarks_head.part(nb_list[0]).y)

    area_landmarks2 = (landmarks_head.part(nb_list[1]).x + nb_list[5],
                       landmarks_head.part(nb_list[1]).y)

    area_landmarks3 = (landmarks_head.part(nb_list[2]).x + nb_list[6],
                       landmarks_head.part(nb_list[2]).y)

    #Make a list.
    area_landmarks = [area_landmarks1, area_landmarks2, area_landmarks3]

    return area_landmarks


def masks(frame_head, threshold, area_landmarks):
    """Recuperate mask of the thresold region interest"""
    
    #Recuperate convexHull points.
    convexHull = cv2.convexHull(np.array(area_landmarks))
    #cv2.drawContours(frame_head, [convexHull], -1, (255, 0, 0), 1)

    #Create a mask of the convexHull region interest.
    height_frame, width_frame = frame_head.shape[:2]
    black_frame = np.zeros((height_frame, width_frame), np.uint8)
    mask = np.full((height_frame, width_frame), 255, np.uint8)
    cv2.fillPoly(mask, [convexHull], (0, 0, 255))

    #Threshold mask for wrinkles
    mask_threhsold = cv2.bitwise_not(black_frame, threshold.copy(), mask=mask)

    #Make a box of the region.
    box = cv2.boundingRect(convexHull)
    x ,y, w, h = box

    #Make the masks.
    crop_threhsold = mask_threhsold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)

    crop_frame     = frame_head[y:y+h, x:x+w]

    #cv2.rectangle(frame_head, (x, y), (x+w, y+h), (0, 0, 255), 1)

    return crop_threhsold, crop_frame, box


def identification_wrinkle(crop_threhsold, crop_frame, box, head_box_head):
    """Recuperate only contour situate from 2 extemums.
    Recuperate only contour with length comportement."""

    x ,y, w, h = box
    _, _, width_head, _ = head_box_head
    #Find contours of threshold mask.
    contours, _ = cv2.findContours(crop_threhsold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:

        #Constituate the extremums area of contours.
        max_contour = (w * h)* 0.8        #285 de 357
        min_contour = int((w * h) * 0.008) #3 de 465 
        if min_contour < cv2.contourArea(cnt) < max_contour:

            #Recuperate extremums points from the contour.
            xe, ye, we, he = extremums(cnt)
            #Recuperate the width and the length of the contour.
            largeur  = we[0] - xe[0]
            longueur = he[1] - ye[1]

            #Recuperate only length contour and contour > h * 0.62
            #Recuperate only width contour where head > 0.10

            min_length = int(h * 0.26) #10 de 39
            min_width = int(width_head * 0.08) #8 de 100
            if longueur > min_length and largeur>= min_width:
                cv2.line(crop_frame, he, ye, (0, 0, 255), 1)
                #print("ride joie")




def side_one(frame_head, landmarks_head, gray, threshold, head_box_head, nb_list):
    """Side one wrinkle of the mouse"""

    #Recuperate coordinates from landmarks.
    area_landmarks = recuperate_points(landmarks_head, nb_list)
    #Recuperate the region interest.
    crop_threhsold, crop_frame, box = masks(frame_head, threshold, area_landmarks)
    #Identification of the wrinkle.
    identification_wrinkle(crop_threhsold, crop_frame, box, head_box_head)

