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


def recuperate_coordinate(crop_color_skin, extremums):
    """Recuperate from extremums contours the color
    from the picture"""

    xe, ye, we, he = extremums

    right = crop_color_skin[xe[1], xe[0]] #right coord
    left  = crop_color_skin[ye[1], ye[0]] #left coord

    left_bot = crop_color_skin[ye[1], xe[0]] #left bot coord
    left_top = crop_color_skin[he[1], xe[0]] #left top coord

    right_bot = crop_color_skin[ye[1], we[0]] #right bot coord
    right_top = crop_color_skin[he[1], we[0]] #right top coord

    top_right = crop_color_skin[ye[1], xe[0]] #right bot coord
    top_left  = crop_color_skin[ye[1], we[0]] #right top coord

    bot_right = crop_color_skin[he[1], xe[0]] #right bot coord
    bot_left  = crop_color_skin[he[1], we[0]] #right top coord

    return (right, left, left_bot, left_top, right_bot, right_top,
            top_right, top_left, bot_right, bot_left)


def no_skin_color(c, crop_color_skin):

    xe, ye, we, he = extremums(c)

    #Recuperate extremums contours colors from the skin picture.
    coordinates = recuperate_coordinate(crop_color_skin, (xe, ye, we, he))
    right, left, left_bot, left_top, right_bot, right_top,\
            top_right, top_left, bot_right, bot_left = coordinates

    black = False
    #If extremums left and right touch black pixel (255, 255, 255)
    #isn't skin pixels.
    #Verify if sides touch black pixels. (no skin pixels)
    verification = lambda i: True if i[0] == 0 and i[1] == 0 and i[2] == 0 else False

    #right left extremums.
    contour_touch_black1 = [verification(i) for i in [right, left]]
    contour_touch_black2 = [verification(i) for i in [left_bot, left_top]]
    contour_touch_black3 = [verification(i) for i in [right_bot, right_top]]
    contour_touch_black4 = [verification(i) for i in [top_right, top_left]]
    contour_touch_black5 = [verification(i) for i in [bot_right, bot_left]]     #A ALLEGER

    #Break and return True if the 2 px touchs black pixel.
    for i in [contour_touch_black1, contour_touch_black2, contour_touch_black3,
              contour_touch_black4, contour_touch_black5]:
        if i.count(True) == 2:
            black = True
            break

    return black, (xe, ye, we, he)



def recuperate_forehead_area(landmarks_head1, head_box_head1):
    """We recuperate landmarks from 81 points dlib (forehead region)"""

    #17 - 26 = on eyes
    #69 - 80 = forehead https://github.com/codeniko/shape_predictor_81_face_landmarks
    landmarks_forehead = [75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74]
    landmarks_on_eyes  = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17]

    #Recuperate coordinates
    landmarks_forehead = [(landmarks_head1.part(n).x, landmarks_head1.part(n).y) for n in landmarks_forehead]

    _, _, _, h = head_box_head1
    landmarks_on_eyes  = [(landmarks_head1.part(n).x, landmarks_head1.part(n).y - int( (5 * h) / 100))
                          for n in landmarks_on_eyes]

    landmarks_on_eyes += landmarks_forehead

    #Recuperate points into a matrice
    landmarks_on_eyes = np.array(landmarks_on_eyes)

    return landmarks_on_eyes


def recuperate_forehead_mask(points, gray, height_frame, width_frame,
                             threshold, frame_skin, frame, blur_frame):
    """Here we recuperate the region from the picture"""

    black_frame = np.zeros((height_frame, width_frame), np.uint8)
    mask = np.full((height_frame, width_frame), 255, np.uint8)
    cv2.fillPoly(mask, [points], (0, 0, 255))

    #Skin mask (raise hair)
    gray_skin = cv2.cvtColor(frame_skin, cv2.COLOR_BGR2GRAY)
    mask_skin = cv2.bitwise_not(black_frame, gray_skin.copy(), mask=mask)

    #Threshold mask for wrinkles
    mask_threhsold = cv2.bitwise_not(black_frame, threshold.copy(), mask=mask) 

    #Recuperate region interest.
    box = cv2.boundingRect(points)
    x ,y, w, h = box

    crop_skin      = frame_skin[y:y+h, x:x+w]

    crop_threhsold = mask_threhsold[y:y+h, x:x+w]
    crop_threhsold = make_line(crop_threhsold)

    crop_frame     = frame[y:y+h, x:x+w]
    crop_blur      = blur_frame[y:y+h, x:x+w]

    #Put white pixel into none skin pixel (hair) on threshold
    for j in range(crop_skin.shape[1]):
        for i in range(crop_skin.shape[0]):
            if crop_skin[i, j][0] == 0 and\
               crop_skin[i, j][1] == 0 and\
               crop_skin[i, j][2] == 0:
                crop_threhsold[i, j] = 255   #A ALLEGER

    return (crop_skin, crop_threhsold, crop_frame, crop_blur)


def hair_color(crop_blur, contour): #a verifier
    """Recuperate color of a contour different of a skin color, who could be hair"""

    #Hair color.
    noir = 0; marron = 0; blond = 0; no = 0; out = ""

    #Dimension of the crop.
    height, width = crop_blur.shape[:2]

    #Recuperate color of the contour.
    couleur_hair = [crop_blur[j, i] for j in range(height) for i in range(width)]

    #Condition of color pixels.
    for nb, couleur in enumerate(couleur_hair): #A ALLEGER

        if couleur[1] + 10 < couleur[0] > couleur[2] + 10 and 50 < couleur[1] < 130:
            marron += 1

        elif couleur[0] < 50 and couleur[1] < 50 and couleur[2] < 50:
            noir += 1

        elif couleur[0] > 90 and couleur[1] > 90 and couleur[2] > 90 and\
           couleur[0] >= couleur[1] + 10 and couleur[1] >= couleur[2] + 10:
            blond += 1

    #Sort our dictionnary and recuperate the highter number.
    dico_color = {"noir":noir, "marron":marron, "blond":blond, "no":no}
    dico_color = sorted(dico_color.items(), key=lambda t: t[1])

    #If we have more 50 % of black.
    if (dico_color[-1][1] * nb) / 100 > 50:
        out = dico_color[-1][0]

    return out



def frange_side(extrems, head_box_head1):
    """frange side = hair parting side"""

    #width of the head box
    _, _, width_head, _ = head_box_head1
    #Extremums of the contours.
    xe, ye, we, he = extrems

    #Middle of the face.
    mid_face = int(width_head / 2)
    #We want a wick hair (with a length).
    length_contour = int(we[0] - xe[0]) > width_head * 0.4 #30 de 77

    situation_meche = ""
    #Search the higther side (left or right).
    if ye[0] > mid_face:  situation_meche = "meche droite donc raie gauche"
    else:                 situation_meche = "meche gauche donc raie droite"

    return situation_meche




def wrinkle_draw(cnt, mask_frame, landrmarks_forehead):

    #Recuperate extremums points of the contour.
    xe, ye, we, he = extremums(cnt)
    convexHull = cv2.convexHull(np.array(landrmarks_forehead))
    box_forehead = cv2.boundingRect(convexHull)
    _, _, w_forehead, _ = box_forehead

    
    """Define the direction of the contour"""
    #Recuperate dimensions of the contour.
    largeur  = we[0] - xe[0]
    longueur = he[1] - ye[1]

    #Recuperate width wrinkle comportement.
    min_width = largeur > w_forehead * 0.115 #10 de 87
    if largeur > longueur and min_width == True:
        return [we, xe]


    return False


def identification_wrinkles(frame, mask_skin, mask_threhsold,
                            mask_frame, mask_blur, head_box_head1,
                            landrmarks_forehead):
    """Recuperate contours of the thresold crop, and treat them for know
    if we have wick or wrinkles of the forehead"""

    #Count area region of the forehead
    area_forehead = cv2.contourArea(landrmarks_forehead)

    height, width = mask_threhsold.shape[:2]
    contours, _ = cv2.findContours(mask_threhsold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    wrinkle_list = []
    for cnt in contours:

        #Conditions for keep the contour. (max and min length).
        max_contour = cv2.contourArea(cnt) < (height * width) * 0.9 #5220 de 5800
        min_contour = cv2.contourArea(cnt) > int( (height * width) * 0.0039 ) #16 de 4140


        if max_contour is True and min_contour is True:

            #Not a skin color
            black, extrems = no_skin_color(cnt, mask_skin)
            if black is True:
                color = hair_color(mask_blur, cnt)

                #Could be hair, verify colors
                if color is not None:
                    situation_meche = frange_side(extrems, head_box_head1)
                    if situation_meche is not "":
                        cv2.drawContours(mask_frame, [cnt], -1, (255, 0, 0), 1)

            #Skin color
            elif black is False:

                presence = wrinkle_draw(cnt, mask_frame, landrmarks_forehead)
                if presence is not False:
                    wrinkle_list.append(presence)

    if len(wrinkle_list) > 1:
        #print(len(wrinkle_list))
        #print("frustration, colere, stresse, reflexion")
        [cv2.line(mask_frame, i[0], i[1], (0, 0, 255), 1) for i in wrinkle_list]




def front(frame_head, landmarks_head, landmarks_head1, gray,
          threshold, frame_skin, frame_blur, head_box_head1):

    #Recuperate forehead region from dlib model.
    landrmarks_forehead = recuperate_forehead_area(landmarks_head1, head_box_head1)
    #cv2.drawContours(frame_head, [landrmarks_forehead], -1, (0, 0, 255), 1)


    #Recuperate dimensions of the picture.
    height_frame, width_frame = frame_head.shape[:2]

    #Recuperate mask interest of the picture.
    masks = recuperate_forehead_mask(landrmarks_forehead, gray, height_frame,
                                     width_frame, threshold, frame_skin, frame_head, frame_blur)

    mask_skin, mask_threhsold, crop_frame, crop_blur = masks
    #cv2.imshow("mask_threhsold", mask_threhsold)
    #cv2.imshow("mask_skin", mask_skin)


    identification_wrinkles(frame_head, mask_skin, mask_threhsold,
                            crop_frame, crop_blur, head_box_head1,
                            landrmarks_forehead)


