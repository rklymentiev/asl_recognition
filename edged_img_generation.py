# SCRIPT TO GENERATE EDGED IMAGES FROM THE WEBCAM

import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)
cv2.namedWindow('edged')

def nothing(x):
    pass

def get_class_label(val, dictionary):
    """
    Function returns the key (Letter: a/b/c/...) value from the alphabet dictionary
    based on its class index (1/2/3/...)
    """
    for key, value in dictionary.items():
        if value == val:
            return key

alphabet = {chr(i+96).upper():i for i in range(1,27)}
alphabet['del'] = 27
alphabet['nothing'] = 28
alphabet['space'] = 29

cv2.createTrackbar('lower_threshold', 'edged', 0, 255, nothing)
cv2.createTrackbar('upper_threshold', 'edged', 0, 255, nothing)
cv2.setTrackbarPos('lower_threshold', 'edged', 100)
cv2.setTrackbarPos('upper_threshold', 'edged', 0)

index = 29
current_letter = get_class_label(index, alphabet)
try:
    os.mkdir('my_dataset_edges/all/' + str(current_letter) + '/')
except:
    pass
path = 'my_dataset_edges/all/' + str(current_letter) + '/'

i = 0
while(1):
    ret, frame = cap.read()
    x_0 = int(frame.shape[1] * 0.1)
    y_0 = int(frame.shape[0] * 0.25)
    x_1 = int(x_0 + 200)
    y_1 = int(y_0 + 200)


    hand = frame.copy()[y_0:y_1, x_0:x_1]
    gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (5, 5), 0)
    blured = cv2.erode(blured, None, iterations=2)
    blured = cv2.dilate(blured, None, iterations=2)

    lower = cv2.getTrackbarPos('lower_threshold', 'edged')
    upper = cv2.getTrackbarPos('upper_threshold', 'edged')
    edged = cv2.Canny(blured,lower,upper)

    cv2.imshow('frame',frame)
    cv2.imshow('edged',edged)

    if i > 50:
        cv2.imwrite(
            path + str(i) + '.jpg',
            edged
        )
    i += 1

    if i > 1200:
        break

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
