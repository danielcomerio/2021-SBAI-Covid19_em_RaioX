import numpy as np
import cv2


def emphasizeHeatMapIntensity(img_path, img_type):
    img_array = cv2.imread(img_path)
    emphasized_img = img_array.copy()

    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    if img_type == "points":
        lower = np.array([0, 100, 50])
    else:
        lower = np.array([0, 1, 50])

    upper = np.array([10, 255, 255])
    mask = cv2.inRange(img_array, lower, upper)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(emphasized_img, contours, -1, (0, 0, 255), 1)

    return emphasized_img


image = 'imageExample.png'
image = emphasizeHeatMapIntensity(image, "points")
cv2.imshow('imageExample', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
