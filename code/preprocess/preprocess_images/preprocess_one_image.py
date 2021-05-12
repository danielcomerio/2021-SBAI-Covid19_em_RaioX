import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_name = 'test21.png'
img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (250, 250))  # Imagem original

img_equalized = cv.equalizeHist(img)  # Imagem original equalizada

img = img.astype(np.float32)

kernel = np.ones((32, 32), np.uint8)
top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)  # Imagem TopHat
black_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)  # Imagem BlackHat

img_transformed = img + top_hat - black_hat  # Imagem transformada

img_transformed -= np.min(img_transformed)
img_transformed /= np.max(img_transformed)
img_transformed *= 255

img_transformed = img_transformed.astype(np.uint8)
# Imagem transformada e equalizada
img_transformed = cv.equalizeHist(img_transformed)


titles = ['Imagem Original', 'Imagem Transformada']
images = [img, img_transformed]

for i in range(len(images)):
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i], 'gray')

    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
