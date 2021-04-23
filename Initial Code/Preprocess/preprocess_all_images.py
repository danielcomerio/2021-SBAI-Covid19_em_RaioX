import os

import cv2 as cv
import numpy as np


def preprocessImg(img):
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

    img_transformed = img_transformed.astype(np.uint8) # Imagem transformada e equalizada
    img_transformed = cv.equalizeHist(img_transformed)

    return img_transformed


def main():
    PATH_BASE = "C:\\Users\\danie\\Desktop\\Artigo-Daniel\\DATA_SET"
    PATH_TRAIN_COVID_OLD = os.path.join(PATH_BASE, "DONE_DATA", "train", "COVID-19")
    PATH_TRAIN_NORMAL_OLD = os.path.join(PATH_BASE, "DONE_DATA", "train", "NORMAL")
    PATH_TRAIN_PNEUMONIA_OLD = os.path.join(PATH_BASE, "DONE_DATA", "train", "Pneumonia")
    PATH_TEST_COVID_OLD = os.path.join(PATH_BASE, "DONE_DATA", "test", "COVID-19")
    PATH_TEST_NORMAL_OLD = os.path.join(PATH_BASE, "DONE_DATA", "test", "NORMAL")
    PATH_TEST_PNEUMONIA_OLD = os.path.join(PATH_BASE, "DONE_DATA", "test", "Pneumonia")
    list_path_old = [PATH_TRAIN_COVID_OLD, PATH_TRAIN_NORMAL_OLD, PATH_TRAIN_PNEUMONIA_OLD,
                     PATH_TEST_COVID_OLD, PATH_TEST_NORMAL_OLD, PATH_TEST_PNEUMONIA_OLD]

    PATH_TRAIN_COVID_NEW = os.path.join(PATH_BASE, "PROCESSED_DATA", "train", "COVID-19")
    PATH_TRAIN_NORMAL_NEW = os.path.join(PATH_BASE, "PROCESSED_DATA", "train", "NORMAL")
    PATH_TRAIN_PNEUMONIA_NEW = os.path.join(PATH_BASE, "PROCESSED_DATA", "train", "Pneumonia")
    PATH_TEST_COVID_NEW = os.path.join(PATH_BASE, "PROCESSED_DATA", "test", "COVID-19")
    PATH_TEST_NORMAL_NEW = os.path.join(PATH_BASE, "PROCESSED_DATA", "test", "NORMAL")
    PATH_TEST_PNEUMONIA_NEW = os.path.join(PATH_BASE, "PROCESSED_DATA", "test", "Pneumonia")
    list_path_new = [PATH_TRAIN_COVID_NEW, PATH_TRAIN_NORMAL_NEW, PATH_TRAIN_PNEUMONIA_NEW,
                     PATH_TEST_COVID_NEW, PATH_TEST_NORMAL_NEW, PATH_TEST_PNEUMONIA_NEW]

    os.mkdir(os.path.join(PATH_BASE, "PROCESSED_DATA"))
    os.mkdir(os.path.join(PATH_BASE, "PROCESSED_DATA", "train"))
    os.mkdir(os.path.join(PATH_BASE, "PROCESSED_DATA", "test"))

    for i in range(len(list_path_old)):
        oldPath = list_path_old[i]
        newPath = list_path_new[i]
        os.mkdir(newPath)

        for imagePath in os.listdir(oldPath):
            # imagePath contains name of the image
            inputPath = os.path.join(oldPath, imagePath)

            # inputPath contains the full directory name
            img = cv.imread(inputPath, cv.IMREAD_GRAYSCALE)

            # fullOutPath contains the path of the output
            fullOutPath = os.path.join(newPath, imagePath)

            cv.imwrite(fullOutPath, preprocessImg(img))


if __name__ == "__main__":
    main()
