import os

import cv2 as cv
import numpy as np
import argparse


def preprocessImg(img):
    img = img.astype(np.float32)
    kernel = np.ones((32, 32), np.float32)

    # Imagem TopHat
    top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    # Imagem BlackHat
    black_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
    # Imagem transformada
    img_transformed = img + top_hat - black_hat

    # normalization
    img_transformed -= np.min(img_transformed)
    img_transformed /= np.max(img_transformed)
    img_transformed *= 255

    # Imagem transformada e equalizada
    img_transformed = img_transformed.astype(np.uint8)
    img_transformed = cv.equalizeHist(img_transformed)
    img_transformed = cv.resize(img_transformed, (250, 250))

    return img_transformed


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to the dataset", type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_command_line_args()

    ORIG_DIR = "DONE_DATA"
    OUTPUT_DIR = "PROCESSED_DATA"

    PATH_TRAIN_COVID_OLD = os.path.join(
        args.dataset, ORIG_DIR, "train", "COVID-19")
    PATH_TRAIN_NORMAL_OLD = os.path.join(
        args.dataset, ORIG_DIR, "train", "NORMAL")
    PATH_TRAIN_PNEUMONIA_OLD = os.path.join(
        args.dataset, ORIG_DIR, "train", "Pneumonia")
    PATH_TEST_COVID_OLD = os.path.join(
        args.dataset, ORIG_DIR, "test", "COVID-19")
    PATH_TEST_NORMAL_OLD = os.path.join(
        args.dataset, ORIG_DIR, "test", "NORMAL")
    PATH_TEST_PNEUMONIA_OLD = os.path.join(
        args.dataset, ORIG_DIR, "test", "Pneumonia")
    list_path_old = [PATH_TRAIN_COVID_OLD, PATH_TRAIN_NORMAL_OLD, PATH_TRAIN_PNEUMONIA_OLD,
                     PATH_TEST_COVID_OLD, PATH_TEST_NORMAL_OLD, PATH_TEST_PNEUMONIA_OLD]

    PATH_TRAIN_COVID_NEW = os.path.join(
        args.dataset, OUTPUT_DIR, "train", "COVID-19")
    PATH_TRAIN_NORMAL_NEW = os.path.join(
        args.dataset, OUTPUT_DIR, "train", "NORMAL")
    PATH_TRAIN_PNEUMONIA_NEW = os.path.join(
        args.dataset, OUTPUT_DIR, "train", "Pneumonia")
    PATH_TEST_COVID_NEW = os.path.join(
        args.dataset, OUTPUT_DIR, "test", "COVID-19")
    PATH_TEST_NORMAL_NEW = os.path.join(
        args.dataset, OUTPUT_DIR, "test", "NORMAL")
    PATH_TEST_PNEUMONIA_NEW = os.path.join(
        args.dataset, OUTPUT_DIR, "test", "Pneumonia")
    list_path_new = [PATH_TRAIN_COVID_NEW, PATH_TRAIN_NORMAL_NEW, PATH_TRAIN_PNEUMONIA_NEW,
                     PATH_TEST_COVID_NEW, PATH_TEST_NORMAL_NEW, PATH_TEST_PNEUMONIA_NEW]

    os.mkdir(os.path.join(args.dataset, OUTPUT_DIR))
    os.mkdir(os.path.join(args.dataset, OUTPUT_DIR, "train"))
    os.mkdir(os.path.join(args.dataset, OUTPUT_DIR, "test"))

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
