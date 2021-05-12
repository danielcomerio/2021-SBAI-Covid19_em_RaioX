import numpy as np
import argparse
import os
import cv2 as cv


def emphasizeHeatMapIntensity(img_array, img_type):
    img_array = cv.imread(img_array)
    emphasized_img = img_array.copy()

    img_array = cv.cvtColor(img_array, cv.COLOR_BGR2HSV)

    if img_type == "points":
        lower = np.array([0, 100, 50])
    else:
        lower = np.array([0, 1, 50])

    upper = np.array([10, 255, 255])
    mask = cv.inRange(img_array, lower, upper)

    contours, hierarchy = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cv.drawContours(emphasized_img, contours, -1, (0, 0, 255), 1)

    img_array = cv.cvtColor(img_array, cv.COLOR_HSV2BGR)

    return emphasized_img


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to the dataset", type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_command_line_args()

    ORIG_DIR = "HEATMAPS_ORIGINAL"
    OUTPUT_DIR = "HEATMAPS_EMPHASIZED"
    os.mkdir(os.path.join(args.dataset, OUTPUT_DIR))

    models = ["resnet", "mobilenet", "efficientnet", "inception"]
    image_dirs = ["COVID-19", "NORMAL", "Pneumonia"]
    for name in models:
        os.mkdir(os.path.join(args.dataset, OUTPUT_DIR, name))

        models_type = ["normal", "processed"]
        for model_type in models_type:
            os.mkdir(os.path.join(args.dataset, OUTPUT_DIR, name, model_type))

            PATH_TEST_COVID_OLD = os.path.join(
                args.dataset, ORIG_DIR, name, model_type, "COVID-19")
            PATH_TEST_NORMAL_OLD = os.path.join(
                args.dataset, ORIG_DIR, name, model_type, "NORMAL")
            PATH_TEST_PNEUMONIA_OLD = os.path.join(
                args.dataset, ORIG_DIR, name, model_type, "Pneumonia")
            list_path_old = [PATH_TEST_COVID_OLD,
                             PATH_TEST_NORMAL_OLD, PATH_TEST_PNEUMONIA_OLD]

            img_type = ["points", "area"]
            for i in range(len(list_path_old)):

                for typ in img_type:
                    oldPath = os.path.join(list_path_old[i], typ)

                    newPathPointsIntegrated = os.path.join(
                        args.dataset, OUTPUT_DIR, name, model_type, image_dirs[i], typ)
                    os.makedirs(newPathPointsIntegrated)

                    for imagePath in os.listdir(oldPath):
                        # imagePath contains name of the image
                        inputPath = os.path.join(oldPath, imagePath)

                        img_points_integrated = emphasizeHeatMapIntensity(
                            inputPath, typ)

                        # fullOutPath contains the path of the output
                        fullOutPath = os.path.join(
                            newPathPointsIntegrated, imagePath)
                        cv.imwrite(fullOutPath, img_points_integrated)


if __name__ == "__main__":
    main()
