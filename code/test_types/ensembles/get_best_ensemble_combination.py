import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score, classification_report
from more_itertools import powerset
import random


models_path = [
    "test_mobilenetNormal.txt",
    "test_mobilenetProcessed.txt",
    "test_resnetNormal.txt",
    "test_resnetProcessed.txt",
    "test_efficientnetNormal.txt",
    "test_efficientnetProcessed.txt",
    "test_inceptionNormal.txt",
    "test_inceptionProcessed.txt",
]


def get_file_line_values(line):
    line = line.strip().split(", ")
    image = line[0].split('\\')[-1]
    label = str(line[1])

    predictions = []
    for pos in range(2, len(line)):
        predictions.append(float(line[pos]))

    #predictions = np.argmax(predictions, axis=-1)

    return image, label, predictions


def define_class_maximum(maximums, predicted_class):
    if predicted_class[0] > maximums[0]:
        maximums[0] = predicted_class[0]

    if predicted_class[1] > maximums[1]:
        maximums[1] = predicted_class[1]

    if predicted_class[2] > maximums[2]:
        maximums[2] = predicted_class[2]

    return maximums


def define_class_vote(votes, predicted_class):
    if str(predicted_class) == '0':
        votes[0] = votes[0] + 1
    elif str(predicted_class) == '1':
        votes[1] = votes[1] + 1
    else:
        votes[2] = votes[2] + 1

    return votes


def define_class_average(predictions_sum, predicted_class):
    predictions_sum[0] += predicted_class[0]
    predictions_sum[1] += predicted_class[1]
    predictions_sum[2] += predicted_class[2]

    return predictions_sum


def predictions_average(predictions_sum):
    n_classes = int(len(predictions_sum))
    for i in range(n_classes):
        predictions_sum[i] = predictions_sum[i]/n_classes

    return predictions_sum


def create_prediction_string(predicted_class):
    predicted_string = ''
    if str(predicted_class) == '0':
        predicted_string = "1, 0, 0"
    elif str(predicted_class) == '1':
        predicted_string = "0, 1, 0"
    else:
        predicted_string = "0, 0, 1"

    return predicted_string


def get_predicted_class(predicted_values):
    predicted_class = np.argmax(predicted_values, axis=-1)

    more_than_one = False
    tied_classes = []
    for pos in range(len(predicted_values)):
        if predicted_values[pos] == predicted_values[predicted_class]:
            tied_classes.append(pos)
            more_than_one = True

    if more_than_one:
        predicted_class = random.choice(tied_classes)

    return predicted_class


def get_metrics(file_path, best_accuracy):
    file_metrics = open(file_path, "r")
    #file_metrics.write("path_imagem, classe_real, classe_predita")

    melhorou = False
    label_list = []
    predict_list = []
    line = file_metrics.readline()
    while line:
        line = line.strip().split(", ")
        label_list.append(line[1])
        prediction = [float(line[2]), float(line[3]), float(line[4])]
        prediction = np.argmax(prediction, axis=-1)
        predict_list.append(str(prediction))

        line = file_metrics.readline()

    file_metrics.close()

    accuracy = accuracy_score(label_list, predict_list)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        melhorou = True

    return best_accuracy, melhorou


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filespath", help="path to the dataset", type=str)
    parser.add_argument("-me", "--metrics", type=str, default="metrics.txt")

    args = parser.parse_args()

    return args


# py get_best_ensemble_combination.py C:\Users\danie\Desktop\ArtigoDaniel\2021-SBAI-Covid19_em_RaioX\tests_results\files_results -me results.txt
# py ..\..\metrics.py -me C:\Users\danie\Desktop\ArtigoDaniel\2021-SBAI-Covid19_em_RaioX\code\test_types\ensembles\results.txt


def main():
    args = parse_command_line_args()
    BASE_PATH = args.filespath
    best_accuracy = 0
    best_combination = []

    files_path = []
    for model in models_path:
        path = os.path.join(BASE_PATH, model)
        files_path.append(path)

    combination_list = list(powerset(files_path))[1:]

    for combination in combination_list:
        final_file = open(args.metrics, 'w')

        comb = []
        for model in combination:
            comb.append(open(model, 'r'))
        combination = comb

        line = " "
        while line != '':
            predictions_sum = [0, 0, 0]

            line = combination[0].readline()
            if line == '':
                break

            image, label, prediction = get_file_line_values(line)
            predictions_sum = define_class_average(predictions_sum, prediction)

            for pos in range(1, len(combination)):
                line = combination[pos].readline()
                image_compare, _, prediction = get_file_line_values(line)

                if image != image_compare:
                    raise Exception(
                        "Erro, ocorreu manipulação de imagens diferentes.")

                predictions_sum = define_class_average(
                    predictions_sum, prediction)

                image = image_compare

            predictions_sum = predictions_average(predictions_sum)  # -

            predicted_class = get_predicted_class(predictions_sum)

            prediction_string = create_prediction_string(predicted_class)

            final_file.write(str(image) + ", " + str(label) +
                             ", " + prediction_string + '\n')

        for file in combination:
            file.close()
        final_file.close()

        best_accuracy, melhorou = get_metrics(args.metrics, best_accuracy)

        if melhorou:
            best_combination = combination

    print("accuracy_score:", best_accuracy)
    print("best_combination:", best_combination)


if __name__ == "__main__":
    main()
