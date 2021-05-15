import argparse
import os
import numpy as np


models_path = [
    "test_mobilenetNormal.txt",
    "test_mobilenetProcessed.txt",
    "test_resnetNormal.txt",
    "test_resnetProcessed.txt",
    "test_efficientnetNormal.txt",
    "test_efficientnetProcessed.txt",
    "test_inceptionNormal.txt",
    "test_inceptionProcessed.txt"
]


def get_file_line_values(line):
    line = line.strip().split(", ")
    image = line[0].split('\\')[-1]
    label = str(line[1])

    predictions = []
    for pos in range(2, len(line)):
        predictions.append(float(line[pos]))

    # mudar aqui para as outras estratégias
    predicted_class = np.argmax(predictions, axis=-1)

    return image, label, predicted_class


def define_class(votes, predicted_class):
    if str(predicted_class) == '0':
        votes[0] = votes[0] + 1
    elif str(predicted_class) == '1':
        votes[1] = votes[1] + 1
    else:
        votes[2] = votes[2] + 1

    return votes


def create_prediction_string(predicted_class):
    predicted_string = ''
    if str(predicted_class) == '0':
        predicted_string = "1, 0, 0"
    elif str(predicted_class) == '1':
        predicted_string = "0, 1, 0"
    else:
        predicted_string = "0, 0, 1"

    return predicted_string


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filespath", help="path to the dataset", type=str)
    parser.add_argument("-me", "--metrics", type=str, default="metrics.txt")

    args = parser.parse_args()

    return args

# py ensemble_vote.py C:\Users\danie\Desktop\ArtigoDaniel\2021-SBAI-Covid19_em_RaioX\tests_results\files_results -me results.txt
# py ..\..\..\metrics.py -me C:\Users\danie\Desktop\ArtigoDaniel\2021-SBAI-Covid19_em_RaioX\code\test_types\ensembles\ensemble_vote\results.txt


def main():
    args = parse_command_line_args()
    BASE_PATH = args.filespath
    final_file = open(args.metrics, 'w')

    files_path = []
    for model in models_path:
        path = os.path.join(BASE_PATH, model)
        files_path.append(open(path, 'r'))

    line = " "
    while line != '':
        votes = [0, 0, 0]

        line = files_path[0].readline()
        if line == '':
            break

        image, label, prediction = get_file_line_values(line)
        votes = define_class(votes, prediction)

        for pos in range(1, len(files_path)):
            line = files_path[pos].readline()
            image_compare, _, prediction = get_file_line_values(line)

            if image != image_compare:
                raise Exception(
                    "Erro, ocorreu manipulação de imagens diferentes.")

            votes = define_class(votes, prediction)

            image = image_compare

        predicted_class = np.argmax(votes, axis=-1)

        prediction_string = create_prediction_string(predicted_class)

        final_file.write(str(image) + ", " + str(label) +
                         ", " + prediction_string + '\n')

    for file in files_path:
        file.close()
    final_file.close()


if __name__ == "__main__":
    main()
