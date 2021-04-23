from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, classification_report

import numpy as np


def main():
    file_metrics = open("metrics.txt", "r")
    #file_metrics.write("path_imagem, classe_real, classe_predita")

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

    print("confusion_matrix: \n", confusion_matrix(label_list, predict_list))
    print("accuracy_score:", accuracy_score(label_list, predict_list))
    print("precision_score:", precision_score(label_list, predict_list, average='macro'))
    print("f1_score:", f1_score(label_list, predict_list, average='macro'))
    print("recall_score:", recall_score(label_list, predict_list, average='macro'))


if __name__ == "__main__":
    main()
