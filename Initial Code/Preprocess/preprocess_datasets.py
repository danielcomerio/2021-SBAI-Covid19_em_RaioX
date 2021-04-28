
import time
from tqdm import tqdm
import pandas as pd
import glob
import numpy as np
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
import pydicom

import argparse


def get_datasets_dirs(datasets_dir):
    files_and_dirs = os.listdir(datasets_dir)
    files_and_dirs_with_paths = map(
        lambda x: os.path.join(datasets_dir, x), files_and_dirs)
    dirs_with_paths = list(
        filter(lambda x: os.path.isdir(x), files_and_dirs_with_paths))
    return dirs_with_paths


def build_output_dirs(dataset_dir):
    if os.path.exists(dataset_dir):
        print(f"Cleaning dataset directory {dataset_dir}")
        shutil.rmtree(dataset_dir)
    os.mkdir(f"{dataset_dir}")
    os.mkdir(f"{dataset_dir}/train")
    os.mkdir(f"{dataset_dir}/train/NORMAL")
    os.mkdir(f"{dataset_dir}/train/Pneumonia")
    os.mkdir(f"{dataset_dir}/train/COVID-19")
    os.mkdir(f"{dataset_dir}/test")
    os.mkdir(f"{dataset_dir}/test/NORMAL")
    os.mkdir(f"{dataset_dir}/test/Pneumonia")
    os.mkdir(f"{dataset_dir}/test/COVID-19")


def copy_files_to_appropriate_dirs(files, labels, output_dir, load_fn, offset):
    idx = offset
    files_and_labels = list(zip(files, labels))
    for f, l in tqdm(files_and_labels):
        output_path = (f"{output_dir}/{l}/%06d.png") % idx
        if not os.path.exists(f):
            msg = f"Arquivo de entrada {f} nao encontrado."
            raise Exception(msg)
        if os.path.exists(output_path):
            msg = f'O arquivo {output_path} já existe!'
            raise Exception(msg)
        data = load_fn(f)
        data = data.astype(np.float64)
        data = data - np.min(data)
        data = 255.0 * (data / np.max(data))
        data = data.astype(np.uint8)
        cv2.imwrite(output_path, data)
        #print(f"file {output_path} saved.")
        idx += 1


def split_dataset_and_copy_files(files, labels, output_dir, load_fn, offset):
    # remove duplicates
    for f in files:
        ids = [i for i in range(len(files)) if files[i] == f]
        if len(ids) > 1:
            ids_to_remove = sorted(ids)[1:]
            while len(ids_to_remove) > 0:
                idx = ids_to_remove.pop()
                files.pop(idx)
                labels.pop(idx)

    # double check
    for f in files:
        ids = [i for i in range(len(files)) if files[i] == f]
        if len(ids) > 1:
            msg = "Ainda existem itens duplicados."
            raise Exception(msg)

    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=0.1, stratify=labels, shuffle=True)

    print("Copying train files")
    copy_files_to_appropriate_dirs(
        train_files, train_labels, f"{output_dir}/train", load_fn, offset)
    offset += len(train_files)

    print("Copying test files")
    copy_files_to_appropriate_dirs(
        test_files, test_labels, f"{output_dir}/test", load_fn, offset)
    offset += len(test_files)

    return offset


def preprocess_covid19_radiography_ds(dir_ds):
    files = glob.glob(dir_ds + "\\*\\*.png")
    labels = map(lambda x: x.split(os.path.sep)[-2].replace(
        "Viral Pneumonia", "Pneumonia").replace("COVID", "COVID-19"), files)
    labels = list(labels)
    files = [os.path.join(dir_ds, f) for f in files]
    return files, labels


def preprocess_actualmed_ds(dir_ds):
    data = pd.read_csv(os.path.join(dir_ds, "metadata.csv"))
    data = data[['imagename', 'finding']]
    data = data.dropna()
    data['finding'] = data['finding'].replace("No finding", "NORMAL")
    files = data['imagename'].to_numpy()
    labels = data['finding'].to_numpy()
    # print("labels:", data.groupby('finding').count())
    files = [os.path.join(dir_ds, "images", f) for f in files]
    return files, labels


def preprocess_covid_chestxray_dataset(dir_ds):
    data = pd.read_csv(os.path.join(dir_ds, "metadata.csv"))
    data = data[['filename', 'finding', 'modality']]

    # remove imagens que nao sao de raio-x
    data = data[data['modality'] != 'CT']

    # normaliza os labels
    data['finding'] = data['finding'].replace("No Finding", "NORMAL")
    data['finding'] = data['finding'].replace(
        "Pneumonia/Viral/COVID-19", "COVID-19")

    # descarta doencas com poucos samples, que nao foram classificadas ou que nao estao no escopo
    findings_to_discard = ['todo', 'Tuberculosis', 'Unknown']
    data = data.drop(data[data['finding'].map(
        lambda x: x in findings_to_discard)].index)

    # normalize os labels de pneumonia
    def rename_fn(x): return 'Pneumonia' if 'Pneumonia' in x else x
    data['finding'] = data['finding'].map(rename_fn)
    #print("findings:", data.groupby("finding").count())
    #print("modalities:", data.groupby("modality").count())

    files = data['filename'].to_numpy()
    files = [os.path.join(dir_ds, "images", f) for f in files]
    labels = data['finding'].to_numpy()

    return files, labels


def preprocess_figure1_covid_dataset(dir_ds):
    path = os.path.join(dir_ds, "metadata.csv")
    with open(path, 'r') as f:
        data = f.readlines()
    data = [line.rstrip().rsplit(",") for line in data]
    data = data[1:]
    data = map(lambda x: [x[0], x[4]], data)
    data = filter(lambda x: len(x[1]) > 2, data)
    data = pd.DataFrame(data, columns=["patientid", "finding"])
    data["finding"] = data["finding"].replace("No finding", "NORMAL")

    labels = data['finding'].to_numpy()
    files = data['patientid'].to_numpy()

    # find the correct extension for the files
    for i, f in enumerate(files):
        for ext in ['.png', '.jpg']:
            path = os.path.join(dir_ds, "images", f + ext)
            if os.path.isfile(path):
                files[i] = path
                break

    return files, labels


def preprocess_rnsa_dataset(dir_ds):
    # Adapted from "create_COVIDx_binary.ipynb" de https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md
    # get all the normal from here
    csv_normal = pd.read_csv(os.path.join(
        dir_ds, "stage_2_detailed_class_info.csv"), nrows=None)
    # get all the 1s from here since 1 indicate pneumonia
    # found that images that aren't pneunmonia and also not normal are classified as 0s
    csv_pneu = pd.read_csv(os.path.join(
        dir_ds, "stage_2_train_labels.csv"), nrows=None)

    files, labels = [], []

    for _, row in csv_normal.iterrows():
        if row['class'] == 'Normal':
            if row['patientId'] not in files:
                files.append(row['patientId'])
                labels.append("NORMAL")

    for _, row in csv_pneu.iterrows():
        if int(row['Target']) == 1:
            if row['patientId'] not in files:
                files.append(row['patientId'])
                labels.append("Pneumonia")

    files = [os.path.join(dir_ds, "stage_2_train_images",
                          f + ".dcm") for f in files]

    return files, labels


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to the dataset", type=str)
    args = parser.parse_args()

    return args


def main():
    # py preprocess_datasets.py C:\\Users\\danie\\Desktop\\Artigo-Daniel\\DATA_SET
    args = parse_command_line_args()
    os.mkdir(os.path.join(args.dataset, "DONE_DATA"))

    # Necessário que tal pasta esteja criada e preenchida com os datasets individuais
    DATASETS_DIR = os.path.join(args.dataset, "RAW_DATA")
    OUTPUT_DIR = os.path.join(args.dataset, "DONE_DATA")

    covid19_radiography_path = os.path.join(
        DATASETS_DIR, "COVID-19_Radiography_Database")
    covid_actualmed_path = os.path.join(
        DATASETS_DIR, "Actualmed-COVID-chestxray-dataset-master")
    covid_chestxray_path = os.path.join(
        DATASETS_DIR, "covid-chestxray-dataset-master")
    figure1_covid_path = os.path.join(
        DATASETS_DIR, "Figure1-COVID-chestxray-dataset-master")
    rnsa_path = os.path.join(
        DATASETS_DIR, "rsna-pneumonia-detection-challenge")

    build_output_dirs(OUTPUT_DIR)

    def general_loader(x): return cv2.imread(x)
    def dicom_loader(x): return pydicom.dcmread(x).pixel_array
    offset = 1
    start_t = time.time()

    print("Processing the dataset covid19_radiography")
    files, labels = preprocess_covid19_radiography_ds(covid19_radiography_path)
    offset = split_dataset_and_copy_files(
        files, labels, OUTPUT_DIR, general_loader, offset)

    print("Processing the dataset actualmed")
    files, labels = preprocess_actualmed_ds(covid_actualmed_path)
    offset = split_dataset_and_copy_files(
        files, labels, OUTPUT_DIR, general_loader, offset)

    print("Processing the dataset covid_chestxray")
    files, labels = preprocess_covid_chestxray_dataset(covid_chestxray_path)
    offset = split_dataset_and_copy_files(
        files, labels, OUTPUT_DIR, general_loader, offset)

    print("Processing the dataset figure1")
    files, labels = preprocess_figure1_covid_dataset(figure1_covid_path)
    offset = split_dataset_and_copy_files(
        files, labels, OUTPUT_DIR, general_loader, offset)

    print("Processing the dataset rnsa")
    files, labels = preprocess_rnsa_dataset(rnsa_path)
    offset = split_dataset_and_copy_files(
        files, labels, OUTPUT_DIR, dicom_loader, offset)

    print("Preprocessing completed in %.3fs" % (time.time() - start_t))
    print("OK!")


if __name__ == "__main__":
    main()
