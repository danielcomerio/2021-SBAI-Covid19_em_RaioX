
import time
import pickle
import argparse
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

import pandas as pd

# https://www.tensorflow.org/api_docs/python/tf/keras/applications
models = {
    "resnet": {
        "preproc": tf.keras.applications.resnet50.preprocess_input,
        "model": tf.keras.applications.resnet50.ResNet50
    },
    "mobilenet": {
        "preproc": tf.keras.applications.mobilenet_v2.preprocess_input,
        "model": tf.keras.applications.mobilenet_v2.MobileNetV2,
    },
    "efficientnet": {
        "preproc": tf.keras.applications.efficientnet.preprocess_input,
        "model": tf.keras.applications.efficientnet.EfficientNetB2,
    },
    "inception": {
        "preproc": tf.keras.applications.inception_v3.preprocess_input,
        "model": tf.keras.applications.inception_v3.InceptionV3,
    },
    # BiT Model: https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html
    # Sinclrv2: https://github.com/google-research/simclr
}


def build_train_val_datasets(dataset_dir, batch_size, img_size):
    train_dir = os.path.join(dataset_dir, "train")

    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 image_size=img_size)

    # build validation dataset from the train dataset.
    train_batches = tf.data.experimental.cardinality(train_dataset)
    n_validation_samples = int(train_batches // 10)
    val_dataset = train_dataset.take(n_validation_samples)
    train_dataset = train_dataset.skip(n_validation_samples)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset


def build_model(input_size, data_augmentation, args):
    global models

    IMG_SHAPE = input_size + (3,)

    preprocess_input = models[args.model]["preproc"]
    base_model = models[args.model]["model"](input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(3)

    inputs = tf.keras.Input(shape=(250, 250, 3))
    if data_augmentation is not None:
        x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    if args.add_dense:
        x = tf.keras.layers.Dense(256)
        x = tf.keras.layers.Dense(256)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def plot_history(history_fine, output_dir):
    acc = history_fine.history['accuracy']
    val_acc = history_fine.history['val_accuracy']
    loss = history_fine.history['loss']
    val_loss = history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig(os.path.join(output_dir, "history.png"))
    plt.show()


def convertHistoryToCSV(hist_df):
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history_fine.history)

    # save to csv:
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


def build_data_augmentation(do_augmentation):
    if not do_augmentation:
        return None

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip(
            mode="horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            (-0.1, 0.1), (-0.1, 0.1)),
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            (-0.2, 0.2), (-0.2, 0.2))
    ])

    return data_augmentation


def save(output_dir, data, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, filename)
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print(f"File {filename} saved.")


def parse_command_line_args():
    global models

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to the dataset", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-l", "--lr", help="learning rate",
                        type=float, default=1e-4)
    parser.add_argument("-n", "--n_epochs", type=int, default=20)
    parser.add_argument("-m", "--model", type=str,
                        choices=models.keys(), default="mobilenet")
    parser.add_argument("-a", "--augment",
                        help="perform data augmentation",
                        type=int, choices=[0, 1], default=1)
    parser.add_argument("-d", "--add_dense",
                        help="wether to add dense layers before the output",
                        type=int, choices=[0, 1], default=0)
    parser.add_argument("-t", "--tag", type=str, default="",
                        help="add the text to the output dir name")

    args = parser.parse_args()

    return args


def main():
    args = parse_command_line_args()

    if len(args.tag) > 0:
        tag = "_" + args.tag
    output_dir = os.path.join("results", str(time.time()) + tag)
    print(f"** OUTPUT_DIR: {output_dir}")
    save(output_dir, args, "args.pickle")

    IMG_SIZE = (250, 250)
    train_ds, val_ds = build_train_val_datasets(
        args.dataset, args.batch_size, IMG_SIZE)

    data_augmentation = build_data_augmentation(args.augment)
    model = build_model(IMG_SIZE, data_augmentation, args)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    history_fine = model.fit(
        train_ds, epochs=args.n_epochs, validation_data=val_ds)
    plot_history(history_fine, output_dir)

    save(output_dir, history_fine, "history.pickle")
    model.save(os.path.join(output_dir, 'model.h5'))

    # esse arquivo serve apenas para que a gente consiga verificar
    # quais experimentos terminaram olhando para os dados do diretorio.
    save(output_dir, "OK!", "ok.pickle")
    print("Ok.")


if __name__ == "__main__":
    main()
