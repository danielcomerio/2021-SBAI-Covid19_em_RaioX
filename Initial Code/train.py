import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

import pandas as pd


def build_train_val_test_datasets(dataset_dir, batch_size, img_size):
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


def build_model(input_size, data_augmentation):
    IMG_SHAPE = input_size + (3,)

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(3, activation='softmax')

    inputs = tf.keras.Input(shape=(250, 250, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def plot_history(history_fine):
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
    plt.show()

def convertHistoryToCSV(hist_df):
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history_fine.history)

    # save to csv:
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


def main():
    DS_PATH = "C:\\Users\\danie\\Desktop\\Artigo-Daniel\\DATA_SET\\DONE_DATA"
    BATCH_SIZE = 32
    IMG_SIZE = (250, 250)
    LR = 1e-4
    N_EPOCHS = 20

    train_ds, val_ds = build_train_val_test_datasets(DS_PATH, BATCH_SIZE, IMG_SIZE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
        tf.keras.layers.experimental.preprocessing.RandomTranslation((-0.1, 0.1), (-0.1, 0.1)),
        tf.keras.layers.experimental.preprocessing.RandomZoom((-0.2, 0.2), (-0.2, 0.2))
    ])

    #####
    model = build_model(IMG_SIZE, data_augmentation)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
                  metrics=['accuracy'])

    model.summary()
    #####

    history_fine = model.fit(train_ds, epochs=N_EPOCHS, validation_data=val_ds)
    plot_history(history_fine)

    model.save('modeloMobileNetV2-1.0.h5')


if __name__ == "__main__":
    main()
