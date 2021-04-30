from tensorflow.python.ops import io_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.data.ops import dataset_ops
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.models import load_model
import argparse


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
}


# Todo esse código é utilizado para sobrescrever a função nativa do keras "image_dataset_from_directory"
# Para poder capturar além do Dataset os PATHs das imagens

WHITELIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')


def image_dataset_from_directory(directory,
                                 labels='inferred',
                                 label_mode='int',
                                 class_names=None,
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear',
                                 follow_links=False):

    if labels != 'inferred':
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                '`labels` argument should be a list/tuple of integer labels, of '
                'the same size as the number of image files in the target '
                'directory. If you wish to infer the labels from the subdirectory '
                'names in the target directory, pass `labels="inferred"`. '
                'If you wish to get a dataset that only contains images '
                '(no labels), pass `label_mode=None`.')
        if class_names:
            raise ValueError('You can only pass `class_names` if the labels are '
                             'inferred from the subdirectory names in the target '
                             'directory (`labels="inferred"`).')
    if label_mode not in {'int', 'categorical', 'binary', None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "categorical", "binary", '
            'or None. Received: %s' % (label_mode,))
    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
            'Received: %s' % (color_mode,))
    interpolation = image_preprocessing.get_interpolation(interpolation)
    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed)

    if seed is None:
        seed = np.random.randint(1e6)
    image_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        labels,
        formats=WHITELIST_FORMATS,
        class_names=class_names,
        shuffle=shuffle,
        seed=seed,
        follow_links=follow_links)

    if label_mode == 'binary' and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary", there must exactly 2 classes. '
            'Found the following classes: %s' % (class_names,))

    image_paths, labels = dataset_utils.get_training_or_validation_split(
        image_paths, labels, validation_split, subset)

    dataset = paths_and_labels_to_dataset(
        image_paths=image_paths,
        image_size=image_size,
        num_channels=num_channels,
        labels=labels,
        label_mode=label_mode,
        num_classes=len(class_names),
        interpolation=interpolation)
    if shuffle:
        # Shuffle locally at each iteration
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.batch(batch_size)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    return dataset, image_paths


def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation):
    """Constructs a dataset of images and labels."""
    # TODO(fchollet): consider making num_parallel_calls settable
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(
        lambda x: path_to_image(x, image_size, num_channels, interpolation))
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(
            labels, label_mode, num_classes)
        img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))
    return img_ds


def path_to_image(path, image_size, num_channels, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img
# Todo esse código é utilizado para sobrescrever a função nativa do keras "image_dataset_from_directory"
# Para poder capturar além do Dataset os PATHs das imagens


def build_train_val_test_datasets(dataset_dir, batch_size, img_size):
    test_dir = os.path.join(dataset_dir, "test")

    test_dataset, test_paths = image_dataset_from_directory(test_dir,
                                                            shuffle=False,
                                                            batch_size=batch_size,
                                                            image_size=img_size)

    AUTOTUNE = tf.data.AUTOTUNE
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return test_dataset, test_paths


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to the dataset", type=str)
    parser.add_argument("-m", "--model", help="type of the model",
                        type=str, choices=models.keys(), default="mobilenet")
    parser.add_argument("-mp", "--modelpath",
                        help="path of the model", type=str, default="model.h5")
    parser.add_argument("-me", "--metrics", type=str, default="metrics.txt")
    parser.add_argument("-b", "--batch_size", type=int, default=32)

    args = parser.parse_args()

    return args


def main():
    args = parse_command_line_args()
    IMG_SIZE = (250, 250)

    test_ds, test_paths = build_train_val_test_datasets(
        args.dataset, args.batch_size, IMG_SIZE)

    model = load_model(args.modelpath)
    model.summary()

    file_metrics = open(args.metrics, "w")
    img_pos = 0
    #file_metrics.write("path_imagem, classe_real, classe_predita")

    for image_batch, label_batch in test_ds.as_numpy_iterator():
        preprocess_input = models[args.model]["preproc"]
        preprocess_input = preprocess_input(image_batch)

        predictions = model.predict_on_batch(preprocess_input)
        predictions = tf.nn.softmax(predictions)

        for pos in range(len(label_batch)):
            probs = ", ".join(
                map(str, tf.keras.backend.get_value(predictions[pos])))
            file_metrics.write(
                str(test_paths[img_pos]) + ", " + str(label_batch[pos]) + ", " + probs + '\n')
            img_pos = img_pos + 1

    file_metrics.close()


if __name__ == "__main__":
    main()
