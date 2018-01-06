from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import os
import shutil

from classifier import img_width, img_height, batch_size


def prepare_dataset(data_dir, training_data_dir, validation_data_dir, data_ratio):
    '''
    Takes a dataset and splits it into training and validation splits with the 
    provided ratio
    :param data_dir: location of the dataset for the model
    :param training_data_dir: output destination for training dataset
    :param validation_data_dir: output destination for validation dataset
    :param data_ratio: a float value between 0 and 1 of what percentage of data to put in the validation set
    :return: returns when complete
    '''
    classes = [folder for folder in sorted(os.listdir(data_dir))]

    images = []
    for cls in classes:
        images += ([os.path.join(cls, path) for path in os.listdir(os.path.join(data_dir, cls))])

    images = shuffle(images, random_state=42)

    validation_split = int(len(images) * data_ratio)
    training_data = images[:-validation_split]
    validation_data = images[-validation_split:]

    for dataset, dataset_dir in [(training_data, training_data_dir), (validation_data, validation_data_dir)]:
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)

        for cls in classes:
            os.makedirs(os.path.join(dataset_dir, cls))

        for image in dataset:
            shutil.copy(os.path.join(data_dir, image), os.path.join(dataset_dir, image))

    return


def create_data_generators(training_data_dir, validation_data_dir):
    prepare_dataset()
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1. / 255,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.1)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        training_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator