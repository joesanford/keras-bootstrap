from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
from classifier import img_width, img_height, batch_size, training_data_dir
import numpy as np
import os


def get_classification_name(index):
    generator = ImageDataGenerator().flow_from_directory(training_data_dir, batch_size=batch_size)
    label_map = generator.class_indices
    for name, num in label_map.items():
        if num == index:
            return name

    return 'No name found'


def get_classification_names(indexes):
    generator = ImageDataGenerator().flow_from_directory(training_data_dir, batch_size=batch_size)
    label_map = generator.class_indices

    results = []

    for index in indexes:
        for name, num in label_map.items():
            if num == index:
                results.append(name)

    return results


def get_prediction(model_file, test_image):
    model = load_model(model_file)
    img = image.load_img(test_image, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images = np.vstack([img])

    class_index = model.predict_classes(images, batch_size=10)
    prediction = get_classification_name(class_index)

    return prediction


def get_predictions(model_file, image_directory):
    model = load_model(model_file)

    images = [os.path.join(image_directory, path) for path in os.listdir(image_directory)]
    loaded_images = []

    for test_img in images:
        img = image.load_img(test_img, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        loaded_images.append(img)

    images = np.vstack(loaded_images)

    class_indexes = model.predict_classes(images, batch_size=10)
    prediction = get_classification_names(class_indexes)

    return prediction


def evaluate_model(model_file, test_image_path):
    test_generator = ImageDataGenerator().flow_from_directory(
        test_image_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    model = load_model(model_file)
    score = model.evaluate_generator(test_generator)

    return score
