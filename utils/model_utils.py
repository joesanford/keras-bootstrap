from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os


def get_classification_name(index, data_dir):
    '''
    Gets the classification name for a single index of the classification list
    
    :param index: the resulting index from model.predict
    :param data_dir: location of training dataset
    :return: classification name
    '''
    generator = ImageDataGenerator().flow_from_directory(data_dir)
    label_map = generator.class_indices
    for name, num in label_map.items():
        if num == index:
            return name

    return 'No name found'


def get_classification_names(indices, data_dir):
    '''
    Gets the classification names for multiple indices of the classification list

    :param indices: the resulting indices from model.predict
    :param data_dir: location of training dataset
    :return: classification name
    '''
    generator = ImageDataGenerator().flow_from_directory(data_dir)
    label_map = generator.class_indices

    results = []

    for index in indices:
        for name, num in label_map.items():
            if num == index:
                results.append(name)

    return results


def get_prediction(model_file, test_image, img_width, img_height, data_dir):
    '''
    Gets a prediction for a single input image
    
    :param model_file: name of the file that the model is saved to
    :param test_image: location of image to be predicted
    :param img_width: width of images in pixels
    :param img_height: height of images in pixels
    :param data_dir: location of data 
    :return: predicted class of test_image
    '''
    model = load_model(model_file)
    img = image.load_img(test_image, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images = np.vstack([img])

    class_index = model.predict_classes(images)
    prediction = get_classification_name(class_index, data_dir)

    return prediction


def get_predictions(model_file, image_directory, img_width, img_height):
    '''
    Gets predictions for a directory of input images
    
    :param model_file: name of the file that the model is saved to
    :param test_image: location of images to be predicted
    :param img_width: width of images in pixels
    :param img_height: height of images in pixels
    :return: list of predictions for each input file
    '''
    model = load_model(model_file)

    images = [os.path.join(image_directory, path) for path in os.listdir(image_directory)]
    loaded_images = []

    for test_img in images:
        img = image.load_img(test_img, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        loaded_images.append(img)

    images = np.vstack(loaded_images)

    class_indexes = model.predict_classes(images, image_directory)
    prediction = get_classification_names(class_indexes)

    return prediction


def evaluate_model(model_file, test_image_path, img_width, img_height, batch_size):
    '''
    Evaluates a model with a given test set
    
    :param model_file: name of the file that the model is saved to
    :param test_image_path: location of images to be used for evaluation
    :param img_width: width of images in pixels
    :param img_height: height of images in pixels
    :param batch_size: batch size for running the evaluation
    :return: resulting score of model evaluation
    '''
    test_generator = ImageDataGenerator().flow_from_directory(
        test_image_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    model = load_model(model_file)
    score = model.evaluate_generator(test_generator)

    return score
