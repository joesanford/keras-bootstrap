from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from utils.data_utils import create_data_generators
from utils.model_utils import get_prediction, get_predictions, evaluate_model
import argparse

img_width, img_height = 150, 150
data_dir = './images'
model_file = 'model.h5'
training_data_dir = './training'
validation_data_dir = './validation'
data_ratio = 0.15
test_data_dir = './test'
epochs = 100
batch_size = 16
input_shape = (img_width, img_height, 3)
num_classes = 10


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


def create_and_train_model(model_name):
    model = create_model()
    train_generator, validation_generator = create_data_generators(data_dir, training_data_dir, validation_data_dir,
                                                                   img_height, img_width, batch_size, data_ratio)

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size)

    model.save(model_name)

    return 'Done!'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classifier')
    parser.add_argument('--train', help='trains the model', action='store_true')
    parser.add_argument('--evaluate', help='evaluates the model on a test set', action='store_true')
    parser.add_argument('--predict', help='makes a prediction based on a single image', action='store_true')
    parser.add_argument('--predict_dir', action='store_true')
    args = parser.parse_args()

    if args.train:
        results = create_and_train_model(model_file)
    elif args.predict:
        results = get_prediction(model_file, 'test1.jpg', img_width, img_height, training_data_dir)
    elif args.predict_dir:
        results = get_predictions(model_file, './test/electric_guitar')
    elif args.evaluate:
        results = evaluate_model(model_file, './test', img_width, img_height, batch_size)
    else:
        results = 'No commands passed, see --help for more information'

    print(results)
