import os
os.environ['MPLBACKEND'] = 'Agg'

import pickle
import copy

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from data_prep import prep_and_load_data
from model import get_model
import constants as CONST


def plotter(history_file):
    with open(history_file, 'rb') as file:
        history = pickle.load(file)

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.tight_layout()
    plt.savefig('18_000_15epoch_accuracy.png')
    plt.show()
    plt.close()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.tight_layout()
    plt.savefig('18_000_15epoch_loss.png')
    plt.show()
    plt.close()


def video_write(model):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("./prediction.mp4", fourcc, 1.0, (400, 400))
    val_map = {1: 'Dog', 0: 'Cat'}

    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (20, 20)
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_type = 2

    DIR = CONST.TEST_DIR
    image_paths = sorted(os.listdir(DIR))
    image_paths = image_paths[:200]
    count = 0
    for img_path in image_paths:
        image, image_std = process_image(DIR, img_path)

        image_std = image_std.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
        pred = model.predict(image_std, verbose=0)
        arg_max = np.argmax(pred, axis=1)
        max_val = np.max(pred, axis=1)
        label = val_map[arg_max[0]]
        text = f'{label} - {max_val[0] * 100:.2f}%'
        cv2.putText(image, text, location, font, font_scale, font_color, line_type)

        frame = cv2.resize(image, (400, 400))
        out.write(frame)

        count += 1
        print(count)
    out.release()


def process_image(directory, img_path):
    path = os.path.join(directory, img_path)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'Unable to read image: {path}')
    image_copy = copy.deepcopy(image)

    image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
    image_std = image.astype('float32') / 255.0
    return image_copy, image_std


if __name__ == "__main__":
    data = np.array(prep_and_load_data(), dtype=object)
    train_size = int(CONST.DATA_SIZE * CONST.SPLIT_RATIO)
    print('dataset length:', len(data), 'train size:', train_size)

    train_data = data[:train_size]
    train_images = np.array([i[0] for i in train_data]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
    train_labels = np.array([i[1] for i in train_data])
    print('train data fetched..')

    test_data = data[train_size:]
    test_images = np.array([i[0] for i in test_data]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
    test_labels = np.array([i[1] for i in test_data])
    print('data fetched..')

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    model = get_model()
    print('training started...')
    history = model.fit(
        train_images,
        train_labels,
        batch_size=50,
        epochs=15,
        verbose=1,
        validation_data=(test_images, test_labels),
        callbacks=[tensorboard],
    )
    print('training done...')

    model.save('18_000.h5')

    history_file = '18_000_history.pickle'
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    plotter(history_file)
    video_write(model)



