import os

import cv2
import numpy as np
import tensorflow as tf

import constants as CONST


MODEL_PATH = os.path.join(CONST.PROJECT_DIR, 'output', '18_000.h5')
CLASS_NAMES = ('Cat', 'Dog')


def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f'Trained CNN model not found at {MODEL_PATH}. Run main.py first '
            'or restore the trained model file.'
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def classify_frame(model, frame):
    image = cv2.resize(frame, (CONST.IMG_SIZE, CONST.IMG_SIZE))
    image = image.astype('float32') / 255.0
    probabilities = model(image[np.newaxis, ...], training=False).numpy()[0]
    class_index = int(np.argmax(probabilities))
    return CLASS_NAMES[class_index], float(probabilities[class_index])


def main():
    model = load_model()
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            'Could not open camera 0. Check that a webcam is connected and not in use.'
        )
    print('CNN camera classifier started. Press q in the camera window to quit.')

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError('Could not read a frame from the webcam.')

            label, confidence = classify_frame(model, frame)
            text = f'{label} - {confidence * 100:.1f}%'
            cv2.rectangle(frame, (0, 0), (270, 48), (0, 0, 0), -1)
            cv2.putText(
                frame,
                text,
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (80, 255, 80),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow('CNN Cat/Dog Classifier - press Q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
