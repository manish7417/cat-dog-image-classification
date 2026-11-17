import os

import cv2
import numpy as np

import constants as CONST


MODEL_PATH = os.path.join(
    CONST.PROJECT_DIR, 'models', 'object_detection_yolox_2022nov.onnx'
)
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45
CLASS_NAMES = {0: 'Person', 15: 'Cat', 16: 'Dog'}


def create_anchors():
    grids = []
    expanded_strides = []
    for size, stride in ((80, 8), (40, 16), (20, 32)):
        grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
        grids.append(np.stack((grid_x, grid_y), axis=2).reshape(-1, 2))
        expanded_strides.append(np.full((size * size, 1), stride))
    return (
        np.concatenate(grids).astype(np.float32),
        np.concatenate(expanded_strides).astype(np.float32),
    )


GRIDS, EXPANDED_STRIDES = create_anchors()


def load_detector():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f'Object detector model not found at {MODEL_PATH}. '
            'Check that the YOLOX ONNX model is present in the models folder.'
        )

    detector = cv2.dnn.readNetFromONNX(MODEL_PATH)
    return detector


def detect_objects(detector, frame):
    frame_height, frame_width = frame.shape[:2]
    scale = min(INPUT_SIZE / frame_height, INPUT_SIZE / frame_width)
    resized_width = int(frame_width * scale)
    resized_height = int(frame_height * scale)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resized_width, resized_height)).astype(np.float32)
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.float32)
    padded[:resized_height, :resized_width] = resized
    blob = np.transpose(padded, (2, 0, 1))[np.newaxis, ...]

    detector.setInput(blob)
    predictions = detector.forward(detector.getUnconnectedOutLayersNames())[0][0].copy()
    if predictions.shape[0] != GRIDS.shape[0]:
        raise RuntimeError(
            f'Unexpected object detector output shape: {predictions.shape}'
        )

    predictions[:, :2] = (
        (predictions[:, :2] + GRIDS) * EXPANDED_STRIDES
    )
    predictions[:, 2:4] = (
        np.exp(np.clip(predictions[:, 2:4], -20, 20)) * EXPANDED_STRIDES
    )

    class_scores = predictions[:, 4:5] * predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    selected = np.where(
        (confidences >= CONFIDENCE_THRESHOLD)
        & np.isin(class_ids, tuple(CLASS_NAMES))
    )[0]
    if selected.size == 0:
        return []

    centers = predictions[selected, :2]
    sizes = predictions[selected, 2:4]
    boxes = np.column_stack(
        (centers[:, 0] - sizes[:, 0] / 2,
         centers[:, 1] - sizes[:, 1] / 2,
         sizes[:, 0],
         sizes[:, 1])
    )
    selected_scores = confidences[selected]
    selected_classes = class_ids[selected]

    kept = cv2.dnn.NMSBoxesBatched(
        boxes.tolist(),
        selected_scores.tolist(),
        selected_classes.tolist(),
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
    )
    detections = []
    for index in np.asarray(kept).reshape(-1):
        x, y, width, height = boxes[index]
        x1 = max(0, min(frame_width, int(x / scale)))
        y1 = max(0, min(frame_height, int(y / scale)))
        x2 = max(0, min(frame_width, int((x + width) / scale)))
        y2 = max(0, min(frame_height, int((y + height) / scale)))
        detections.append(
            (
                CLASS_NAMES[int(selected_classes[index])],
                float(selected_scores[index]),
                (x1, y1, x2, y2),
            )
        )
    return detections


def draw_detections(frame, detections):
    animal_names = []
    for label, confidence, (x1, y1, x2, y2) in detections:
        is_animal = label in ('Cat', 'Dog')
        color = (80, 255, 80) if is_animal else (0, 190, 255)
        text = f'{label} - {confidence * 100:.1f}%'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            text,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        if label in ('Cat', 'Dog') and label not in animal_names:
            animal_names.append(label)

    if animal_names:
        status = 'Detected: ' + ', '.join(animal_names)
    elif any(label == 'Person' for label, _, _ in detections):
        status = 'Person detected - no cat/dog detected'
    else:
        status = 'No cat/dog detected'

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(
        frame,
        status,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    detector = load_detector()
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            'Could not open camera 0. Check that a webcam is connected and not in use.'
        )
    print('Live detector started. Show a clear cat/dog image; press q to quit.')

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError('Could not read a frame from the webcam.')

            detections = detect_objects(detector, frame)
            cv2.imshow(
                'Cat and Dog Detector - press Q to quit',
                draw_detections(frame, detections),
            )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
