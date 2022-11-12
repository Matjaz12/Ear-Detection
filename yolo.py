import numpy as np
import numpy.typing as npt
import torch

from load_data import load_data_pickle
from visualize import show_detection


class YOLO:
    def __init__(self, weights_path: str):
        self.model = torch.hub.load("ultralytics/yolov5", 'custom', path=weights_path)

    @staticmethod
    def convert_detection(image_height: int, image_width: int, detection: npt.NDArray) -> npt.NDArray:
        """
        Convert detection returned by YOLO algorithm
        :param image_height: height of the input image
        :param image_width: width of the input image
        :param detection: Detection returned by YOLO algorithm,
            [x_min, y_min, x_max, y_max]
        :return:
            Detection in the following format:
            [x_center, y_center, box_width, box_height]
        """

        x_min, y_min, x_max, y_max = detection

        box_width = np.abs(x_max - x_min)
        box_height = np.abs(y_max - y_min)
        x_center = x_min + box_width // 2
        y_center = y_min + box_height // 2

        x_center = x_center / image_width
        y_center = y_center / image_height
        box_width = box_width / image_width
        box_height = box_height / image_height

        converted_detection = np.array([x_center, y_center, box_width, box_height])
        return converted_detection

    def predict(self, images: npt.NDArray, convert_coordinates: bool = True) -> npt.NDArray:
        """
        Function predicts the bounding boxes.
        :param images: Input images
        :param convert_coordinates: Convert predicted coordinates or not.
        :return: List of predictions:
            [[sample_idx, predicted_class, predicted_prob, bounding_box], ...]
        """

        # Compute detections for all images
        predictions = []
        for sample_idx, image in enumerate(images):
            result = self.model(image)
            detections = np.array(result.xyxy[0].tolist())

            # Iterate over detections and construct prediction vector
            # detections = [[x_min, y_min, x_max, y_max, confidence, predicted_class], ...]
            for detection in detections:
                predicted_class = detection[-1]
                predicted_prob = detection[-2]
                prediction = [sample_idx, predicted_class, predicted_prob]
                detection = detection[0:4]

                if convert_coordinates:
                    image_height, image_width = image.shape[0], image.shape[1]
                    detection = YOLO.convert_detection(image_height, image_width, detection)

                prediction.extend(detection)

                # Store prediction
                predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions

    def __str__(self):
        return "YOLO"


if __name__ == "__main__":
    X_test, y_test = load_data_pickle("./ear_data/X_test_and_y_test.pickle")
    yolo = YOLO("./weights/yolo5s.pt")

    predictions = yolo.predict(X_test)

    # Show ground truth and prediction for a random sample.
    index_to_show = int(np.random.choice(predictions[:, 0]))
    pred = predictions[np.where(predictions[:, 0] == index_to_show)]
    show_detection(X_test[index_to_show],
                   y_test[index_to_show],
                   pred,
                   figure_name=str(yolo))
