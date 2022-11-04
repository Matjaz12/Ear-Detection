import torch
import numpy.typing as npt
import numpy as np


class YOLO:
    def __init__(self, weights_path: str):
        self.model = torch.hub.load("ultralytics/yolov5", 'custom', path=weights_path)

    @staticmethod
    def convert_prediction_coordinates(image: npt.NDArray, predicted: npt.NDArray):
        """
        Function converts the predicted bounding box in format which is used in the test samples.

        predicted (x_min, y_min) are the coordinates of the upper-left most pixel.
        predicted (x_max, y_max) are the coordinates of the lower-right most pixel.

        We perform the following two conversion:
        1. Upper-left most pixel coordinates (x_min, y_min) are converted to the coordinates of the central pixel.
        2. We compute (width_hat, height_hat) using (x_min, y_min) and (x_max, y_max)
        3. Values (x_hat, y_hat, width_hat, height_hat) are divided by the width and the height of the image.

        @param image: Input image
        @param predicted: List of predictions
        @return: List of predictions with corrected coordinates
        """

        corrected_result = []

        image_width = len(image[0])
        image_height = len(image)

        for x_min, y_min, x_max, y_max, confidence, predicted_class in predicted:
            width_hat = np.abs(x_max - x_min)
            height_hat = np.abs(y_max - y_min)

            x_hat = x_min + width_hat // 2
            y_hat = y_min + height_hat // 2

            # Values (x_hat, y_hat, width_hat, height_hat) are divided by the width and the height of the image.
            x_hat = x_hat / image_width
            y_hat = y_hat / image_height
            width_hat = width_hat / image_width
            height_hat = height_hat / image_height

            corrected_result.append([x_hat, y_hat, width_hat, height_hat, confidence])

        corrected_result = np.array(corrected_result)
        return corrected_result

    def predict(self, image: npt.NDArray, convert_coords: bool = True) -> npt.NDArray:
        """
        Function predicts the bounding boxes.
        @param image: Input image
        @param convert_coords: Convert predicted coordinates or not.
        @return: List of predictions
        """

        # Make a prediction
        result = self.model(image)
        result = np.array(result.xyxy[0].tolist())

        if convert_coords:
            result = YOLO.convert_prediction_coordinates(image, result)

        return result

    def __str__(self):
        return "YOLO"

