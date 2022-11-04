import cv2
import numpy as np
import numpy.typing as npt


class ViolaJones:
    def __init__(self, cascade_path: str, scale_factor: float = None, min_neighbour: float = None,
                 min_size: float = None, max_size: float = None):

        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbour = min_neighbour
        self.min_size = min_size
        self.max_size = max_size

    @staticmethod
    def convert_prediction_coordinates(image: npt.NDArray, predicted: npt.NDArray):
        """
        Function converts the predicted bounding box in format which is used in the test samples.

        predicted (x_hat, y_hat) are the coordinates of the upper-left most pixel.
        predicted (width_hat, height_hat) are the width and the height of the predicted bounding box.

        We perform the following two conversion:
        1. Upper-left most pixel coordinates (x_hat, y_hat) are converted to the coordinates of the central pixel.
        2. Values (x_hat, y_hat, width_hat, height_hat) are divided by the width and the height of the image.

        @param image: Input image
        @param predicted: List of predicted bounding boxes.
        @return: List of predicted bounding boxes with corrected coordinates

        @param image: Input image
        @param result: List of predictions
        @return: List of predictions with corrected coordinates
        """

        predicted_bounding_boxes_corrected = []

        image_width = len(image[0])
        image_height = len(image)

        for (x_hat, y_hat, width_hat, height_hat, confidence) in predicted:
            # 1. Upper-left most pixel coordinates (x_hat, y_hat) are converted to the coordinates of the central pixel.
            x_hat = x_hat + width_hat // 2
            y_hat = y_hat + height_hat // 2

            # Values (x_hat, y_hat, width_hat, height_hat) are divided by the width and the height of the image.
            x_hat = x_hat / image_width
            y_hat = y_hat / image_height
            width_hat = width_hat / image_width
            height_hat = height_hat / image_height

            predicted_bounding_boxes_corrected.append([x_hat, y_hat, width_hat, height_hat, confidence])

        predicted_bounding_boxes_corrected = np.array(predicted_bounding_boxes_corrected)

        return predicted_bounding_boxes_corrected

    def predict(self, image: npt.NDArray, convert_coords: bool = True) -> npt.NDArray:
        """
        Function predicts the bounding boxes.
        @param image: Input image
        @param convert_coords: Convert predicted coordinates or not.
        @return: List of predicted bounding boxes
        """

        predicted_bounding_boxes = self.cascade.detectMultiScale(image, scaleFactor=self.scale_factor,
                                                                 minNeighbors=self.min_neighbour,
                                                                 minSize=self.min_size,
                                                                 maxSize=self.max_size)
        # Add confidence to predictions
        if self.min_neighbour:
            # todo: here i normalize with 10, i just assumed that 10 is the max value, decide how to handle this !
            confidence = self.min_neighbour / 10
        else:
            confidence = -1.0

        predicted_bounding_boxes = np.concatenate(
            [predicted_bounding_boxes, np.full((predicted_bounding_boxes.shape[0], 1), confidence)], axis=1
        )

        if convert_coords:
            predicted_bounding_boxes = ViolaJones.convert_prediction_coordinates(image, predicted_bounding_boxes)

        return predicted_bounding_boxes

    def __str__(self):
        return "ViolaJones"
