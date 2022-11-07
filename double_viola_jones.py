import cv2
import numpy as np
import numpy.typing as npt
from numba import njit

from load_data import load_data_pickle
from visualize import show_detection


class DoubleViolaJones:
    def __init__(self, cascade1_path: str, cascade2_path: str, scale_factor: float = None, min_neighbour: float = None,
                 min_size: float = None, max_size: float = None):

        self.cascade1 = cv2.CascadeClassifier(cascade1_path)
        self.cascade2 = cv2.CascadeClassifier(cascade2_path)

        self.scale_factor = scale_factor
        self.min_neighbour = min_neighbour
        self.min_size = min_size
        self.max_size = max_size

    @staticmethod
    def convert_detection(image_height: int, image_width: int, detection: npt.NDArray) -> npt.NDArray:
        """
        Convert detection by cv2 viola jones algorithm
        @param image_height: height of the input image
        @param image_width: width of the input image
        @param detection: Detection returned by cv2 viola jones algorithm,
            [x_up, y_up, box_width, box_height]
        @return:
            Detection in the following format
            [x_center, y_center, box_width, box_height]
        """
        x_up, y_up, box_width, box_height = detection

        x_center = x_up + box_width // 2
        y_center = y_up + box_height // 2
        x_center /= image_width
        y_center /= image_height
        box_width /= image_width
        box_height /= image_height

        converted_detection = np.array([x_center, y_center, box_width, box_height])
        return converted_detection

    def predict(self, images: npt.NDArray, convert_coordinates: bool = True) -> npt.NDArray:
        """
        Function predicts the bounding boxes.
        @param images: Input images
        @param convert_coordinates: Convert predicted coordinates or not.
        @return: List of predictions:
            [[sample_idx, predicted_class, predicted_prob, bounding_box], ...]
        """

        # Compute detections for all images
        predictions = []
        for sample_idx, image in enumerate(images):

            # Apply cascade 1
            detections1 = self.cascade1.detectMultiScale(image, scaleFactor=self.scale_factor,
                                                         minNeighbors=self.min_neighbour, minSize=self.min_size,
                                                         maxSize=self.max_size)
            # Apply cascade 2
            detections2 = self.cascade2.detectMultiScale(image, scaleFactor=self.scale_factor,
                                                         minNeighbors=self.min_neighbour, minSize=self.min_size,
                                                         maxSize=self.max_size)

            if len(detections1) > 0 and len(detections2) > 0:
                detections = np.concatenate(
                    (detections1, detections2),
                    axis=0
                )

            elif len(detections1) > 0:
                detections = detections1

            else:
                detections = detections2

            # Iterate over detections and construct prediction vector
            for detection in detections:
                predicted_class = 0
                predicted_prob = self.min_neighbour if self.min_neighbour is not None else -1.0
                prediction = [sample_idx, predicted_class, predicted_prob]

                if convert_coordinates:
                    image_height, image_width = image.shape
                    detection = DoubleViolaJones.convert_detection(image_height, image_width, detection)

                prediction.extend(detection)

                # Store prediction
                predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions

    def __str__(self):
        return "DoubleViolaJones"


if __name__ == "__main__":
    X_test, y_test = load_data_pickle("./ear_data/X_test_and_y_test.pickle")
    viola_jones = DoubleViolaJones("./weights/haarcascade_mcs_rightear.xml",
                                   "./weights/haarcascade_mcs_leftear.xml",
                                   min_neighbour=3)
    predictions = viola_jones.predict(X_test)

    # Show ground truth and prediction for a random sample.
    index_to_show = int(np.random.choice(predictions[:, 0]))
    pred = predictions[np.where(predictions[:, 0] == index_to_show)]
    show_detection(X_test[index_to_show],
                   y_test[index_to_show],
                   pred,
                   figure_name=str(viola_jones))
