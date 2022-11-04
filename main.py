import cv2

from evaluation import intersection_over_union
from load_data import *
import matplotlib.pyplot as plt
from visualize import show_detection
from viola_jones import ViolaJones
from yolo import YOLO


class TaskRunner:
    @staticmethod
    def run(task_number: int) -> None:
        X_test, y_test = load_data_pickle("./ear_data/Xtest_ytest.pickle")

        if task_number == 1:
            X_test_sample = X_test[42]
            y_test_sample = y_test[42]

            viola_jones = ViolaJones("./weights/haarcascade_mcs_rightear.xml",
                                     min_neighbour=3)

            # viola_jones = ViolaJones("./weights/haarcascade_mcs_leftear.xml")
            predicted_bounding_boxes = viola_jones.predict(X_test_sample)
            show_detection(X_test_sample, y_test_sample, predicted_bounding_boxes,
                           figure_name=str(viola_jones))

            iou = intersection_over_union(y_test_sample,
                                          predicted_bounding_boxes[0][:-1],
                                          len(X_test_sample[0]),
                                          len(X_test_sample))
            print(f"IoU: {iou}")

        elif task_number == 2:
            X_test_sample = X_test[42]
            y_test_sample = y_test[42]

            yolo = YOLO("./weights/yolo5s.pt")
            y_hat = yolo.predict(X_test_sample)
            print(y_hat)

            # drop the confidence level
            # y_hat = [prediction[0: -1] for prediction in y_hat]
            show_detection(X_test_sample, y_test_sample, y_hat,
                           figure_name=str(yolo))

            iou = intersection_over_union(y_test_sample,
                                          [prediction[0: -1] for prediction in y_hat][0],
                                          len(X_test_sample[0]),
                                          len(X_test_sample))
            print(f"IoU: {iou}")

        elif task_number == 3:
            pass

        elif task_number == -1:
            # Reference code
            X_test_sample = X_test[42]
            y_test_sample = y_test[42]

            ear_cascade = cv2.CascadeClassifier("./weights/haarcascade_mcs_rightear.xml")
            # ear_cascade = cv2.CascadeClassifier("./weights/haarcascade_mcs_leftear.xml")

            detected_boxes = ear_cascade.detectMultiScale(X_test_sample)
            for (column, row, width, height) in detected_boxes:
                cv2.rectangle(X_test_sample,
                              (column, row),
                              (column + width, row + height),
                              (255, 255, 255),
                              3)
            plt.imshow(X_test_sample, cmap="gray")
            plt.show()


if __name__ == "__main__":
    TaskRunner.run(task_number=2)