import cv2
import numpy as np

from evaluation import intersection_over_union, mean_intersection_over_union, intersection_over_union_vector
from load_data import *
import matplotlib.pyplot as plt
from visualize import show_detection, plot_selected_samples
from viola_jones import ViolaJones
from yolo import YOLO
from enum import Enum


class Task(Enum):
    VJ_MEAN_IOU = 0
    YOLO_MEAN_IOU = 1
    VJ_SHOW_RANDOM_PREDICTION = 2
    YOLO_SHOW_RANDOM_PREDICTION = 3
    VJ_FINE_TUNE = 4
    VJ_SHOW_BEST_PREDICTIONS = 5
    YOLO_SHOW_BEST_PREDICTIONS = 6
    VJ_FAILED_PREDICTIONS = 7


class TaskRunner:
    @staticmethod
    def run(task: Task) -> None:
        X_test, y_test = load_data_pickle("./ear_data/X_test_and_y_test.pickle")

        if task == Task.VJ_MEAN_IOU:
            viola_jones = ViolaJones("./weights/haarcascade_mcs_rightear.xml", min_neighbour=3)
            # viola_jones = ViolaJones("./weights/haarcascade_mcs_leftear.xml", min_neighbour=3)

            predictions = viola_jones.predict(X_test)
            mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
            print(f"mean_iou: {mean_iou}")

        elif task == Task.YOLO_MEAN_IOU:
            yolo = YOLO("./weights/yolo5s.pt")
            predictions = yolo.predict(X_test)
            mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
            print(f"mean_iou: {mean_iou}")

        elif task == Task.VJ_SHOW_RANDOM_PREDICTION:
            # Show ground truth and prediction for a random sample.
            viola_jones = ViolaJones("./weights/haarcascade_mcs_rightear.xml",
                                     min_neighbour=3)
            predictions = viola_jones.predict(X_test)

            index_to_show = int(np.random.choice(predictions[:, 0]))
            print(f"index_to_show: {index_to_show}")
            pred = predictions[np.where(predictions[:, 0] == index_to_show)]

            # Print IoU
            image_height, image_width = X_test[index_to_show].shape
            for curr_pred in pred:
                iou = intersection_over_union(y_test[index_to_show], curr_pred, image_height, image_width)
                print(f"IoU: {iou}")

            # Display sample
            show_detection(X_test[index_to_show],
                           y_test[index_to_show],
                           pred,
                           figure_name=str(viola_jones))

        elif task == Task.YOLO_SHOW_RANDOM_PREDICTION:
            yolo = YOLO("./weights/yolo5s.pt")
            predictions = yolo.predict(X_test)

            index_to_show = int(np.random.choice(predictions[:, 0]))
            print(f"index_to_show: {index_to_show}")
            pred = predictions[np.where(predictions[:, 0] == index_to_show)]

            # Print IoU
            image_height, image_width = X_test[index_to_show].shape
            for curr_pred in pred:
                iou = intersection_over_union(y_test[index_to_show], curr_pred, image_height, image_width)
                print(f"IoU: {iou}")

            # Display sample
            show_detection(X_test[index_to_show],
                           y_test[index_to_show],
                           pred,
                           figure_name=str(yolo))

        elif task == Task.VJ_SHOW_BEST_PREDICTIONS:
            viola_jones = ViolaJones("./weights/haarcascade_mcs_rightear.xml",
                                     min_neighbour=3)
            predictions = viola_jones.predict(X_test)

            iou_s = intersection_over_union_vector(X_test, y_test, predictions)
            iou_s_sorted = iou_s[iou_s[:, 2].argsort()[::-1]]

            TOP_N = 6
            plot_selected_samples(X_test, y_test, predictions,
                                  iou_s_sorted[0:TOP_N], 2, 3,
                                  figure_name="vj_top6_predictions")

        elif task == Task.YOLO_SHOW_BEST_PREDICTIONS:
            yolo = YOLO("./weights/yolo5s.pt")
            predictions = yolo.predict(X_test)
            iou_s = intersection_over_union_vector(X_test, y_test, predictions)
            iou_s_sorted = iou_s[iou_s[:, 2].argsort()[::-1]]

            TOP_N = 6
            plot_selected_samples(X_test, y_test, predictions,
                                  iou_s_sorted[0:TOP_N], 2, 3,
                                  figure_name="yolo_top6_predictions")

        elif task == Task.VJ_FAILED_PREDICTIONS:
            viola_jones = ViolaJones("./weights/haarcascade_mcs_rightear.xml",
                                     min_neighbour=3)
            predictions = viola_jones.predict(X_test)

            iou_s = intersection_over_union_vector(X_test, y_test, predictions)
            iou_s_sorted = iou_s[iou_s[:, 2].argsort()]

            TOP_N = 10
            plot_selected_samples(X_test, y_test, predictions,
                                  iou_s_sorted[0:TOP_N], 5, 2,
                                  figure_name="vj_worst6_predictions")


if __name__ == "__main__":
    TaskRunner.run(Task.YOLO_MEAN_IOU)