from collections import OrderedDict, namedtuple
from datetime import datetime
from itertools import product

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from double_viola_jones import DoubleViolaJones
from evaluation import intersection_over_union, mean_intersection_over_union, intersection_over_union_vector
from load_data import *
import matplotlib.pyplot as plt
from visualize import show_detection, plot_selected_samples
from viola_jones import ViolaJones
from yolo import YOLO
from enum import Enum


class Task(Enum):
    VJ_MEAN_IOU = 0
    VJ_DOUBLE_MEAN_IOU = 1
    YOLO_MEAN_IOU = 2
    VJ_SHOW_RANDOM_PREDICTION = 3
    YOLO_SHOW_RANDOM_PREDICTION = 4
    VJ_FINE_TUNE = 5
    VJ_SHOW_BEST_PREDICTIONS = 6
    YOLO_SHOW_BEST_PREDICTIONS = 7
    VJ_FAILED_PREDICTIONS = 8
    VJ_DOUBLE_FINE_TUNE = 9


class RunBuilder:
    @staticmethod
    def get_runs(parameters: OrderedDict):
        """
        Function returns all permutations of parameters.
        :param parameters: Dictionary of parameters
        :return: List of all parameter permutations.
        """

        Run = namedtuple("Run", parameters.keys())
        runs = []

        for val in product(*parameters.values()):
            runs.append(Run(*val))

        return runs


class TaskRunner:
    @staticmethod
    def run(task: Task) -> None:
        X_test, y_test = load_data_pickle("./ear_data/X_test_and_y_test.pickle")

        if task == Task.VJ_MEAN_IOU:
            # viola_jones = ViolaJones("./weights/haarcascade_mcs_rightear.xml")
            viola_jones = ViolaJones("./weights/haarcascade_mcs_leftear.xml")

            predictions = viola_jones.predict(X_test)
            mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
            print(f"mean_iou: {mean_iou}")

        elif task == Task.YOLO_MEAN_IOU:
            yolo = YOLO("./weights/yolo5s.pt")
            predictions = yolo.predict(X_test)
            mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
            print(f"mean_iou: {mean_iou}")

        elif task == Task.VJ_DOUBLE_MEAN_IOU:
            X_test, y_test = load_data_pickle("./ear_data/X_test_and_y_test.pickle")

            viola_jones = DoubleViolaJones("./weights/haarcascade_mcs_rightear.xml",
                                           "./weights/haarcascade_mcs_leftear.xml")
            predictions = viola_jones.predict(X_test)
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

        elif task == Task.VJ_FINE_TUNE:
            hyper_parameters = OrderedDict(
                scale_factor=[1.05, 1.1, 1.2, 1.3],
                min_neighbors=[3, 4, 5, 6, 7]
            )

            run_data = []
            run_time = datetime.now()
            run_params = RunBuilder.get_runs(hyper_parameters)

            # Try each combination of parameters
            print(f"Running {len(run_params)} experiments...")

            for param_set in tqdm(run_params):
                viola_jones = ViolaJones(cascade_path="./weights/haarcascade_mcs_leftear.xml",
                                         scale_factor=param_set.scale_factor,
                                         min_neighbour=param_set.min_neighbors)

                predictions = viola_jones.predict(X_test)
                mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
                print(f"mean_iou: {mean_iou}")

                results = OrderedDict()
                results["method"] = str(viola_jones)
                results["scale_factor"] = param_set.scale_factor
                results["min_neighbors"] = param_set.min_neighbors
                results["mean_iou"] = mean_iou

                run_data.append(results)

            run_data_df = pd.DataFrame.from_dict(run_data, orient="columns")
            run_data_df = run_data_df.sort_values("mean_iou", ascending=False)
            print(run_data_df)

            run_data_df.to_csv(f"./results/{viola_jones}_run_{str(run_time).replace(' ', '_')}", index=False)

        elif task == Task.VJ_DOUBLE_FINE_TUNE:
            hyper_parameters = OrderedDict(
                scale_factor=[1.05, 1.1, 1.2, 1.3],
                min_neighbors=[3, 4, 5, 6, 7]
            )

            run_data = []
            run_time = datetime.now()
            run_params = RunBuilder.get_runs(hyper_parameters)

            # Try each combination of parameters
            print(f"Running {len(run_params)} experiments...")

            for param_set in tqdm(run_params):
                viola_jones = DoubleViolaJones("./weights/haarcascade_mcs_rightear.xml",
                                               "./weights/haarcascade_mcs_leftear.xml",
                                               scale_factor=param_set.scale_factor,
                                               min_neighbour=param_set.min_neighbors)

                predictions = viola_jones.predict(X_test)
                mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
                print(f"mean_iou: {mean_iou}")

                results = OrderedDict()
                results["method"] = str(viola_jones)
                results["scale_factor"] = param_set.scale_factor
                results["min_neighbors"] = param_set.min_neighbors
                results["mean_iou"] = mean_iou

                run_data.append(results)

            run_data_df = pd.DataFrame.from_dict(run_data, orient="columns")
            run_data_df = run_data_df.sort_values("mean_iou", ascending=False)
            print(run_data_df)

            run_data_df.to_csv(f"./results/{viola_jones}_run_{str(run_time).replace(' ', '_')}", index=False)


if __name__ == "__main__":
    TaskRunner.run(Task.VJ_DOUBLE_FINE_TUNE)
