from collections import OrderedDict, namedtuple
from datetime import datetime
from enum import Enum
from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from double_viola_jones import DoubleViolaJones
from evaluation import (intersection_over_union,
                        intersection_over_union_vector,
                        mean_accuracy_precision, mean_intersection_over_union,
                        plot_precision_recall_curve,
                        plot_precision_recall_curves,
                        precision_recall_curve_all_thresholds,
                        precision_recall_curve_fixed_threshold)
from load_data import *
from viola_jones import ViolaJones
from visualize import plot_selected_samples, show_detection
from yolo import YOLO


class Task(Enum):
    VJ_MEAN_IOU = 0
    VJ_DOUBLE_MEAN_IOU = 1
    YOLO_MEAN_IOU = 2
    VJ_SHOW_RANDOM_PREDICTION = 3
    YOLO_SHOW_RANDOM_PREDICTION = 4
    VJ_FINE_TUNE = 5
    VJ_SHOW_BEST_PREDICTIONS = 6
    YOLO_SHOW_BEST_PREDICTIONS = 7
    VJ_SHOW_FAILED_PREDICTIONS = 8
    VJ_DOUBLE_FINE_TUNE = 9
    PR_CURVE = 10
    MAP_TABLE = 11
    MAP_TABLE_HYPER_PARAMS = 12


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
        IMAGE_MODE = "GRAY"
        LEFT_CASCADE = "./weights/haarcascade_mcs_rightear.xml"
        RIGHT_CASCADE = "./weights/haarcascade_mcs_leftear.xml"
        YOLO_WEIGHTS = "./weights/yolo5s.pt"

        X_test, y_test = load_data_pickle(f"./ear_data/X_test_and_y_test_{IMAGE_MODE}.pickle")

        if task == Task.VJ_MEAN_IOU:
            # viola_jones = ViolaJones(LEFT_CASCADE)
            viola_jones = ViolaJones(RIGHT_CASCADE)

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
            viola_jones = DoubleViolaJones(RIGHT_CASCADE, LEFT_CASCADE)
            predictions = viola_jones.predict(X_test)
            mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
            print(f"mean_iou: {mean_iou}")

        elif task == Task.VJ_SHOW_RANDOM_PREDICTION:
            # Show ground truth and prediction for a random sample.
            viola_jones = DoubleViolaJones(RIGHT_CASCADE, LEFT_CASCADE)
            predictions = viola_jones.predict(X_test)

            index_to_show = int(np.random.choice(predictions[:, 0]))
            print(f"index_to_show: {index_to_show}")
            pred = predictions[np.where(predictions[:, 0] == index_to_show)]

            # Print IoU
            image_height, image_width = X_test[index_to_show].shape[0], X_test[index_to_show].shape[1]
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
            image_height, image_width = X_test[index_to_show].shape[0], X_test[index_to_show].shape[1]
            for curr_pred in pred:
                iou = intersection_over_union(y_test[index_to_show], curr_pred, image_height, image_width)
                print(f"IoU: {iou}")

            # Display sample
            show_detection(X_test[index_to_show],
                           y_test[index_to_show],
                           pred,
                           figure_name=str(yolo),
                           save=True)

        elif task == Task.VJ_SHOW_BEST_PREDICTIONS:
            viola_jones = DoubleViolaJones(RIGHT_CASCADE,
                                        LEFT_CASCADE,
                                        scale_factor=1.05,
                                        min_neighbour=1)
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

        elif task == Task.VJ_SHOW_FAILED_PREDICTIONS:
            viola_jones = DoubleViolaJones(RIGHT_CASCADE,
                                        LEFT_CASCADE,
                                        scale_factor=1.05,
                                        min_neighbour=1)

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
                min_neighbors=[1, 2, 3, 4, 5, 6, 7],
                min_size=[None]
            )

            run_data = []
            run_time = datetime.now()
            run_params = RunBuilder.get_runs(hyper_parameters)

            # Try each combination of parameters
            print(f"Running {len(run_params)} experiments...")

            for param_set in tqdm(run_params):
                viola_jones = ViolaJones(cascade_path=RIGHT_CASCADE,
                                         scale_factor=param_set.scale_factor,
                                         min_neighbour=param_set.min_neighbors,
                                         min_size=param_set.min_size)

                predictions = viola_jones.predict(X_test)
                mean_iou = mean_intersection_over_union(X_test, y_test, predictions)

                mAP = mean_accuracy_precision(X_test, y_test, predictions, iou_threshold_start=0.5,
                        iou_threshold_step=0.05, iou_threshold_stop=0.95)

                print(f"mean_iou: {mean_iou}")
                print(f"mAP: {mAP}")

                results = OrderedDict()
                results["method"] = str(viola_jones)
                results["scale_factor"] = param_set.scale_factor
                results["min_neighbors"] = param_set.min_neighbors
                results["min_size"] = param_set.min_size
                results["mean_iou"] = mean_iou
                results["mAP"] = mAP

                run_data.append(results)

            run_data_df = pd.DataFrame.from_dict(run_data, orient="columns")
            run_data_df = run_data_df.sort_values("mAP", ascending=False)
            print(run_data_df)

            run_data_df.to_csv(f"./results/{viola_jones}_min_size_run_{str(run_time).replace(' ', '_')}", index=False)

        elif task == Task.VJ_DOUBLE_FINE_TUNE:
            hyper_parameters = OrderedDict(
                scale_factor=[1.05, 1.1, 1.2, 1.3],
                min_neighbors=[1, 2, 3, 4, 5, 6, 7],
                min_size=[None]
            )
            run_data = []
            run_time = datetime.now()
            run_params = RunBuilder.get_runs(hyper_parameters)

            # Try each combination of parameters
            print(f"Running {len(run_params)} experiments...")

            for param_set in tqdm(run_params):
                viola_jones = DoubleViolaJones(RIGHT_CASCADE,
                                               LEFT_CASCADE,
                                               scale_factor=param_set.scale_factor,
                                               min_neighbour=param_set.min_neighbors)

                predictions = viola_jones.predict(X_test)
                mean_iou = mean_intersection_over_union(X_test, y_test, predictions)

                mAP = mean_accuracy_precision(X_test, y_test, predictions, iou_threshold_start=0.0,
                iou_threshold_step=0.01, iou_threshold_stop=1.0)

                print(f"mean_iou: {mean_iou}")
                print(f"mAP: {mAP}")

                results = OrderedDict()
                results["method"] = str(viola_jones)
                results["scale_factor"] = param_set.scale_factor
                results["min_neighbors"] = param_set.min_neighbors
                results["mean_iou"] = np.round(mean_iou, 3)
                results["mAP"] = np.round(mAP, 3)

                run_data.append(results)

            run_data_df = pd.DataFrame.from_dict(run_data, orient="columns")
            run_data_df = run_data_df.sort_values("mAP", ascending=False)
            print(run_data_df)

            run_data_df.to_csv(f"./results/{viola_jones}_run_{str(run_time).replace(' ', '_')}", index=False)

        elif task == Task.PR_CURVE:
            hyper_parameters=OrderedDict(
                model=[YOLO(YOLO_WEIGHTS), DoubleViolaJones(RIGHT_CASCADE, LEFT_CASCADE)],
                iou_threshold=[0.5]
            )

            r_vects, p_vects, labels = [], [], []
            run_params = RunBuilder.get_runs(hyper_parameters)

            print(f"Plotting {len(run_params)} PR curves...")
            for param_set in run_params:
                predictions = param_set.model.predict(X_test)
                r_vect, p_vect = precision_recall_curve_fixed_threshold(X_test, y_test, 
                                                                        predictions, param_set.iou_threshold)
                
                r_vects.append(r_vect)
                p_vects.append(p_vect)
                labels.append(str(param_set.model))
            

            plot_precision_recall_curves(r_vects, p_vects, 
                                        labels, figure_title="Precision Recal Curve (IoU Threshold=0.5)")

        elif task == Task.MAP_TABLE:
            hyper_parameters=OrderedDict(
                model=[YOLO(YOLO_WEIGHTS)],
                iou_threshold=[0.5]
            )

            results = []
            mAPs, labels = [], []
            run_params = RunBuilder.get_runs(hyper_parameters)

            for param_set in run_params:
                predictions = param_set.model.predict(X_test)
                mAP = mean_accuracy_precision(X_test, y_test, predictions, iou_threshold_start=0.0,
                                        iou_threshold_step=0.01, iou_threshold_stop=1.0)

                result = OrderedDict()
                result["mAP"] = mAP
                result["model"] = str(param_set.model)

                mAPs.append(mAP)    
                results.append(result)

            results_df = pd.DataFrame.from_dict(results, orient="columns")
            results_df = results_df.sort_values("mAP", ascending=False)

            print(results_df)
            results_df.to_csv(f"./results/mAP_table", index=False)

if __name__ == "__main__":
    TaskRunner.run(Task.VJ_SHOW_BEST_PREDICTIONS)