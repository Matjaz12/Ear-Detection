from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from load_data import load_data_pickle
from viola_jones import ViolaJones
from double_viola_jones import DoubleViolaJones
from visualize import show_detection
from yolo import YOLO
from collections import Counter


def intersection_over_union(ground_truth: npt.NDArray, prediction: npt.NDArray,
                            image_height: int, image_width: int) -> float:
    """
    Function computes intersection over union given ground truth annotation
    and a prediction.

    @param ground_truth: Ground truth (single groundtruth).
    @param prediction: Predicted bounding box (single predicted sample).
    @param image_height: Height of the image.
    @param image_width: Width of the image.
    @return: intersection over union for the given prediction.
    """

    sample_idx, true_class, true_prob, x_center, y_center, box_width, box_height = ground_truth
    x_center *= image_width
    y_center *= image_height
    box_width *= image_width
    box_height *= image_height

    sample_idx, class_hat, prob_hat, x_center_hat, y_center_hat, box_width_hat, box_height_hat = prediction
    x_center_hat *= image_width
    y_center_hat *= image_height
    box_width_hat *= image_width
    box_height_hat *= image_height

    x1, y1 = x_center - box_width // 2, y_center - box_height // 2
    x2, y2 = x_center + box_width // 2, y_center + box_height // 2

    x1_hat, y1_hat = x_center_hat - box_width_hat // 2, y_center_hat - box_height_hat // 2
    x2_hat, y2_hat = x_center_hat + box_width_hat // 2, y_center_hat + box_height_hat // 2

    x1_inter = max(x1, x1_hat)
    y1_inter = max(y1, y1_hat)
    x2_inter = min(x2, x2_hat)
    y2_inter = min(y2, y2_hat)

    width_inter = (x2_inter - x1_inter) if (x2_inter - x1_inter) > 0 else 0
    height_inter = (y2_inter - y1_inter) if (y2_inter - y1_inter) > 0 else 0

    intersection = width_inter * height_inter
    union = box_width * box_height + box_width_hat * box_height_hat - intersection

    return intersection / (union + 1e-6)


def intersection_over_union_vector(images: npt.NDArray, ground_truths: npt.NDArray,
                                   predictions: npt.NDArray) -> npt.NDArray:
    """
    Function computes intersection over union for a set of provided
    predictions.

    @param images: Images on which predictions were made.
    @param ground_truths: Ground truth annotations for each image.
    @param predictions: A set of predictions
    @return: Vector containing each prediction and its corresponding
        intersection over union.
        [sample_idx, prediction, iou]
    """
    p_iou = []

    for prediction in predictions:
        sample_idx = prediction[0]
        ground_truth = ground_truths[np.where(ground_truths[:, 0] == sample_idx)][0]
        image_height, image_width = images[int(sample_idx)].shape[0], images[int(sample_idx)].shape[1]
        iou = intersection_over_union(ground_truth, prediction, image_height, image_width)
        p_iou.append(np.array([sample_idx, prediction, iou]))

    p_iou = np.array(p_iou)
    return p_iou


def mean_intersection_over_union(images: npt.NDArray, ground_truths: npt.NDArray,
                                 predictions: npt.NDArray) -> float:
    """
    Function computes the mean intersection over union (mIoU).

    @param images: Images on which predictions were made.
    @param ground_truths: Ground truth annotations for each image.
    @param predictions: A set of predictions.
    @return: mean intersection over union.
    """

    p_iou = intersection_over_union_vector(images, ground_truths, predictions)
    mean_iou = np.sum(p_iou[:, 2]) / p_iou.shape[0]

    return mean_iou


def plot_precision_recall_curve(recall_vector:npt.NDArray, precision_vector: npt.NDArray, 
                                figure_title: str = "") -> None:
    """
    Function plots the precision recall curve.
    
    @param: recall_vector: Vector holding recall values.
    @param: presicion_vector: Vector holding precision values.
    @param: figure_title: Title of the ploted figure.
    """
    
    plt.style.use(['science','ieee', 'notebook', 'grid'])
    plt.figure(figsize=(10, 10))
    plt.plot(recall_vector, precision_vector, "-", color="blue")
    plt.title(figure_title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    if figure_title != "":
        plt.savefig(f"./results/{figure_title}.png")

    plt.show()


def plot_precision_recall_curves(recall_vectors: npt.NDArray, precision_vectors: npt.NDArray,
                                labels: npt.NDArray, figure_title: str = "") -> None:

    """
    Function plots the precision recall curves
    
    @param: recall_vector: Vector of recall vectors.
    @param: presicion_vector: Vector of precision vectors.
    @param labels: Corresponding lables for each recall and precision vector.
    @param: figure_title: Title of the ploted figure.
    """

    plt.style.use(['science','ieee', 'notebook', 'grid'])
    plt.figure(figsize=(15, 15))
    
    for recall_vector, precision_vector, label in zip(recall_vectors, precision_vectors, labels):
        auc = np.abs(np.trapz(recall_vector, precision_vector))

        plt.plot(recall_vector, precision_vector, "-", label=f"{label}, AUC: {np.round(auc, 2)}")
        plt.title(figure_title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")


    plt.legend(loc="lower right")
    if figure_title != "":
        plt.savefig(f"./results/{figure_title}.png")


def precision_recall_curve_fixed_threshold(images: npt.NDArray, ground_truths: npt.NDArray,
                            predictions: npt.NDArray, iou_threshold: float = 0.5) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Function computes the precision-recall curve for a specified threshold
    `iou_threshold`. 
    
    It utilizes the confidence score associated with each
    prediction returned by detection models.

    @param images: Images on which predictions were made.
    @param ground_truths: Ground truth annotations for each image.
    @param predictions: A set of predictions.
    @param iou_threshold: Threshold used to classify a prediction as a TP or FP.
    @return: recall and precision vectors.
    """

    iou_vector = intersection_over_union_vector(images, ground_truths, predictions)
    confidence_vector = np.expand_dims(predictions[:, 2], axis=1)

    """
    plt.hist(Counter(predictions[:, 2]))
    plt.title("Confidence histogram")
    plt.plot()
    """

    iou_conf_vector = np.concatenate(
        (iou_vector, confidence_vector), axis=1
    )

    # drop the bounding box
    iou_conf_vector = np.delete(iou_conf_vector, 1, axis=1)

    # Sort by confidence
    iou_conf_vector = iou_conf_vector[iou_conf_vector[:, -1].argsort()[::-1]]

    # Use iou_threshold to determine FP or TP
    tp_vector = np.expand_dims(iou_conf_vector[:, 1] >= iou_threshold, axis=1)
    fp_vector = np.logical_not(tp_vector)

    iou_conf_vector = np.concatenate(
        (iou_conf_vector, tp_vector),
        axis=1
    )

    iou_conf_vector = np.concatenate(
        (iou_conf_vector, fp_vector),
        axis=1
    )

    tp_csum_vector = np.expand_dims(np.cumsum(iou_conf_vector[:, -2]), axis=1)
    fp_csum_vector = np.expand_dims(np.cumsum(iou_conf_vector[:, -1]), axis=1)

    iou_conf_vector = np.concatenate(
        (iou_conf_vector, tp_csum_vector),
        axis=1
    )

    iou_conf_vector = np.concatenate(
        (iou_conf_vector, fp_csum_vector),
        axis=1
    )

    precision_vector = tp_csum_vector / (tp_csum_vector + fp_csum_vector)
    recall_vector = tp_csum_vector / ground_truths.shape[0]

    iou_conf_vector = np.concatenate(
        (iou_conf_vector, precision_vector),
        axis=1
    )

    iou_conf_vector = np.concatenate(
        (iou_conf_vector, recall_vector),
        axis=1
    )

    return recall_vector.flatten(), precision_vector.flatten()


def precision_recall_curve_all_thresholds(images: npt.NDArray, ground_truths: npt.NDArray,
                            predictions: npt.NDArray, iou_threshold_step: float = 0.01,
                            iou_threshold_start: float = 0.5, iou_threshold_stop: float = 0.95) -> None:
    """
    Function plots the precision-recall curve for each threshold in range
    0.0 to 1.0 using step size of `iou_threshold_step`.

    @param images: Images on which predictions were made.
    @param ground_truths: Ground truth annotations for each image.
    @param predictions: A set of predictions.
    @param iou_threshold_step: Delta between two consecutive threshold values.
    @param iou_threshold_start: Starting value for iou threshold.
    @param iou_threshold_stop: Stopping value for iou threshold.
    """

    recall_vector, precision_vector = [], []

    iou_vector = intersection_over_union_vector(images, ground_truths, predictions)[:, -1]

    for iou_threshold in np.arange(0.0, 1.0, iou_threshold_step):
        # iou_threshold = 1 - iou_threshold

        num_tps = np.sum(iou_vector >= iou_threshold)
        num_fp = np.sum(iou_vector < iou_threshold)

        precision = num_tps / (num_tps + num_fp + 1e-8)
        recall = num_tps / ground_truths.shape[0]

        """
        if iou_threshold == 0.2 or iou_threshold == 0.5 or iou_threshold == 0.6 or iou_threshold== 0.8 or iou_threshold == 0.9:
            print(f"iou_threshold: {iou_threshold}, precision: {precision}, recall: {recall}")
        """

        precision_vector.append(precision)
        recall_vector.append(recall)

    precision_vector = np.array(precision_vector)
    recall_vector = np.array(recall_vector)

    return recall_vector, precision_vector


def mean_accuracy_precision_for_threshold(images: npt.NDArray, ground_truths: npt.NDArray,
                             predictions: npt.NDArray, iou_threshold: float) -> float:
    """
    Function computes mean accuracy precision (mAP_t), which is defined as
    the area under the precision-recall curve (computed for a given threshold)

    @param images: Images on which predictions were made.
    @param ground_truths: Ground truth annotations for each image.
    @param predictions: A set of predictions.
    @param iou_threshold: Threshold used to classify a prediction as a TP or FP.
    """

    recall_vector, precision_vector = precision_recall_curve_fixed_threshold(images, ground_truths,
                                                            predictions, iou_threshold)

    # Compute the area under the precision-recall curve for given threshold.
    mAP_t = np.trapz(precision_vector, recall_vector)
    return mAP_t


def mean_accuracy_precision(images: npt.NDArray, ground_truths: npt.NDArray,
                            predictions: npt.NDArray, iou_threshold_step: float,
                            iou_threshold_start: float = 0.5, iou_threshold_stop: float = 0.95) -> float:
    """
    Function computes the mean accuracy precision (mAP) which is the defined as
    the mean of mAP_t's for all thesholds from:
         iou_threshold_start: iou_threshold_step: iou_threshold_stop

    @param images: Images on which predictions were made.
    @param ground_truths: Ground truth annotations for each image.
    @param predictions: A set of predictions.
    @param iou_threshold_step: Delta between two consecutive threshold values.
    @param iou_threshold_start: Starting value for iou threshold.
    @param iou_threshold_stop: Stopping value for iou threshold.
    """

    mAP_ts = []
    for iou_threshold in np.arange(iou_threshold_start, iou_threshold_stop, iou_threshold_step):
        mAP_ts.append(
            mean_accuracy_precision_for_threshold(images, ground_truths, predictions, iou_threshold)
        )

    mAP = np.mean(mAP_ts)
    return mAP


if __name__ == "__main__":
    X_test, y_test = load_data_pickle("./ear_data/X_test_and_y_test_GRAY.pickle")

    viola_jones = DoubleViolaJones("./weights/haarcascade_mcs_rightear.xml",
                                           "./weights/haarcascade_mcs_leftear.xml")
    predictions = viola_jones.predict(X_test)

    recall, precision = precision_recall_curve_all_thresholds(X_test, y_test, predictions)
    plot_precision_recall_curve(recall, precision)

    """
    yolo = YOLO("./weights/yolo5s.pt")
    predictions = yolo.predict(X_test)
    mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
    print(f"mean_iou: {mean_iou}")

    recall, precision = precision_recall_curve_fixed_threshold(X_test, y_test, predictions, figure_title="yolo_pr_curve")
    plot_precision_recall_curve(recall, precision, figure_title="test_yolo_pr_curve")
    """