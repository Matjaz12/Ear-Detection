import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from load_data import load_data_pickle
from viola_jones import ViolaJones
from visualize import show_detection
from yolo import YOLO


def intersection_over_union(ground_truth: npt.NDArray, prediction: npt.NDArray,
                            image_height: int, image_width: int) -> float:
    # https://www.google.com/search?channel=fs&client=ubuntu&q=python+interesection+over+union#fpstate=ive&vld=cid:fe6ebfe5,vid:XXYG5ZWtjj0

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
    predictions_iou_s = []

    for prediction in predictions:
        sample_idx = prediction[0]
        ground_truth = ground_truths[np.where(ground_truths[:, 0] == sample_idx)][0]
        image_height, image_width = images[int(sample_idx)].shape
        iou = intersection_over_union(ground_truth, prediction, image_height, image_width)
        predictions_iou_s.append(np.array([sample_idx, prediction, iou]))

    predictions_iou_s = np.array(predictions_iou_s)
    return predictions_iou_s


def mean_intersection_over_union(images: npt.NDArray, ground_truths: npt.NDArray,
                                 predictions: npt.NDArray) -> float:
    predictions_iou_s = intersection_over_union_vector(images, ground_truths, predictions)
    mean_iou = np.sum(predictions_iou_s[:, 2]) / predictions_iou_s.shape[0]

    return mean_iou


def precision_recall_curve(images: npt.NDArray, ground_truths: npt.NDArray,
                           predictions: npt.NDArray, iou_threshold: float = 0.5,
                           figure_title: str = ""):

    iou_vector = intersection_over_union_vector(images, ground_truths, predictions)
    confidence_vector = np.expand_dims(predictions[:, 2], axis=1)

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

    plt.style.use(['science','ieee', 'notebook', 'grid'])
    plt.figure(figsize=(10, 10))
    plt.plot(recall_vector.flatten(), precision_vector.flatten(), "-", color="purple")
    plt.title(figure_title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"./img/{figure_title}.png")
    plt.show()


if __name__ == "__main__":
    X_test, y_test = load_data_pickle("./ear_data/X_test_and_y_test.pickle")

    # Viola Jones
    """
    viola_jones = ViolaJones("./weights/haarcascade_mcs_rightear.xml",
                             min_neighbour=3)
    predictions = viola_jones.predict(X_test)

    # Show ground truth and prediction for a random sample.
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

    mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
    print(f"mean_iou: {mean_iou}")
    """

    # YOLO
    yolo = YOLO("./weights/yolo5s.pt")
    predictions = yolo.predict(X_test)
    mean_iou = mean_intersection_over_union(X_test, y_test, predictions)
    print(f"mean_iou: {mean_iou}")

    precision_recall_curve(X_test, y_test, predictions, figure_title="yolo_pr_curve")