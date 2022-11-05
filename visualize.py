import os
import pickle

import numpy as np
import numpy.typing as npt
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from load_data import load_data_pickle


def show_detection(image: npt.NDArray, ground_truth: npt.NDArray, predictions: npt.NDArray,
                   figure_name: str = "example", save: bool = False) -> None:
    """
    Function displays the ground truth bounding box and all the predicted bounding boxes.

    @param image: Input image
    @param ground_truth:
            Ground truth image label.
            [sample_idx, true_class, prob, x_center, y_center, box_width, box_height]
    @param predictions:
            List of predictions
            [[sample_idx, class_hat, prob_hat, x_center_hat, y_center_hat, box_width_hat, box_height_hat], ...]
    @param figure_name: Name of the figure to be saved.
    """

    image_height, image_width = image.shape

    # Compute coordinates of ground truth box and display it
    sample_idx, true_class, prob, x_center, y_center, box_width, box_height = ground_truth
    x_center *= image_width
    y_center *= image_height
    box_width *= image_width
    box_height *= image_height
    # In order to plot bounding box we have to compute
    # the upper-left coordinates denoted with (`x_ul`, `y_ul`)
    x_ul = x_center - box_width // 2
    y_ul = y_center - box_height // 2

    # Plot ground truth
    # plt.style.use(['science','ieee', 'notebook'])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")
    ax.add_patch(Rectangle((x_ul, y_ul), box_width, box_height, fill=None, color="green", lw=1))

    for (sample_idx, class_hat, prob_hat, x_center_hat, y_center_hat, box_width_hat, box_height_hat) in predictions:
        x_center_hat *= image_width
        y_center_hat *= image_height
        box_width_hat *= image_width
        box_height_hat *= image_height

        # In order to plot bounding box we have to compute
        # the upper-left coordinates denoted with (`x_ul_hat`, `y_ul_hat`)
        x_ul_hat = x_center_hat - box_width_hat // 2
        y_ul_hat = y_center_hat - box_height_hat // 2

        ax.add_patch(Rectangle((x_ul_hat, y_ul_hat), box_width_hat, box_height_hat, fill=None, color="red", lw=1))
        ax.text(x_ul_hat, y_ul_hat - 10, f"{np.round(prob_hat, 2)}",  color="red", fontsize=12, weight="bold")

    plt.title(figure_name)
    if save:
        plt.savefig(f"./img/{figure_name}.png")
    plt.show()


def plot_selected_samples(images: npt.NDArray, ground_truths: npt.NDArray,
                          predictions: npt.NDArray, iou_s: npt.NDArray,
                          figure_rows: int, figure_cols: int,
                          figure_name: str = "selected_examples") -> None:

    fig, axes = plt.subplots(figure_rows, figure_cols, figsize=(15, 15))
    fig.tight_layout()

    for idx, ax in enumerate(fig.axes):
        sample_idx, prediction, iou = iou_s[idx]
        image = images[int(sample_idx)]
        ground_truth = ground_truths[np.where(ground_truths[:, 0] == sample_idx)][0]

        # Compute image shape
        image_height, image_width = image.shape

        # Compute coordinates of ground truth box and display it
        sample_idx, true_class, prob, x_center, y_center, box_width, box_height = ground_truth
        x_center *= image_width
        y_center *= image_height
        box_width *= image_width
        box_height *= image_height
        x_ul = x_center - box_width // 2
        y_ul = y_center - box_height // 2

        ax.imshow(image, cmap="gray")
        ax.add_patch(Rectangle((x_ul, y_ul), box_width, box_height, fill=None, color="green", lw=1))

        # Show all predicted bounding boxes
        sample_idx, class_hat, prob_hat, x_center_hat, y_center_hat, box_width_hat, box_height_hat = \
            prediction

        x_center_hat *= image_width
        y_center_hat *= image_height
        box_width_hat *= image_width
        box_height_hat *= image_height
        x_ul_hat = x_center_hat - box_width_hat // 2
        y_ul_hat = y_center_hat - box_height_hat // 2

        ax.add_patch(Rectangle((x_ul_hat, y_ul_hat), box_width_hat, box_height_hat, fill=None, color="red", lw=1))
        ax.text(x_ul_hat, y_ul_hat - 10, f"P: {np.round(prob_hat, 2)}, Iou: {np.round(iou, 2)}", color="red", fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(f"./img/{figure_name}.png")
    plt.show()


if __name__ == "__main__":
    X_test, y_test = load_data_pickle("./ear_data/X_test_and_y_test.pickle")

    for i in range(400, 500):
        show_detection(X_test[i], y_test[i], [], figure_name="ground_truth_example")
