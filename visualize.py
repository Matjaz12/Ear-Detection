import os
import pickle

import numpy as np
import numpy.typing as npt
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_detection(image: npt.NDArray, ground_truth: npt.NDArray, predicted_bounding_boxes: npt.NDArray,
                   figure_name: str = "example") -> None:
    """
    Function displays the ground truth bounding box and all the predicted bounding boxes.

    @param image: Input image
    @param ground_truth: Location of the ground truth bounding box
    @param predicted_bounding_boxes: Location of predicted bounding boxes
    @param figure_name: Name of the figure to be saved.
    """

    # plt.style.use(['science','ieee', 'notebook'])

    # Compute coordinates of ground truth box and display it
    (x, y, width, height) = ground_truth
    x = x * len(image[0])
    y = y * len(image)
    width = width * len(image[0])
    height = height * len(image)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")

    # Note that x, y are the center of the bounding box therefore we compute the upper-left coordinates
    x_upper_left = x - width // 2
    y_upper_left = y - height // 2
    ax.add_patch(Rectangle((x_upper_left, y_upper_left), width, height, fill=None, color="green", lw=1))

    # Display all predicted bounding boxes
    for (x_hat, y_hat, width_hat, height_hat, confidence) in predicted_bounding_boxes:
        x_hat = x_hat * len(image[0])
        y_hat = y_hat * len(image)
        width_hat = width_hat * len(image[0])
        height_hat = height_hat * len(image)

        # Note that x_hat, y_hat are the center of the bounding box therefore we compute the upper-left coordinates
        x_hat_upper_left = x_hat - width_hat // 2
        y_hat_upper_left = y_hat - height_hat // 2

        ax.add_patch(Rectangle((x_hat_upper_left, y_hat_upper_left), width_hat, height_hat, fill=None, color="red", lw=1))
        ax.text(x_hat_upper_left, y_hat_upper_left - 10, f"{np.round(confidence, 2)}",  color="red", fontsize=12, weight="bold")

    plt.title(figure_name)
    plt.savefig(f"./img/{figure_name}.png")
    plt.show()
