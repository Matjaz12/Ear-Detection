import os
import pickle

import numpy as np
import numpy.typing as npt
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_data(data_path: str) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Function loads data and returns tensors X, y.
    @param data_path: Path to data.
    @return: X (samples) and y (labels).
    """
    X, y = [], []
    data = {}

    # Store labels
    for filename in sorted(os.listdir(data_path)):
        if filename.endswith(".png"):
            image_data = Image.open(data_path + "/" + filename).convert("L")
            image = np.array(image_data)

            sample_id = filename.split(".")[0]
            data[sample_id] = [image]

    # Store data samples
    for filename in sorted(os.listdir(data_path)):
        if filename.endswith(".txt"):
            sample_id = filename.split(".")[0]

            with open(data_path + "/" + filename, "r", encoding="utf-8") as file:
                line = file.readline()
                bounding_box = [float(val) for val in line.split(" ")][1:]
                data[sample_id].append(bounding_box)

    for image, bounding_box in data.values():
        X.append(image)
        y.append(bounding_box)

    X = np.array(X)
    y = np.array(y)

    return X, y


def load_data_pickle(data_path: str) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Load data stored in pickle format.
    @param data_path: Path to data.
    @return: X (samples) and y (labels).
    """
    with open(data_path, 'rb') as handle:
        X_test, y_test = pickle.load(handle)
    return X_test, y_test


def save_data_pickle(data_path, X: npt.NDArray, y: npt.NDArray) -> None:
    """
    Save data in pickle format.
    @param data_path: Path to data.
    @param X: Samples
    @param y: Labels
    """
    with open(data_path, "wb") as handle:
        pickle.dump((X, y), handle)
