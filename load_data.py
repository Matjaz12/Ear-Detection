import os
import pickle

import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_data(data_path: str, mode: str = "GRAY") -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Function loads data and returns tensors X, y.

    X is of shape (num samples, )
    y is of shape (num samples, 6)

    @param data_path: Path to data.
    @param mode: L ("gray scale"), RGB ("Red Green Blue")
    @return: X (samples) and y (labels).
    """
    X, y = [], []
    data = {}

    # Store images
    for filename in sorted(os.listdir(data_path)):
        if filename.endswith(".png"):
            image = cv2.imread(data_path + "/" + filename)
            
            if mode == "GRAY":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif mode == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            sample_id = filename.split(".")[0]
            data[sample_id] = [image]

    # Store labels
    sample_idx = 0
    for filename in sorted(os.listdir(data_path)):
        if filename.endswith(".txt"):
            sample_id = filename.split(".")[0]

            with open(data_path + "/" + filename, "r", encoding="utf-8") as file:
                # Fetch sample data
                line = file.readline()
                sample_data = [float(val) for val in line.split(" ")]

                # Parse sample data
                true_class = sample_data[0]
                prob = 1
                bounding_box = sample_data[1:]

                # Construct sample label
                """
                sample_label = [sample_idx, true_class, prob, x_center, y_center, w, h]
                """
                sample_label = [sample_idx, true_class, prob]
                sample_label.extend(bounding_box)

                # Store sample label
                data[sample_id].append(np.array(sample_label))

            sample_idx += 1

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

    X_test = np.array(X_test)
    y_test = np.array(y_test)

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


if __name__ == "__main__":
    MODE = "GRAY"

    X_test, y_test = load_data(data_path="./ear_data/test", mode=MODE)
    save_data_pickle(f"./ear_data/X_test_and_y_test_{MODE}.pickle", X_test, y_test)
    X_test, y_test = load_data_pickle(f"./ear_data/X_test_and_y_test_{MODE}.pickle")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")
    print(f"sample_idx: {y_test[0][0]}, true_class: {y_test[0][1]}, prob: {y_test[0][2]}, bounding_box: {y_test[0][3:]}")