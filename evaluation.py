import numpy as np
import numpy.typing as npt


def intersection_over_union(ground_truth: npt.NDArray, predicted_bounding_box: npt.NDArray,
                            image_width: int, image_height: int) -> float:
    # https://www.google.com/search?channel=fs&client=ubuntu&q=python+interesection+over+union#fpstate=ive&vld=cid:fe6ebfe5,vid:XXYG5ZWtjj0

    x, y, w, h = ground_truth
    x *= image_width
    y *= image_height
    w *= image_width
    h *= image_height

    x_hat, y_hat, w_hat, h_hat = predicted_bounding_box
    x_hat *= image_width
    y_hat *= image_height
    w_hat *= image_width
    h_hat *= image_height

    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2

    x1_hat, y1_hat = x_hat - w_hat // 2, y_hat - h_hat // 2
    x2_hat, y2_hat = x_hat + w_hat // 2, y_hat + h_hat // 2

    x1_inter = max(x1, x1_hat)
    y1_inter = max(y1, y1_hat)
    x2_inter = min(x2, x2_hat)
    y2_inter = min(y2, y2_hat)

    intersection = abs(x1_inter - x2_inter) * abs(y1_inter - y2_inter)
    union = w * h + w_hat * h_hat - intersection

    return intersection / (union + 1e-6)
