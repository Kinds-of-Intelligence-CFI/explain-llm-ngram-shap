import errno
import os
from typing import Tuple, Dict, List

import numpy as np
import numpy.typing as npt

from src.utils.data_explore import get_num_words_per_sample


def brier_decomposition(predictions: npt.NDArray,
                        outputs: npt.NDArray) -> Tuple[float, float, float]:
    brier = 1 / len(predictions) * sum((predictions - outputs) ** 2)

    # bin predictions
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_pred_indices = np.digitize(predictions, bin_centers)
    binned_predictions = bins[bin_pred_indices]

    bin_true_frequencies = np.zeros(10)
    bin_pred_frequencies = np.zeros(10)
    bin_counts = np.zeros(10)

    for i in range(10):
        idx = (predictions >= bins[i]) & (predictions < bins[i + 1])

        bin_true_frequencies[i] = np.sum(outputs[idx]) / np.sum(idx) if np.sum(idx) > 0 else 0
        # print(np.sum(outs[idx]), np.sum(idx), bin_true_frequencies[i])
        bin_pred_frequencies[i] = np.mean(predictions[idx]) if np.sum(idx) > 0 else 0
        bin_counts[i] = np.sum(idx)

    calibration = np.sum(bin_counts * (bin_true_frequencies - bin_pred_frequencies) ** 2) / np.sum(bin_counts) if np.sum(
        bin_counts) > 0 else 0
    refinement = np.sum(bin_counts * (bin_true_frequencies * (1 - bin_true_frequencies))) / np.sum(bin_counts) if np.sum(
        bin_counts) > 0 else 0

    # Compute refinement component
    # refinement = brier - calibration
    return brier, calibration, refinement


def calculate_sw_ratio(data: Dict[str, Dict],
                       llms: List[str]) -> Dict[str, float]:
    ratios = {}
    for llm in llms:
        # Count the number of samples (S)
        S = len(data[llm])

        # Num words per sample (W)
        W = get_num_words_per_sample(data[llm]['prompt'])

        # Calculate the S/W ratio
        ratio = S / W
        ratios[llm] = ratio
    return ratios


def try_mkdir(path: str) -> None:
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
