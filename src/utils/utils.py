import errno
import os

import numpy as np

from src.utils.data_explore import get_num_words_per_sample


def brier_decomposition(preds, outs):
    brier = 1 / len(preds) * sum((preds - outs) ** 2)

    # bin predictions
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_pred_inds = np.digitize(preds, bin_centers)
    binned_preds = bins[bin_pred_inds]

    bin_true_freqs = np.zeros(10)
    bin_pred_freqs = np.zeros(10)
    bin_counts = np.zeros(10)

    for i in range(10):
        idx = (preds >= bins[i]) & (preds < bins[i + 1])

        bin_true_freqs[i] = np.sum(outs[idx]) / np.sum(idx) if np.sum(idx) > 0 else 0
        # print(np.sum(outs[idx]), np.sum(idx), bin_true_freqs[i])
        bin_pred_freqs[i] = np.mean(preds[idx]) if np.sum(idx) > 0 else 0
        bin_counts[i] = np.sum(idx)

    calibration = np.sum(bin_counts * (bin_true_freqs - bin_pred_freqs) ** 2) / np.sum(bin_counts) if np.sum(
        bin_counts) > 0 else 0
    refinement = np.sum(bin_counts * (bin_true_freqs * (1 - bin_true_freqs))) / np.sum(bin_counts) if np.sum(
        bin_counts) > 0 else 0

    # Compute refinement component
    # refinement = brier - calibration
    return brier, calibration, refinement


def calculate_sw_ratio(data, llms):
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


def try_mkdir(path):
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
