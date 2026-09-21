# -*- coding: utf-8 -*-
"""
Metrics shared by the report builders and experiment scripts, matching the definitions in
the first report (build_report.py): the reservoir's "inferPC" and the breathing-mode period.

@author: Adrian Wong
"""
import numpy as np


def infer_pc(y_true, y_pred):
    """Squared correlation between each true channel and each predicted one (channels x time).

    The diagonal is the per-channel "PC" quoted in the report; it is invariant to the scale
    and offset of the prediction, so it measures shape, not amplitude.
    """
    yy = y_true - y_true.mean(axis=1, keepdims=True)
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    return (yp @ yy.T) ** 2 / ((yp @ yp.T) * (yy @ yy.T))


def period_steps(z1, distance=100, win=41):
    """Median spacing (in steps) between breathing-mode peaks of a latent channel, and the peak count.

    The slow trend is removed with a moving average first. A signal with no oscillation
    (e.g. a reservoir that collapsed to a fixed point) yields (nan, n_peaks < 2).
    """
    from scipy.signal import find_peaks
    z1 = np.asarray(z1, dtype=float)
    z1d = z1 - np.convolve(z1, np.ones(win) / win, mode="same")
    peaks, _ = find_peaks(z1d, distance=distance, prominence=0.4 * np.std(z1d))
    if len(peaks) < 2:
        return float("nan"), len(peaks)
    return float(np.median(np.diff(peaks))), len(peaks)
