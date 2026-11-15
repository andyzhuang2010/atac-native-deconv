"""Evaluation metrics for spatial deconvolution."""

from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def rmse(
    true: Union[pd.DataFrame, np.ndarray],
    predicted: Union[pd.DataFrame, np.ndarray],
) -> float:
    """Return RMSE for two arrays of proportions with the same shape."""
    return float(np.sqrt(np.mean((np.asarray(true) - np.asarray(predicted)) ** 2)))


def jsd(
    true: Union[pd.DataFrame, np.ndarray],
    predicted: Union[pd.DataFrame, np.ndarray],
) -> float:
    """Return mean Jensen-Shannon distance (base 2), skipping nonfinite values.

    Rows are samples and columns are cell types."""
    d = jensenshannon(np.asarray(true), np.asarray(predicted), axis=1, base=2)
    return float(np.mean(d[np.isfinite(d)]))


def pearson_per_celltype(
    true: Union[pd.DataFrame, np.ndarray],
    predicted: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Return Pearson correlation for each cell type across spots.

    Rows are spots and columns are cell types."""
    true_arr = np.asarray(true)
    pred_arr = np.asarray(predicted)
    n_ct = true_arr.shape[1]
    corrs = np.array(
        [np.corrcoef(true_arr[:, i], pred_arr[:, i])[0, 1] for i in range(n_ct)]
    )
    return corrs


def spearman_per_celltype(
    true: Union[pd.DataFrame, np.ndarray],
    predicted: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Return Spearman correlation for each cell type across spots.

    Rows are spots and columns are cell types."""
    from scipy.stats import spearmanr

    true_arr = np.asarray(true)
    pred_arr = np.asarray(predicted)
    n_ct = true_arr.shape[1]
    corrs = np.array(
        [spearmanr(true_arr[:, i], pred_arr[:, i]).statistic for i in range(n_ct)]
    )
    return corrs
