# utils/metrics.py

import numpy as np

from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# ============================================================
# EER (Equal Error Rate)
# ============================================================

def calculate_eer(labels, scores):
    """
    Equal Error Rate for CM system.
    bonafide = 1, spoof = 0
    """

    fpr, tpr, _ = metrics.roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    eer = brentq(
        lambda x: 1.0 - x - interp1d(fpr, tpr)(x),
        0.0,
        1.0,
    )

    return eer


# ============================================================
# t-DCF (CM-only)
# ============================================================

def compute_tDCF(
    bonafide_score_cm,
    spoof_score_cm,
    Pfa_asv,
    Pmiss_asv,
    Pfa_spoof_asv,
    cost_model,
):

    # Concatenate scores
    cm_scores = np.concatenate(
        [bonafide_score_cm, spoof_score_cm]
    )

    labels = np.concatenate(
        [
            np.ones_like(bonafide_score_cm),
            np.zeros_like(spoof_score_cm),
        ]
    )


    # Sort descending
    sorted_idx = np.argsort(cm_scores)[::-1]

    sorted_labels = labels[sorted_idx]


    tar = np.sum(sorted_labels)
    non = len(sorted_labels) - tar


    # CM miss / false alarm
    cm_miss = np.cumsum(sorted_labels == 1) / tar

    cm_fa = np.cumsum(sorted_labels == 0) / non


    # Cost model
    Cmiss = cost_model["Cmiss"]
    Cfa = cost_model["Cfa"]
    Cfa_spoof = cost_model["Cfa_spoof"]

    Ptar = cost_model["Ptar"]
    Pnon = cost_model["Pnon"]
    Pspoof = cost_model["Pspoof"]


    # tDCF
    tDCF = (

        Cmiss * Ptar * Pmiss_asv * (1 - cm_miss)
        + Cfa * Pnon * Pfa_asv * cm_fa
        + Cfa_spoof * Pspoof * Pfa_spoof_asv * (1 - cm_miss)

    ) / (

        Cmiss * Ptar * Pmiss_asv
        + Cfa * Pnon * Pfa_asv

    )


    tDCF_norm = tDCF / np.min(tDCF)

    thresholds = cm_scores[sorted_idx]

    return tDCF_norm, thresholds
