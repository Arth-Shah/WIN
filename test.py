# test.py

import numpy as np
import torch

from tqdm import tqdm


# ---------------- Local Imports ---------------- #

from config import exp_cfg, SAVE_PATH
from utils.device import get_device
from utils.metrics import calculate_eer, compute_tDCF

from data.dataloader import create_dataloaders

from models.preprocess import PreEmphasis


# ============================================================
# Main
# ============================================================

def main():

    # ---------------- Device ---------------- #
    device = get_device()


    # ---------------- Data ---------------- #
    _, _, test_loader = create_dataloaders()


    # ---------------- Pre-Emphasis ---------------- #
    pre = PreEmphasis(
        exp_cfg.PRE_EMPHASIS
    ).to(device)


    # ---------------- Load Best Model ---------------- #
    best_model = torch.load(
        SAVE_PATH,
        map_location=device,
        weights_only=False,
    )

    best_model.eval()


    # ---------------- t-DCF Parameters ---------------- #
    Pfa_asv = 0.05
    Pmiss_asv = 0.01
    Pfa_spoof_asv = 0.05

    cost_model = {
        "Cmiss": 1,
        "Cfa": 10,
        "Cfa_spoof": 10,
        "Ptar": 0.9801,
        "Pnon": 0.0099,
        "Pspoof": 0.01,
    }


    # ---------------- Testing ---------------- #
    test_scores = []
    test_labels = []


    with torch.no_grad():

        pbar = tqdm(
            test_loader,
            desc="Testing",
            leave=True,
        )


        for wav, label in pbar:

            wav = wav.to(device)
            label = label.to(device)

            wav = pre(wav)


            pred = best_model(wav).squeeze(-1)


            test_scores.extend(
                pred.cpu().numpy()
            )

            test_labels.extend(
                label.cpu().numpy()
            )


    # ---------------- Metrics ---------------- #

    test_eer = calculate_eer(
        test_labels,
        test_scores,
    )


    bona_cm = np.array(test_scores)[
        np.array(test_labels) == 1
    ]

    spoof_cm = np.array(test_scores)[
        np.array(test_labels) == 0
    ]


    tDCF_curve, thr = compute_tDCF(
        bona_cm,
        spoof_cm,
        Pfa_asv,
        Pmiss_asv,
        Pfa_spoof_asv,
        cost_model,
    )


    min_tDCF = np.min(tDCF_curve)


    # ---------------- Report ---------------- #

    print(f"\n🎯 Final Test EER:  {test_eer * 100:.2f}%")

    print(f"📊 Final min-tDCF: {min_tDCF:.4f}")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
