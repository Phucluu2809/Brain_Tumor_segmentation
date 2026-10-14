"""Smoke-test exported checkpoints and optionally run one complete sample case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from brats_demo import build_model, dice_scores, load_modalities, predict_case  # noqa: E402
from brats_demo.models import region_probabilities  # noqa: E402
from brats_demo.pipeline import seg_to_regions  # noqa: E402


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run a complete sample volume.")
    parser.add_argument(
        "--full-model",
        choices=("unet", "transunet"),
        default="unet",
        help="Checkpoint used by --full (default: unet).",
    )
    args = parser.parse_args()

    model_paths = sorted((PROJECT_DIR / "model").glob("hk7_*_best.pt"))
    if len(model_paths) != 2:
        raise RuntimeError("Expected two exported hk7 checkpoints in model/.")

    for path in model_paths:
        checkpoint = load_checkpoint(path)
        model = build_model(checkpoint["cfg"]).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        with torch.inference_mode():
            probabilities = region_probabilities(model(torch.zeros(1, 4, 32, 32)))
        print(
            f"OK {path.name}: {checkpoint['cfg']['model']['name']}, "
            f"{sum(parameter.numel() for parameter in model.parameters()):,} params, "
            f"output={tuple(probabilities.shape)}"
        )

    if not args.full:
        return

    sample_paths = {
        modality: PROJECT_DIR / "datasets" / f"BraTS2021_00495_{modality}.nii.gz"
        for modality in ("t1", "t1ce", "t2", "flair")
    }
    volume, _, _ = load_modalities(sample_paths)
    ground_truth = np.asanyarray(
        nib.load(str(PROJECT_DIR / "datasets" / "BraTS2021_00495_seg.nii.gz")).dataobj
    ).astype(np.uint8)
    checkpoint_name = f"hk7_{args.full_model}2d_best.pt"
    checkpoint = load_checkpoint(PROJECT_DIR / "model" / checkpoint_name)
    model = build_model(checkpoint["cfg"]).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    result = predict_case(
        model,
        volume,
        torch.device("cpu"),
        threshold=0.5,
        batch_size=4,
        ground_truth=ground_truth,
    )
    scores = dice_scores(result["regions"], result["ground_truth"])
    labels = set(np.unique(result["segmentation"]).tolist())
    if not labels.issubset({0, 1, 2, 4}):
        raise RuntimeError(f"Unexpected BraTS labels: {labels}")
    if result["segmentation"].shape != ground_truth.shape:
        raise RuntimeError("Prediction was not restored to the original shape.")
    expected_regions = seg_to_regions(ground_truth)
    print(
        f"FULL OK model={args.full_model}, shape={result['segmentation'].shape}, labels={sorted(labels)}, "
        f"dice={scores}, gt_voxels={int(expected_regions.sum())}"
    )


if __name__ == "__main__":
    main()
