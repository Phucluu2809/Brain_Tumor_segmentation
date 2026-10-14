"""BraTS NIfTI loading, preprocessing and volume inference."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Callable, Mapping

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from .models import region_probabilities

MODALITIES = ("t1", "t1ce", "t2", "flair")
REGION_NAMES = ("WT", "TC", "ET")


def parse_modality(filename: str) -> str | None:
    """Infer a BraTS modality from a NIfTI filename."""
    name = Path(filename).name.lower()
    name = re.sub(r"\.nii(?:\.gz)?$", "", name)
    if "flair" in name or re.search(r"(?:^|[_-])t2f(?:$|[_-])", name):
        return "flair"
    if "t1ce" in name or "t1c" in name:
        return "t1ce"
    if re.search(r"(?:^|[_-])t2(?:w)?(?:$|[_-])", name):
        return "t2"
    if re.search(r"(?:^|[_-])t1(?:n)?(?:$|[_-])", name):
        return "t1"
    return None


def load_modalities(paths: Mapping[str, str | Path]):
    missing = [modality for modality in MODALITIES if modality not in paths]
    if missing:
        raise ValueError("Thiếu modality: " + ", ".join(missing))

    arrays = []
    affine = None
    header = None
    expected_shape = None
    for modality in MODALITIES:
        image = nib.load(str(paths[modality]))
        data = np.asanyarray(image.dataobj).astype(np.float32)
        if data.ndim != 3:
            raise ValueError(f"{modality} phải là volume 3D, nhận được shape {data.shape}.")
        if expected_shape is None:
            expected_shape = data.shape
            affine = image.affine.copy()
            header = image.header.copy()
        elif data.shape != expected_shape:
            raise ValueError(
                f"Các modality phải cùng shape: cần {expected_shape}, "
                f"nhưng {modality} là {data.shape}."
            )
        arrays.append(data)
    return np.stack(arrays, axis=0), affine, header


def brain_bbox(image: np.ndarray) -> np.ndarray:
    mask = (image != 0).any(axis=0)
    indexes = np.nonzero(mask)
    if not indexes[0].size:
        raise ValueError("Volume không chứa voxel não khác 0.")
    lower = [int(index.min()) for index in indexes]
    upper = [int(index.max()) + 1 for index in indexes]
    return np.asarray([lower, upper], dtype=np.int32)


def normalize(image: np.ndarray) -> np.ndarray:
    output = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[0]):
        values = image[channel]
        foreground = values != 0
        if foreground.any():
            selected = values[foreground]
            output[channel][foreground] = (
                selected - selected.mean()
            ) / (selected.std() + 1e-8)
    return output


def crop_to_bbox(array: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    slices = tuple(slice(start, stop) for start, stop in zip(bbox[0], bbox[1]))
    return array[(Ellipsis,) + slices]


def uncrop(array: np.ndarray, bbox: np.ndarray, original_shape) -> np.ndarray:
    output = np.zeros(array.shape[:-3] + tuple(original_shape), dtype=array.dtype)
    slices = tuple(slice(start, stop) for start, stop in zip(bbox[0], bbox[1]))
    output[(Ellipsis,) + slices] = array
    return output


def seg_to_regions(segmentation: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            segmentation > 0,
            (segmentation == 1) | (segmentation == 4),
            segmentation == 4,
        ],
        axis=0,
    ).astype(np.uint8)


def regions_to_seg(regions: np.ndarray) -> np.ndarray:
    whole, core, enhancing = (regions[index].astype(bool) for index in range(3))
    core = core & whole
    enhancing = enhancing & core
    segmentation = np.zeros(whole.shape, dtype=np.uint8)
    segmentation[whole] = 2
    segmentation[core] = 1
    segmentation[enhancing] = 4
    return segmentation


def binarize(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probabilities = probabilities.copy()
    probabilities[1] = np.maximum(probabilities[1], probabilities[2])
    probabilities[0] = np.maximum(probabilities[0], probabilities[1])
    return (probabilities > threshold).astype(np.uint8)


@torch.inference_mode()
def predict_volume(
    model,
    image: np.ndarray,
    device: torch.device,
    batch_size: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(image).to(device)
    size_x, size_y, _ = image.shape[1:]
    padding_x, padding_y = (-size_x) % 16, (-size_y) % 16
    probabilities = torch.zeros(3, *image.shape[1:], device=device)
    brain_slices = (tensor != 0).any(0).any(0).any(0)
    indexes = torch.nonzero(brain_slices).squeeze(1).tolist()
    if not indexes:
        return probabilities.cpu().numpy()

    autocast = (
        torch.autocast("cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    total_batches = (len(indexes) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(indexes), batch_size), start=1):
        selected = indexes[start : start + batch_size]
        inputs = tensor[..., selected].permute(3, 0, 1, 2)
        inputs = F.pad(inputs, (0, padding_y, 0, padding_x))
        with autocast:
            batch_probabilities = region_probabilities(model(inputs))
        batch_probabilities = batch_probabilities[..., :size_x, :size_y]
        probabilities[..., selected] = batch_probabilities.permute(1, 2, 3, 0).float()
        if progress:
            progress(batch_index, total_batches)
    return probabilities.cpu().numpy()


def predict_case(
    model,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 4,
    ground_truth: np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    original_shape = image.shape[1:]
    bbox = brain_bbox(image)
    cropped_image = normalize(crop_to_bbox(image, bbox))
    probabilities = predict_volume(
        model, cropped_image, device, batch_size=batch_size, progress=progress
    )
    regions = binarize(probabilities, threshold)
    output = {
        "image": cropped_image,
        "regions": regions,
        "bbox": bbox,
        "original_shape": original_shape,
        "segmentation": uncrop(regions_to_seg(regions), bbox, original_shape),
    }
    if ground_truth is not None:
        if tuple(ground_truth.shape) != tuple(original_shape):
            raise ValueError(
                f"Ground truth có shape {ground_truth.shape}, cần {original_shape}."
            )
        output["ground_truth"] = seg_to_regions(crop_to_bbox(ground_truth, bbox))
    return output


def dice_scores(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scores = {}
    for index, name in enumerate(REGION_NAMES):
        predicted = prediction[index].astype(bool)
        expected = target[index].astype(bool)
        denominator = predicted.sum() + expected.sum()
        scores[name] = (
            float(2 * np.logical_and(predicted, expected).sum() / denominator)
            if denominator
            else 1.0
        )
    scores["mean"] = float(np.mean(list(scores.values())))
    return scores
