"""Export inference-only checkpoints from the HK7-CV-Class training runs."""

from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR.parent / "HK7-CV-Class" / "brats-transunet" / "runs"
OUTPUT_DIR = PROJECT_DIR / "model"

CHECKPOINTS = {
    "hk7_unet2d_best.pt": SOURCE_DIR / "unet2d_200" / "best.pt",
    "hk7_transunet2d_best.pt": SOURCE_DIR / "tu2d_200" / "best.pt",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for output_name, source_path in CHECKPOINTS.items():
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        try:
            source = torch.load(source_path, map_location="cpu", weights_only=False)
        except TypeError:
            source = torch.load(source_path, map_location="cpu")
        exported = {
            "model": source["model"],
            "cfg": source["cfg"],
            "epoch": source.get("epoch"),
            "best": source.get("best"),
            "source": str(source_path),
        }
        output_path = OUTPUT_DIR / output_name
        torch.save(exported, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Exported {output_path.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
