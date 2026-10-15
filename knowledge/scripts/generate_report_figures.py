"""Generate original, report-ready architecture diagrams for knowledge/README.md.

Run from the Brain-Tumor-segmentation repository root:
    .\myenv\Scripts\python.exe knowledge\scripts\generate_report_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
WORKSPACE = ROOT.parents[1]
ASSETS.mkdir(parents=True, exist_ok=True)

NAVY = "#17324D"
BLUE = "#3A86D6"
LIGHT_BLUE = "#DCEEFF"
PURPLE = "#8B5FC0"
LIGHT_PURPLE = "#EEE5F8"
TEAL = "#20A39E"
LIGHT_TEAL = "#DDF5F2"
ORANGE = "#F3A712"
GREEN = "#5DBB63"
RED = "#ED6A5A"
GRAY = "#607080"
LIGHT_GRAY = "#F4F7FA"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 18,
        "figure.facecolor": "white",
    }
)


def canvas(width: float = 16, height: float = 7):
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, xy, width, height, text, color=LIGHT_BLUE, edge=NAVY, fontsize=11,
        radius=0.02, text_color=NAVY, weight="bold", lw=1.8):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, linespacing=1.35)
    return patch


def arrow(ax, start, end, color=NAVY, lw=2.0, style="-|>", connection="arc3"):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=15,
                            linewidth=lw, color=color, connectionstyle=connection)
    ax.add_patch(patch)
    return patch


def title(ax, text, subtitle=None):
    ax.text(0.5, 0.95, text, ha="center", va="center", fontsize=20,
            color=NAVY, weight="bold")
    if subtitle:
        ax.text(0.5, 0.905, subtitle, ha="center", va="center", fontsize=10.5,
                color=GRAY)


def save(fig, name):
    fig.savefig(ASSETS / name, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)


def pipeline_figure():
    fig, ax = canvas(16, 7)
    title(ax, "PIPELINE PHÂN ĐOẠN KHỐI U NÃO", "Luồng dữ liệu end-to-end gồm sáu giai đoạn")

    items = [
        ("1", "ĐỌC & TIỀN XỬ LÝ", "NIfTI / TAR / NPZ\nCrop não · Z-score", BLUE),
        ("2", "LẤY MẪU", "Patch 160² (2D) / 96³ (3D)\nfg_prob = 0,5", PURPLE),
        ("3", "PHÂN ĐOẠN", "U-Net 2D\nTransUNet 2D", TEAL),
        ("4", "SUY LUẬN TOÀN VOLUME", "Batch lát cắt 2D\nSliding window 3D", BLUE),
        ("5", "HỢP NHẤT", "Ghép lát / Gaussian blend\nUncrop về kích thước gốc", PURPLE),
        ("6", "HẬU XỬ LÝ & DEMO", "ET ⊆ TC ⊆ WT\nThreshold · NIfTI · Overlay", TEAL),
    ]
    xs = [0.05, 0.36, 0.67]
    ys = [0.60, 0.25]
    positions = [(xs[i % 3], ys[i // 3]) for i in range(6)]
    for (num, heading, detail, color), (x, y) in zip(items, positions):
        ax.add_patch(Circle((x + 0.035, y + 0.12), 0.028, facecolor=color, edgecolor="white", lw=1.5))
        ax.text(x + 0.035, y + 0.12, num, ha="center", va="center", color="white", weight="bold")
        box(ax, (x + 0.065, y), 0.235, 0.24, f"{heading}\n\n{detail}",
            color=LIGHT_GRAY, edge=color, fontsize=10.5)

    arrow(ax, (0.35, 0.72), (0.415, 0.72), ORANGE, 2.5)
    arrow(ax, (0.66, 0.72), (0.725, 0.72), ORANGE, 2.5)
    arrow(ax, (0.86, 0.58), (0.86, 0.50), ORANGE, 2.5)
    arrow(ax, (0.725, 0.37), (0.66, 0.37), ORANGE, 2.5)
    arrow(ax, (0.415, 0.37), (0.35, 0.37), ORANGE, 2.5)
    ax.text(0.5, 0.08, "Đầu ra: ba mặt nạ xác suất WT, TC, ET và file phân đoạn BraTS",
            ha="center", va="center", fontsize=12, color=NAVY, weight="bold")
    save(fig, "pipeline_6_stage.png")


def brats_regions_figure():
    fig, ax = canvas(15, 7)
    title(ax, "BIỂU DIỄN NHÃN BRATS", "Từ nhãn mô tổn thương sang ba vùng đánh giá lồng nhau")

    box(ax, (0.05, 0.60), 0.23, 0.20, "Nhãn gốc\n{0, 1, 2, 4}", LIGHT_GRAY, GRAY, 14)
    box(ax, (0.05, 0.32), 0.23, 0.16, "seg > 0\n→ WT", "#E8F7E9", GREEN, 12)
    box(ax, (0.35, 0.32), 0.23, 0.16, "seg = 1 hoặc 4\n→ TC", "#FFF5D6", ORANGE, 12)
    box(ax, (0.65, 0.32), 0.23, 0.16, "seg = 4\n→ ET", "#FDE6E3", RED, 12)
    for x in (0.165, 0.465, 0.765):
        arrow(ax, (0.165, 0.59), (x, 0.50), GRAY, 1.7,
              connection="arc3,rad=0.12" if x != 0.165 else "arc3")

    center = (0.72, 0.73)
    ax.add_patch(Circle(center, 0.15, facecolor="#BFE8C3", edgecolor=GREEN, lw=2.5))
    ax.add_patch(Circle(center, 0.10, facecolor="#FFE08A", edgecolor=ORANGE, lw=2.5))
    ax.add_patch(Circle(center, 0.052, facecolor="#F5988C", edgecolor=RED, lw=2.5))
    ax.text(0.72, 0.85, "WT", ha="center", va="center", weight="bold", color="#1E6B2A")
    ax.text(0.72, 0.79, "TC", ha="center", va="center", weight="bold", color="#8B6200")
    ax.text(0.72, 0.73, "ET", ha="center", va="center", weight="bold", color="#8C2318")
    ax.text(0.72, 0.56, "ET ⊆ TC ⊆ WT", ha="center", va="center", fontsize=14,
            color=NAVY, weight="bold")
    ax.text(0.5, 0.13,
            "Ba kênh dùng sigmoid độc lập; cumulative max được áp dụng trước threshold để bảo đảm tính lồng nhau.",
            ha="center", va="center", fontsize=11.5, color=GRAY)
    save(fig, "brats_label_regions.png")


def unet_figure():
    fig, ax = canvas(17, 8)
    ax.set_xlim(-0.02, 1.06)
    title(ax, "U-NET 2D TRONG DỰ ÁN", "Encoder–decoder đối xứng với skip connection đa tỷ lệ")

    enc = [
        (0.05, 0.69, "E0\n32 × 160²"),
        (0.18, 0.57, "E1\n64 × 80²"),
        (0.31, 0.45, "E2\n128 × 40²"),
        (0.44, 0.33, "E3\n256 × 20²"),
        (0.54, 0.19, "Bottleneck\n320 × 10²"),
    ]
    dec = [
        (0.65, 0.33, "D3\n256 × 20²"),
        (0.75, 0.45, "D2\n128 × 40²"),
        (0.84, 0.57, "D1\n64 × 80²"),
        (0.91, 0.69, "D0\n32 × 160²"),
    ]
    w, h = 0.085, 0.13
    for x, y, label in enc[:-1]:
        box(ax, (x, y), w, h, label, LIGHT_BLUE, BLUE, 10)
    box(ax, (enc[-1][0], enc[-1][1]), 0.105, 0.14, enc[-1][2], LIGHT_PURPLE, PURPLE, 10)
    for x, y, label in dec:
        box(ax, (x, y), w, h, label, LIGHT_TEAL, TEAL, 10)

    points = [(x + w, y + h / 2) for x, y, _ in enc[:-1]] + [(0.645, 0.26)]
    targets = [(enc[i + 1][0], enc[i + 1][1] + 0.07) for i in range(4)]
    for s, e in zip(points, targets):
        arrow(ax, s, e, ORANGE, 2.2)
    dec_starts = [(0.645, 0.26)] + [(x + w, y + h / 2) for x, y, _ in dec[:-1]]
    dec_targets = [(x, y + h / 2) for x, y, _ in dec]
    for s, e in zip(dec_starts, dec_targets):
        arrow(ax, s, e, TEAL, 2.2)

    skip_levels = [0.855, 0.765, 0.665, 0.565]
    for (xe, ye, _), (xd, yd, _), level in zip(enc[:4], reversed(dec), skip_levels):
        x1, x2 = xe + w / 2, xd + w / 2
        y1, y2 = ye + h, yd + h
        ax.plot([x1, x1, x2], [y1, level, level], color=BLUE, lw=1.7,
                linestyle=(0, (5, 3)))
        arrow(ax, (x2, level), (x2, y2), BLUE, 1.7)
    ax.text(0.5, 0.865, "Skip connections: giữ chi tiết không gian từ encoder",
            ha="center", color=BLUE, weight="bold")

    box(ax, (-0.01, 0.69), 0.045, h, "4\nkênh", "#FFFFFF", NAVY, 9)
    arrow(ax, (0.035, 0.755), (0.05, 0.755), NAVY, 2)
    box(ax, (1.005, 0.69), 0.045, h, "3\nmask", "#FFFFFF", NAVY, 9)
    arrow(ax, (0.995, 0.755), (1.005, 0.755), NAVY, 2)
    ax.text(0.50, 0.08, "Downsampling mở rộng ngữ cảnh · Upsampling khôi phục độ phân giải · Đầu ra sigmoid WT/TC/ET",
            ha="center", color=GRAY, fontsize=11)
    save(fig, "unet2d_architecture.png")


def transunet_figure():
    fig, ax = canvas(17, 8)
    ax.set_xlim(0, 1.06)
    title(ax, "TRANSUNET 2D DECODER-ONLY", "U-Net đa tỷ lệ kết hợp 20 query và ba vòng masked cross-attention")

    box(ax, (0.04, 0.61), 0.18, 0.18, "U-Net backbone\n4 MRI → features", LIGHT_BLUE, BLUE, 13)
    feature_y = [0.70, 0.49, 0.28]
    feature_text = ["Feature 1/8", "Feature 1/4", "Feature 1/2"]
    for y, text_ in zip(feature_y, feature_text):
        box(ax, (0.28, y), 0.13, 0.11, text_, LIGHT_BLUE, BLUE, 10)
        arrow(ax, (0.22, 0.70), (0.28, y + 0.055), BLUE, 1.7,
              connection="arc3,rad=0.10")

    box(ax, (0.28, 0.09), 0.13, 0.10, "20 query\nhọc được", LIGHT_PURPLE, PURPLE, 10)
    layers = [
        (0.49, 0.66, "Decoder layer 1\ncoarse"),
        (0.64, 0.45, "Decoder layer 2"),
        (0.79, 0.24, "Decoder layer 3\nfine"),
    ]
    for x, y, label in layers:
        box(ax, (x, y), 0.13, 0.13, label, LIGHT_PURPLE, PURPLE, 10)

    arrow(ax, (0.41, 0.14), (0.49, 0.70), PURPLE, 2, connection="arc3,rad=-0.18")
    arrow(ax, (0.41, 0.755), (0.49, 0.725), BLUE, 2)
    arrow(ax, (0.62, 0.725), (0.64, 0.515), PURPLE, 2)
    arrow(ax, (0.41, 0.545), (0.64, 0.515), BLUE, 2)
    arrow(ax, (0.77, 0.515), (0.79, 0.305), PURPLE, 2)
    arrow(ax, (0.41, 0.335), (0.79, 0.305), BLUE, 2)

    ax.text(0.555, 0.84, "masked cross-attention", ha="center", color=PURPLE, weight="bold")
    ax.text(0.705, 0.61, "mask trung gian", ha="center", color=PURPLE, fontsize=9)
    ax.text(0.85, 0.40, "mask trung gian", ha="center", color=PURPLE, fontsize=9)

    box(ax, (0.95, 0.49), 0.085, 0.16, "Class\nWT/TC/ET/∅", "#FFF5D6", ORANGE, 9)
    box(ax, (0.95, 0.25), 0.085, 0.16, "Query\nmasks", "#E8F7E9", GREEN, 10)
    arrow(ax, (0.92, 0.305), (0.95, 0.33), NAVY, 2)
    arrow(ax, (0.92, 0.305), (0.95, 0.56), NAVY, 2, connection="arc3,rad=-0.22")
    ax.text(0.70, 0.10, "Huấn luyện: Hungarian matching + class loss + BCE mask + Dice mask",
            ha="center", color=NAVY, fontsize=11.5, weight="bold")
    save(fig, "transunet_decoder_architecture.png")


def mamba_figure():
    fig, ax = canvas(16, 7)
    title(ax, "TỪ THỂ TÍCH 3D ĐẾN SELECTIVE SCAN", "Mamba mô hình hóa chuỗi voxel với chi phí tuyến tính theo độ dài chuỗi")

    box(ax, (0.04, 0.61), 0.15, 0.18, "Feature 3D\nC × H × W × D", LIGHT_BLUE, BLUE, 12)
    box(ax, (0.25, 0.61), 0.15, 0.18, "Linearize\nN = H·W·D", LIGHT_GRAY, GRAY, 12)
    arrow(ax, (0.19, 0.70), (0.25, 0.70), NAVY, 2.2)

    labels = [("Forward", 0.74), ("Reverse", 0.54), ("Inter-slice", 0.34)]
    for label, y in labels:
        box(ax, (0.48, y), 0.15, 0.12, label, LIGHT_PURPLE, PURPLE, 11)
        arrow(ax, (0.40, 0.70), (0.48, y + 0.06), PURPLE, 1.8,
              connection="arc3,rad=0.12")

    box(ax, (0.70, 0.48), 0.14, 0.18, "Selective scan\nhₜ = Āhₜ₋₁ + B̄xₜ", LIGHT_TEAL, TEAL, 11)
    for _, y in labels:
        arrow(ax, (0.63, y + 0.06), (0.70, 0.57), TEAL, 1.7,
              connection="arc3,rad=-0.12")
    box(ax, (0.89, 0.48), 0.09, 0.18, "Adaptive\nweights\nαf, αr, αs", "#FFF5D6", ORANGE, 10)
    arrow(ax, (0.84, 0.57), (0.89, 0.57), ORANGE, 2.2)

    ax.text(0.50, 0.20, "Self-attention: O(N²)", ha="center", va="center",
            color=RED, fontsize=14, weight="bold")
    ax.text(0.50, 0.12, "Selective scan: O(N)", ha="center", va="center",
            color=TEAL, fontsize=14, weight="bold")
    ax.text(0.84, 0.30, "Đầu ra toàn cục\nđã hợp nhất hướng", ha="center", va="center",
            color=NAVY, fontsize=11)
    save(fig, "ssm_mamba_flow.png")


def crop_dau_blocks():
    source = ASSETS / "dau_mamba_architecture.png"
    image = Image.open(source).convert("RGB")
    width, height = image.size
    crops = {
        "dsh_block.png": (0, int(height * 0.55), int(width * 0.59), int(height * 0.87)),
        "ugsf_block.png": (int(width * 0.59), int(height * 0.55), width, int(height * 0.785)),
    }
    for name, bounds in crops.items():
        crop = image.crop(bounds)
        canvas_img = Image.new("RGB", (crop.width + 40, crop.height + 40), "white")
        canvas_img.paste(crop, (20, 20))
        canvas_img.save(ASSETS / name, quality=95)


def collect_experiment_figures():
    """Copy stable plots and create a same-case qualitative comparison."""
    project = WORKSPACE / "HK7-CV-Class" / "brats-transunet"
    sources = {
        "training_curves.png": project / "figures" / "training_curves.png",
        "brats2021_00495_modalities.png": project / "figures" / "BraTS2021_00495.png",
    }
    for name, source in sources.items():
        if source.exists():
            Image.open(source).convert("RGB").save(ASSETS / name, quality=95)

    unet_path = project / "runs" / "unet2d_200" / "eval_test" / "BraTS2021_00088.png"
    trans_path = project / "runs" / "tu2d_200" / "eval_test" / "BraTS2021_00088.png"
    if not (unet_path.exists() and trans_path.exists()):
        return
    unet = Image.open(unet_path).convert("RGB")
    trans = Image.open(trans_path).convert("RGB")
    target_width = max(unet.width, trans.width)

    def resize_width(img):
        if img.width == target_width:
            return img
        new_height = round(img.height * target_width / img.width)
        return img.resize((target_width, new_height), Image.Resampling.LANCZOS)

    unet, trans = resize_width(unet), resize_width(trans)
    header_h = 62
    gap = 20
    out = Image.new("RGB", (target_width, header_h * 2 + unet.height + trans.height + gap), "white")
    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, 31)
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, target_width, header_h), fill="#DCEEFF")
    draw.text((25, 13), "U-Net 2D — ca test BraTS2021_00088", fill=NAVY, font=font)
    out.paste(unet, (0, header_h))
    y2 = header_h + unet.height + gap
    draw.rectangle((0, y2, target_width, y2 + header_h), fill="#EEE5F8")
    draw.text((25, y2 + 13), "TransUNet 2D — cùng ca test BraTS2021_00088", fill=NAVY, font=font)
    out.paste(trans, (0, y2 + header_h))
    out.save(ASSETS / "brats2021_00088_model_comparison.png", quality=95)


def main():
    pipeline_figure()
    brats_regions_figure()
    unet_figure()
    transunet_figure()
    mamba_figure()
    crop_dau_blocks()
    collect_experiment_figures()
    print(f"Generated report figures in: {ASSETS}")


if __name__ == "__main__":
    main()
