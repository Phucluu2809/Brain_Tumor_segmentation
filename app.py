from __future__ import annotations

import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import streamlit as st
import torch

from brats_demo import (
    MODALITIES,
    build_model,
    dice_scores,
    load_modalities,
    parse_modality,
    predict_case,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "datasets"
MODEL_DIR = PROJECT_DIR / "model"

MODEL_OPTIONS = {
    "U-Net 2D · Dice 0.905": {
        "path": MODEL_DIR / "hk7_unet2d_best.pt",
        "description": "CNN baseline — chính xác nhất trong hai checkpoint đã huấn luyện.",
    },
    "TransUNet 2D · Dice 0.868": {
        "path": MODEL_DIR / "hk7_transunet2d_best.pt",
        "description": "U-Net kết hợp Masked Transformer Decoder.",
    },
}

SAMPLE_FILES = {
    modality: DATA_DIR / f"BraTS2021_00495_{modality}.nii.gz"
    for modality in MODALITIES
}
SAMPLE_SEGMENTATION = DATA_DIR / "BraTS2021_00495_seg.nii.gz"


st.set_page_config(
    page_title="NeuroScope · Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #102238; --muted: #637083; --accent: #14b8a6; }
      .stApp { background: linear-gradient(160deg, #f7fbff 0%, #f7fafc 50%, #f2fbf9 100%); }
      [data-testid="stSidebar"] { background: #102238; }
      [data-testid="stSidebar"] * { color: #eef7f7; }
      [data-testid="stSidebar"] .stAlert * { color: #102238; }
      .hero {
        padding: 1.8rem 2rem; border-radius: 24px; margin-bottom: 1rem;
        background: radial-gradient(circle at 90% 10%, rgba(45,212,191,.35), transparent 28%),
                    linear-gradient(125deg, #102238, #163a4c);
        box-shadow: 0 18px 45px rgba(16,34,56,.16); color: white;
      }
      .hero-kicker { color: #5eead4; letter-spacing: .14em; font-weight: 700; font-size: .76rem; }
      .hero h1 { color: white; margin: .35rem 0 .45rem; font-size: clamp(2rem, 4vw, 3.5rem); }
      .hero p { color: #cbdbe5; max-width: 760px; margin: 0; font-size: 1.02rem; }
      .info-card {
        background: rgba(255,255,255,.78); border: 1px solid rgba(148,163,184,.24);
        border-radius: 18px; padding: 1rem 1.15rem; min-height: 112px;
        box-shadow: 0 8px 24px rgba(15,23,42,.06);
      }
      .info-label { color: #64748b; font-size: .75rem; letter-spacing: .08em; font-weight: 700; }
      .info-value { color: #102238; font-size: 1.1rem; font-weight: 750; margin-top: .35rem; }
      .legend { display:flex; flex-wrap:wrap; gap:.65rem 1.2rem; color:#475569; font-size:.88rem; }
      .dot { width:.72rem; height:.72rem; border-radius:50%; display:inline-block; margin-right:.38rem; }
      div.stButton > button { border-radius: 12px; font-weight: 700; min-height: 2.8rem; }
      [data-testid="stMetric"] {
        background: rgba(255,255,255,.72); border: 1px solid rgba(148,163,184,.22);
        padding: .8rem 1rem; border-radius: 16px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device_name: str):
    device = torch.device(device_name)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint["cfg"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint, device


def uploaded_paths(uploaded_files, directory: Path):
    paths = {}
    unknown = []
    duplicates = []
    for uploaded in uploaded_files:
        modality = parse_modality(uploaded.name)
        if modality is None:
            unknown.append(uploaded.name)
            continue
        if modality in paths:
            duplicates.append(modality)
            continue
        suffix = ".nii.gz" if uploaded.name.lower().endswith(".nii.gz") else ".nii"
        path = directory / f"{modality}{suffix}"
        path.write_bytes(uploaded.getbuffer())
        paths[modality] = path
    if unknown:
        raise ValueError("Không nhận diện được modality của: " + ", ".join(unknown))
    if duplicates:
        raise ValueError("Modality bị trùng: " + ", ".join(sorted(set(duplicates))))
    return paths


def load_ground_truth(path: Path):
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.uint8)


def display_slice(channel: np.ndarray) -> np.ndarray:
    return np.clip((channel + 3.0) / 6.0, 0.0, 1.0)


def color_mask(regions: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*regions.shape[1:], 3), dtype=np.float32)
    rgb[regions[0] > 0] = (0.20, 0.65, 0.95)
    rgb[regions[1] > 0] = (0.96, 0.40, 0.34)
    rgb[regions[2] > 0] = (0.99, 0.78, 0.25)
    return rgb


def overlay_image(grayscale: np.ndarray, regions: np.ndarray) -> np.ndarray:
    base = np.repeat(grayscale[..., None], 3, axis=-1)
    colors = color_mask(regions)
    foreground = regions[0].astype(bool)
    output = base.copy()
    output[foreground] = 0.48 * base[foreground] + 0.52 * colors[foreground]
    return output


def nifti_bytes(segmentation: np.ndarray, affine: np.ndarray, header) -> bytes:
    output_header = header.copy()
    output_header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(segmentation.astype(np.uint8), affine, output_header)
    return image.to_bytes()


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">COMPUTER VISION · BRATS 2021</div>
      <h1>NeuroScope</h1>
      <p>Phân đoạn tự động ba vùng khối u não từ bốn chuỗi xung MRI bằng
      các checkpoint U-Net và TransUNet được huấn luyện trong HK7-CV-Class.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## 🧠 NeuroScope")
    st.caption("Bảng điều khiển suy luận")
    st.divider()

    st.markdown("#### 1 · Mô hình")
    model_label = st.selectbox("Checkpoint", list(MODEL_OPTIONS), label_visibility="collapsed")
    selected_model = MODEL_OPTIONS[model_label]
    st.caption(selected_model["description"])

    checkpoint_path = selected_model["path"]
    if checkpoint_path.exists():
        st.success(f"Sẵn sàng · {checkpoint_path.stat().st_size / 1024**2:.1f} MB")
    else:
        st.error(f"Thiếu {checkpoint_path.name}")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Thiết bị: **{device_name.upper()}**")
    threshold = st.slider("Ngưỡng xác suất", 0.10, 0.90, 0.50, 0.05)
    batch_size = st.select_slider("Số lát cắt / batch", [1, 2, 4, 8, 16], value=4)

    st.divider()
    st.markdown("#### 2 · Dữ liệu")
    data_source = st.radio(
        "Nguồn dữ liệu",
        ["Ca mẫu BraTS", "Tải MRI lên"],
        horizontal=True,
        label_visibility="collapsed",
    )
    uploaded_files = None
    if data_source == "Ca mẫu BraTS":
        sample_ready = all(path.exists() for path in SAMPLE_FILES.values())
        if sample_ready:
            st.info("BraTS2021_00495 · có ground truth")
        else:
            st.error("Thiếu dữ liệu mẫu trong datasets/")
    else:
        uploaded_files = st.file_uploader(
            "Chọn đủ T1, T1ce, T2 và FLAIR",
            type=["nii", "gz"],
            accept_multiple_files=True,
            help="Tên file cần chứa t1, t1ce, t2 hoặc flair để nhận diện tự động.",
        )
        if uploaded_files:
            detected = [parse_modality(item.name) or "?" for item in uploaded_files]
            st.caption("Đã nhận: " + " · ".join(detected))

    run_clicked = st.button(
        "Chạy phân đoạn",
        type="primary",
        width="stretch",
        disabled=not checkpoint_path.exists(),
    )


if run_clicked:
    try:
        if data_source == "Ca mẫu BraTS":
            missing = [str(path.name) for path in SAMPLE_FILES.values() if not path.exists()]
            if missing:
                raise FileNotFoundError("Thiếu dữ liệu mẫu: " + ", ".join(missing))
            volume, affine, header = load_modalities(SAMPLE_FILES)
            ground_truth = (
                load_ground_truth(SAMPLE_SEGMENTATION)
                if SAMPLE_SEGMENTATION.exists()
                else None
            )
            case_name = "BraTS2021_00495"
        else:
            if not uploaded_files:
                raise ValueError("Hãy chọn đủ bốn file MRI trước khi chạy.")
            with tempfile.TemporaryDirectory() as temporary_directory:
                paths = uploaded_paths(uploaded_files, Path(temporary_directory))
                volume, affine, header = load_modalities(paths)
            ground_truth = None
            case_name = "uploaded_case"

        with st.spinner("Đang nạp checkpoint..."):
            model, checkpoint, device = load_model(str(checkpoint_path), device_name)

        progress_bar = st.progress(0, text="Chuẩn hóa và crop vùng não...")

        def update_progress(current: int, total: int):
            progress_bar.progress(
                current / total,
                text=f"Đang suy luận batch {current}/{total}...",
            )

        started = time.perf_counter()
        result = predict_case(
            model,
            volume,
            device,
            threshold=threshold,
            batch_size=batch_size,
            ground_truth=ground_truth,
            progress=update_progress,
        )
        elapsed = time.perf_counter() - started
        progress_bar.empty()
        result.update(
            {
                "affine": affine,
                "header": header,
                "spacing": tuple(float(value) for value in header.get_zooms()[:3]),
                "case_name": case_name,
                "model_label": model_label,
                "checkpoint_best": float(checkpoint.get("best", float("nan"))),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "config": checkpoint["cfg"],
                "threshold": threshold,
                "elapsed": elapsed,
            }
        )
        st.session_state["segmentation_result"] = result
        st.success(f"Hoàn tất phân đoạn trong {elapsed:.1f} giây.")
    except Exception as error:
        st.error(f"Không thể chạy phân đoạn: {error}")
        with st.expander("Chi tiết kỹ thuật"):
            st.exception(error)


result = st.session_state.get("segmentation_result")
if result is None:
    cards = st.columns(3)
    cards[0].markdown(
        '<div class="info-card"><div class="info-label">ĐẦU VÀO</div>'
        '<div class="info-value">T1 · T1ce · T2 · FLAIR</div></div>',
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        '<div class="info-card"><div class="info-label">ĐẦU RA</div>'
        '<div class="info-value">WT · TC · ET</div></div>',
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        '<div class="info-card"><div class="info-label">MÔ HÌNH MẶC ĐỊNH</div>'
        '<div class="info-value">U-Net 2D · Dice 0.905</div></div>',
        unsafe_allow_html=True,
    )
    st.info("Chọn nguồn dữ liệu ở thanh bên và nhấn **Chạy phân đoạn** để bắt đầu.")
else:
    st.markdown("### Kết quả phân đoạn")
    st.caption(
        f"{result['case_name']} · {result['model_label']} · "
        f"ngưỡng {result['threshold']:.2f} · {result['elapsed']:.1f}s"
    )

    region_counts = result["regions"].sum(axis=(1, 2, 3))
    voxel_ml = float(np.prod(result["spacing"])) / 1000.0
    metrics = st.columns(4)
    metrics[0].metric("Whole Tumor", f"{region_counts[0] * voxel_ml:,.1f} mL")
    metrics[1].metric("Tumor Core", f"{region_counts[1] * voxel_ml:,.1f} mL")
    metrics[2].metric("Enhancing Tumor", f"{region_counts[2] * voxel_ml:,.1f} mL")
    metrics[3].metric("Thời gian", f"{result['elapsed']:.1f} s")

    if "ground_truth" in result:
        scores = dice_scores(result["regions"], result["ground_truth"])
        st.markdown("#### Đối chiếu ground truth")
        dice_columns = st.columns(4)
        dice_columns[0].metric("Dice WT", f"{scores['WT']:.3f}")
        dice_columns[1].metric("Dice TC", f"{scores['TC']:.3f}")
        dice_columns[2].metric("Dice ET", f"{scores['ET']:.3f}")
        dice_columns[3].metric("Dice trung bình", f"{scores['mean']:.3f}")

    tumor_per_slice = result["regions"][0].sum(axis=(0, 1))
    best_slice = int(np.argmax(tumor_per_slice)) if tumor_per_slice.any() else result["image"].shape[-1] // 2

    controls = st.columns([2, 1])
    with controls[0]:
        slice_index = st.slider(
            "Lát cắt axial",
            0,
            result["image"].shape[-1] - 1,
            best_slice,
            key=f"slice-{result['case_name']}-{result['model_label']}",
        )
    with controls[1]:
        modality_label = st.selectbox("Chuỗi xung hiển thị", ["FLAIR", "T1ce", "T2", "T1"])

    modality_index = {"T1": 0, "T1ce": 1, "T2": 2, "FLAIR": 3}[modality_label]
    grayscale = display_slice(result["image"][modality_index, :, :, slice_index])
    predicted_slice = result["regions"][:, :, :, slice_index]
    original_z = int(result["bbox"][0, 2] + slice_index)
    st.caption(f"Lát crop {slice_index} · tọa độ Z gốc {original_z}")

    panels = st.columns(4 if "ground_truth" in result else 3)
    panels[0].markdown(f"**MRI · {modality_label}**")
    panels[0].image(np.rot90(grayscale), width="stretch", clamp=True)

    panel_offset = 1
    if "ground_truth" in result:
        truth_slice = result["ground_truth"][:, :, :, slice_index]
        panels[1].markdown("**Ground truth**")
        panels[1].image(np.rot90(color_mask(truth_slice)), width="stretch", clamp=True)
        panel_offset = 2

    panels[panel_offset].markdown("**Mask AI**")
    panels[panel_offset].image(
        np.rot90(color_mask(predicted_slice)), width="stretch", clamp=True
    )
    panels[panel_offset + 1].markdown("**Overlay**")
    panels[panel_offset + 1].image(
        np.rot90(overlay_image(grayscale, predicted_slice)),
        width="stretch",
        clamp=True,
    )

    st.markdown(
        """
        <div class="legend">
          <span><i class="dot" style="background:#33a6f2"></i>WT · Whole Tumor</span>
          <span><i class="dot" style="background:#f56657"></i>TC · Tumor Core</span>
          <span><i class="dot" style="background:#fcca40"></i>ET · Enhancing Tumor</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    download_columns = st.columns([1, 2])
    with download_columns[0]:
        st.download_button(
            "Tải mask NIfTI",
            data=nifti_bytes(
                result["segmentation"], result["affine"], result["header"]
            ),
            file_name=f"{result['case_name']}_prediction.nii",
            mime="application/octet-stream",
            width="stretch",
        )
    with download_columns[1]:
        with st.expander("Thông tin checkpoint và volume"):
            st.json(
                {
                    "model": result["config"]["model"],
                    "checkpoint_epoch": result["checkpoint_epoch"],
                    "checkpoint_best_dice": result["checkpoint_best"],
                    "original_shape": result["original_shape"],
                    "cropped_shape": result["image"].shape[1:],
                    "voxel_spacing_mm": result["spacing"],
                    "threshold": result["threshold"],
                }
            )

st.divider()
st.caption(
    "Chỉ dùng cho mục đích học tập và trình diễn nghiên cứu; không thay thế chẩn đoán y khoa."
)
