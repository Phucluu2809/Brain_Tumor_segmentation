import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch

# Kiem tra thu vien y khoa MONAI & Nibabel
try:
    import nibabel as nib
    from monai.networks.nets import SwinUNETR
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        NormalizeIntensityd,
        ToTensord,
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

# Cau hinh trang Streamlit
st.set_page_config(
    page_title="Brain Tumor Segmentation (Swin UNETR)",
    page_icon="🧠",
    layout="wide"
)

# Tieu de ung dung Web
st.title("🧠 Demo Phân Đoạn Khối U U Nào MRI 3D (Swin UNETR)")
st.markdown("Ứng dụng hỗ trợ chẩn đoán và phân đoạn u não tự động trên ảnh MRI đa chuỗi xung (BraTS Dataset).")
st.markdown("---")

# Thanh Ben Sidebar - Cau hinh & Tai file
st.sidebar.header("⚙️ Cấu hình Model & Dữ liệu")

model_path = st.sidebar.text_input("Đường dẫn file trọng số (Model Weights):", "model/bestmodel.pt")

@st.cache_resource
def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not MONAI_AVAILABLE:
        return None, device, "Chưa cài đặt thư viện MONAI hoặc Nibabel."
    
    try:
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=True,
        ).to(device)
        
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            return model, device, "LOADED"
        else:
            return None, device, f"Không tìm thấy file trọng số tại `{weights_path}`."
    except Exception as e:
        return None, device, f"Lỗi khi nạp model: {str(e)}"

model, device, status = load_model(model_path)

if status == "LOADED":
    st.sidebar.success(f"✅ Đã nạp thành công: `{model_path}` ({device.type.upper()})")
else:
    st.sidebar.warning(f"⚠️ Trạng thái Model: {status}")

st.sidebar.markdown("---")
st.sidebar.header("📁 Tải Dữ Liệu Đầu Vào")

use_demo = st.sidebar.checkbox("🎮 Sử dụng dữ liệu Mô Phỏng (Demo Mode)", value=True)
uploaded_file = st.sidebar.file_uploader("Tải lên file ảnh MRI 3D (.nii / .nii.gz)", type=["nii", "gz"])

# Ham tao du lieu gia lap de test giao dien ngay lập tức
def generate_simulated_data():
    H, W, D = 128, 128, 64
    y, x, z = np.ogrid[-H//2:H//2, -W//2:W//2, -D//2:D//2]
    brain_mask = (x**2 / (45**2) + y**2 / (55**2) + z**2 / (25**2)) <= 1
    
    mri_flair = np.zeros((H, W, D))
    mri_flair[brain_mask] = np.random.normal(0.5, 0.08, np.sum(brain_mask))
    
    tumor_center = (18, 12, 0)
    tx, ty, tz = tumor_center
    
    wt = ((x-tx)**2 / (18**2) + (y-ty)**2 / (20**2) + (z-tz)**2 / (10**2)) <= 1
    tc = ((x-tx)**2 / (12**2) + (y-ty)**2 / (14**2) + (z-tz)**2 / (7**2)) <= 1
    et = ((x-tx)**2 / (7**2) + (y-ty)**2 / (8**2) + (z-tz)**2 / (4**2)) <= 1
    
    wt = wt & brain_mask
    tc = tc & brain_mask
    et = et & brain_mask
    
    mri_flair[wt] += 0.35
    mri_flair[tc] += 0.25
    mri_flair[et] += 0.3
    mri_flair = np.clip(mri_flair, 0, 1)
    
    mask_3d = np.stack([wt.astype(float), tc.astype(float), et.astype(float)], axis=0)
    return mri_flair, mask_3d

if use_demo:
    st.info("💡 Bạn đang ở chế độ **Demo Mô Phỏng** (Có sẵn ảnh MRI 3D mẫu và mặt nạ u để test nhanh giao diện UI).")
    mri_data, mask_data = generate_simulated_data()
    num_slices = mri_data.shape[2]
    
    slice_idx = st.sidebar.slider("🔪 Chọn lát cắt MRI (Slice Z-axis):", 0, num_slices - 1, num_slices // 2)
    
    col1, col2, col3 = st.columns(3)
    
    flair_slice = mri_data[:, :, slice_idx]
    wt_slice = mask_data[0, :, :, slice_idx]
    tc_slice = mask_data[1, :, :, slice_idx]
    et_slice = mask_data[2, :, :, slice_idx]
    
    rgb_mask = np.zeros((*flair_slice.shape, 3))
    rgb_mask[wt_slice > 0] = [1.0, 0.2, 0.2]  # Do: WT
    rgb_mask[tc_slice > 0] = [0.2, 0.8, 0.2]  # Xanh la: TC
    rgb_mask[et_slice > 0] = [1.0, 0.9, 0.1]  # Vang: ET
    
    with col1:
        st.subheader("1. Ảnh MRI Gốc (FLAIR)")
        fig1, ax1 = plt.subplots()
        ax1.imshow(flair_slice, cmap="gray")
        ax1.axis("off")
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader("2. Mặt Nạ Khối U (AI Mask)")
        fig2, ax2 = plt.subplots()
        ax2.imshow(rgb_mask)
        ax2.axis("off")
        st.pyplot(fig2)
        plt.close(fig2)

    with col3:
        st.subheader("3. Kết Quả Tô Màu Khối U (Overlay)")
        fig3, ax3 = plt.subplots()
        ax3.imshow(flair_slice, cmap="gray")
        ax3.imshow(rgb_mask, alpha=0.45)
        ax3.axis("off")
        st.pyplot(fig3)
        plt.close(fig3)

    st.markdown("### 🎨 Chú thích màu sắc phân vùng u (BraTS Standard):")
    st.markdown("- 🔴 **Màu Đỏ**: **Whole Tumor (WT)** - Toàn bộ vùng khối u & phù nề.")
    st.markdown("- 🟢 **Màu Xanh Lá**: **Tumor Core (TC)** - Lõi khối u đặc.")
    st.markdown("- 🟡 **Màu Vàng**: **Enhancing Tumor (ET)** - Vùng u hoạt hóa sinh trưởng mạnh.")

elif uploaded_file is not None:
    if model is None:
        st.error(f"❌ Chưa nạp được Model từ đường dẫn `{model_path}`. Vui lòng kiểm tra lại file trọng số.")
    else:
        with tempfile.NamedTemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info("🔄 Đang nạp và tiền xử lý ảnh MRI 3D...")
            
            img_obj = nib.load(input_path)
            img_data = img_obj.get_fdata()

            if img_data.ndim == 3:
                img_4d = np.stack([img_data] * 4, axis=-1)
            else:
                img_4d = img_data

            num_slices = img_4d.shape[2]
            slice_idx = st.sidebar.slider("🔪 Chọn lát cắt MRI (Slice Z-axis):", 0, num_slices - 1, num_slices // 2)

            if st.sidebar.button("🚀 Chạy Phân Đoạn U Nào!", type="primary"):
                with st.spinner("🧠 Swin UNETR đang quét thể tích 3D..."):
                    transforms = Compose([
                        LoadImaged(keys=["image"]),
                        EnsureChannelFirstd(keys=["image"]),
                        Orientationd(keys=["image"], axcodes="RAS"),
                        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                        ToTensord(keys=["image"]),
                    ])
                    
                    try:
                        data_prep = transforms({"image": input_path})
                        inputs = data_prep["image"].unsqueeze(0).to(device)

                        with torch.no_grad():
                            val_outputs = sliding_window_inference(
                                inputs, (96, 96, 96), 4, model, overlap=0.5
                            )
                            val_outputs = (torch.sigmoid(val_outputs) > 0.5).float().cpu().numpy()
                        
                        st.success("🎉 Phân đoạn hoàn tất!")

                        col1, col2, col3 = st.columns(3)
                        flair_slice = img_4d[:, :, slice_idx, 0] if img_4d.ndim == 4 else img_4d[:, :, slice_idx]

                        wt_mask = val_outputs[0, 0, :, :, slice_idx]
                        tc_mask = val_outputs[0, 1, :, :, slice_idx]
                        et_mask = val_outputs[0, 2, :, :, slice_idx]

                        rgb_mask = np.zeros((*flair_slice.shape, 3))
                        rgb_mask[wt_mask > 0] = [1.0, 0.2, 0.2]
                        rgb_mask[tc_mask > 0] = [0.2, 0.8, 0.2]
                        rgb_mask[et_slice > 0] = [1.0, 0.9, 0.1]

                        with col1:
                            st.subheader("1. Ảnh MRI Gốc (FLAIR)")
                            fig1, ax1 = plt.subplots()
                            ax1.imshow(flair_slice, cmap="gray")
                            ax1.axis("off")
                            st.pyplot(fig1)
                            plt.close(fig1)

                        with col2:
                            st.subheader("2. Mặt Nạ Khối U (AI Mask)")
                            fig2, ax2 = plt.subplots()
                            ax2.imshow(rgb_mask)
                            ax2.axis("off")
                            st.pyplot(fig2)
                            plt.close(fig2)

                        with col3:
                            st.subheader("3. Kết Quả Tô Màu Khối U (Overlay)")
                            fig3, ax3 = plt.subplots()
                            ax3.imshow(flair_slice, cmap="gray")
                            ax3.imshow(rgb_mask, alpha=0.45)
                            ax3.axis("off")
                            st.pyplot(fig3)
                            plt.close(fig3)

                        st.markdown("### 🎨 Chú thích màu sắc phân vùng u:")
                        st.markdown("- 🔴 **Màu Đỏ**: **Whole Tumor (WT)** - Toàn bộ vùng khối u & phù nề.")
                        st.markdown("- 🟢 **Màu Xanh Lá**: **Tumor Core (TC)** - Lõi khối u đặc.")
                        st.markdown("- 🟡 **Màu Vàng**: **Enhancing Tumor (ET)** - Vùng u hoạt hóa sinh trưởng mạnh.")

                    except Exception as e:
                        st.error(f"Lỗi khi thực thi phân đoạn: {e}")
else:
    st.warning("👈 Vui lòng chọn **Sử dụng dữ liệu Mô Phỏng** hoặc tải lên file **.nii.gz** ở thanh bên trái để xem giao diện demo!")
