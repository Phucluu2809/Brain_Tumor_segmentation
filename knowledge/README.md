# BÁO CÁO PHÂN ĐOẠN KHỐI U NÃO TRÊN ẢNH MRI ĐA CHUỖI XUNG

## Thực nghiệm U-Net/TransUNet và đề xuất kiến trúc DAU-Mamba

> **Phạm vi báo cáo.** Phần thực nghiệm trong tài liệu này được tổng hợp trực
> tiếp từ mã nguồn, cấu hình, checkpoint và log của dự án
> `HK7-CV-Class/brats-transunet`. Kiến trúc **DAU-Mamba** và **CGR-Mamba** là
> các thiết kế đề xuất trong thư mục `knowledge`, hiện **chưa được cài đặt hoặc
> kiểm chứng thực nghiệm**. Hai nhóm nội dung được tách rõ để tránh xem kết quả
> của U-Net/TransUNet là kết quả của Mamba.

---

## Tóm tắt

Phân đoạn tự động khối u não từ ảnh cộng hưởng từ đa chuỗi xung là một bài toán
quan trọng trong hỗ trợ chẩn đoán, lập kế hoạch điều trị và theo dõi tiến triển
u thần kinh đệm. Báo cáo khảo sát các nền tảng U-Net, Transformer và Selective
State Space Model (Mamba), sau đó đánh giá hai mô hình đã được triển khai là
U-Net 2D và TransUNet 2D trên một tập con 200 ca của BraTS 2021. Cả hai mô hình
nhận bốn chuỗi MRI T1, T1ce, T2 và FLAIR, đồng thời dự đoán ba vùng lồng nhau:
Whole Tumor (WT), Tumor Core (TC) và Enhancing Tumor (ET).

Trên 30 ca kiểm thử, U-Net 2D đạt Dice trung bình **0,8521**, cao hơn TransUNet
2D (**0,7997**) trong điều kiện huấn luyện hiện tại. Kết quả cho thấy mô hình
Transformer decoder chưa phát huy lợi thế khi dữ liệu, thời gian huấn luyện và
phép điều chuẩn còn hạn chế. Từ các kết quả này và các công trình U-Mamba,
MedMamba, SegMamba, SegResMamba, báo cáo đề xuất DAU-Mamba: một mạng 3D lai
CNN–Mamba sử dụng hợp nhất hướng thích nghi trong encoder và hợp nhất skip có
hướng dẫn bất định trong decoder. Kiến trúc đề xuất hướng tới khả năng mô hình
hóa ngữ cảnh toàn cục với chi phí tuyến tính theo độ dài chuỗi, đồng thời giữ
được đặc trưng cục bộ và biên tổn thương của CNN.

**Từ khóa:** BraTS 2021, MRI, phân đoạn u não, U-Net, TransUNet, Mamba, SSM,
DAU-Mamba, Dice, HD95.

---

# 1. TỔNG QUAN

## 1.1. Bài toán

Ảnh MRI của một bệnh nhân BraTS gồm bốn chuỗi xung bổ trợ lẫn nhau:

| Chuỗi xung | Vai trò thường gặp trong quan sát tổn thương |
|---|---|
| **T1** | Thể hiện cấu trúc giải phẫu và ranh giới mô não cơ bản. |
| **T1ce** | Làm nổi bật vùng u bắt thuốc sau tiêm chất tương phản, đặc biệt hữu ích cho ET. |
| **T2** | Nhạy với thành phần dịch và vùng tổn thương có tín hiệu cao. |
| **FLAIR** | Khử tín hiệu dịch não tủy, giúp vùng phù nề/quanh u nổi bật hơn. |

Mục tiêu là học ánh xạ từ thể tích MRI đa kênh

\[
X \in \mathbb{R}^{4\times H\times W\times D}
\]

sang ba mặt nạ xác suất

\[
\hat{Y} \in [0,1]^{3\times H\times W\times D},
\]

tương ứng với:

- **WT — Whole Tumor:** toàn bộ vùng u;
- **TC — Tumor Core:** lõi u;
- **ET — Enhancing Tumor:** vùng u tăng cường.

Ba vùng có quan hệ lồng nhau về giải phẫu: \(ET\subseteq TC\subseteq WT\).
Vì vậy, dự án dùng ba kênh sigmoid thay vì softmax loại trừ lẫn nhau.

## 1.2. Động cơ nghiên cứu

Phân đoạn thủ công theo từng lát cắt tốn thời gian và phụ thuộc kinh nghiệm của
chuyên gia. Một mô hình tự động tốt cần đồng thời:

1. nhận biết chi tiết cục bộ và biên tổn thương nhỏ;
2. sử dụng ngữ cảnh xa để phân biệt u với cấu trúc não có cường độ tương tự;
3. xử lý được thể tích 3D có số voxel lớn trong giới hạn bộ nhớ;
4. duy trì tính ổn định khi kích thước, vị trí và hình thái u thay đổi.

CNN, đặc biệt là U-Net, có thiên kiến cục bộ phù hợp với dữ liệu ảnh và thường
hoạt động tốt khi tập dữ liệu chưa lớn. Transformer bổ sung khả năng mô hình hóa
phụ thuộc xa nhưng self-attention có độ phức tạp bậc hai theo số token. Mamba sử
dụng Selective State Space Model, có tiềm năng thu nhận ngữ cảnh dài với độ
phức tạp tuyến tính; đây là lý do chọn hướng CNN–Mamba cho kiến trúc đề xuất.

## 1.3. Mục tiêu

- Xây dựng pipeline đọc và tiền xử lý dữ liệu BraTS 2021 trực tiếp từ `.tar`,
  thư mục NIfTI hoặc cache `.npz`.
- Huấn luyện và so sánh U-Net 2D với TransUNet 2D trong cùng một phân hoạch dữ
  liệu và ngân sách huấn luyện.
- Đánh giá theo Dice và Hausdorff Distance 95% (HD95) trên tập test độc lập.
- Tích hợp checkpoint vào giao diện Streamlit để trình diễn dự đoán theo lát cắt.
- Tổng hợp cơ sở lý thuyết Mamba và đề xuất DAU-Mamba cho phân đoạn 3D trong
  giai đoạn tiếp theo.

## 1.4. Đóng góp của dự án

1. Một pipeline có thể tái lập từ dữ liệu NIfTI đến huấn luyện, đánh giá và
   trực quan hóa.
2. Một phép so sánh thực nghiệm công bằng giữa baseline CNN và decoder dựa trên
   query/Transformer trên cùng 30 ca test.
3. Giao diện demo chạy hai checkpoint, hiển thị MRI, ground truth, mask dự đoán,
   overlay và Dice theo vùng.
4. Một thiết kế 3D CNN–Mamba mới ở mức đề xuất, kết hợp **Direction-Adaptive
   Split Hybrid (DSH)** và **Uncertainty-Guided Skip Fusion (UGSF)**.

## 1.5. Phạm vi và trạng thái các thành phần

| Thành phần | Trạng thái | Bằng chứng trong dự án |
|---|---|---|
| U-Net 2D | Đã cài đặt, huấn luyện và test | mã nguồn, `best.pt`, `log.csv`, `metrics.csv` |
| TransUNet 2D decoder-only | Đã cài đặt, huấn luyện và test | mã nguồn, `best.pt`, `log.csv`, `metrics.csv` |
| Ứng dụng Streamlit | Đã cài đặt và smoke test | `app.py`, `brats_demo/`, checkpoint demo |
| DAU-Mamba 3D | Thiết kế đề xuất | `DAU_Mamba_Kien_truc.md` và hình kiến trúc bên dưới |
| CGR-Mamba 3D | Thiết kế đề xuất thay thế | `CGR_Mamba_Kien_Truc.md` |

---

# 2. CƠ SỞ LÝ THUYẾT

## 2.1. Dữ liệu BraTS 2021 và biểu diễn nhãn

BraTS 2021 cung cấp dữ liệu MRI đa tham số đã được đồng đăng ký, nội suy về cùng
không gian và loại bỏ hộp sọ. Nhãn gốc gồm các mã mô tổn thương; pipeline chuyển
chúng thành ba vùng đánh giá lồng nhau WT, TC và ET. Với mỗi voxel, đầu ra có
thể chứa đồng thời nhiều nhãn vùng, ví dụ một voxel ET cũng thuộc TC và WT.

Trong dự án hiện tại, một ca có dạng gần với:

```text
BraTS2021_xxxxx/
├── BraTS2021_xxxxx_t1.nii.gz
├── BraTS2021_xxxxx_t1ce.nii.gz
├── BraTS2021_xxxxx_t2.nii.gz
├── BraTS2021_xxxxx_flair.nii.gz
└── BraTS2021_xxxxx_seg.nii.gz
```

## 2.2. Tiền xử lý

### Chuẩn hóa cường độ

MRI không có thang cường độ tuyệt đối chung giữa bệnh nhân. Với mỗi chuỗi xung,
dự án chuẩn hóa z-score trên vùng não khác nền:

\[
x' = \frac{x-\mu_{brain}}{\sigma_{brain}+\epsilon}.
\]

Phép chuẩn hóa theo từng ca và từng modality giảm sai khác do máy chụp và quy
trình thu nhận.

### Lấy mẫu patch có ưu tiên vùng u

U chiếm tỷ lệ voxel nhỏ so với nền. Nếu lấy patch hoàn toàn ngẫu nhiên, phần lớn
batch có thể không chứa tổn thương. Cấu hình dùng `fg_prob = 0.5`: khoảng 50%
patch được tâm hóa quanh vùng foreground, giúp cân bằng tín hiệu học.

### Tăng cường dữ liệu

Phép biến đổi affine ngẫu nhiên được áp dụng với xác suất 0,2. Ảnh MRI dùng nội
suy liên tục, trong khi mask dùng nội suy nearest-neighbor để không tạo nhãn giả.

## 2.3. U-Net

U-Net gồm hai nhánh đối xứng [1]:

- **Encoder/contracting path:** tích chập và giảm kích thước để học đặc trưng có
  ngữ nghĩa cao;
- **Decoder/expanding path:** tăng kích thước để khôi phục độ phân giải;
- **Skip connection:** nối đặc trưng encoder cùng tỷ lệ vào decoder, bù lại chi
  tiết không gian bị mất do downsampling.

U-Net 2D của dự án sử dụng các mức kênh `[32, 64, 128, 256, 320]`, nhận patch
`4 × 160 × 160` và trả về ba logit WT/TC/ET. Hàm mất mát là:

\[
\mathcal{L}_{U-Net}=\mathcal{L}_{BCE}+\left(1-DSC_{soft}\right).
\]

BCE cung cấp giám sát theo pixel, còn Dice loss tập trung vào độ chồng lấp và
giảm ảnh hưởng của mất cân bằng foreground/background.

## 2.4. Transformer, UNETR và TransUNet đang triển khai

Self-attention cho phép mỗi token trao đổi trực tiếp với mọi token khác, nhờ đó
mô hình hóa quan hệ xa tốt hơn CNN thuần. Tuy nhiên, với \(N\) token, attention
chuẩn có chi phí thời gian/bộ nhớ xấp xỉ \(O(N^2)\). UNETR [2] dùng Transformer
làm encoder cho thể tích 3D và nối đặc trưng nhiều tầng sang decoder CNN.

Tên “TransUNet” trong mã nguồn dự án cần được hiểu theo **cấu hình thực tế**:

- một U-Net CNN tạo đặc trưng đa tỷ lệ;
- Transformer decoder có 20 query học được;
- ba lớp decoder coarse-to-fine lần lượt cross-attend lên đặc trưng tỷ lệ
  `1/8`, `1/4`, `1/2`;
- masked attention giới hạn vùng chú ý dựa trên mask dự đoán ở lớp trước;
- mỗi query dự đoán một lớp trong `{WT, TC, ET, no-object}` và một mask;
- Hungarian matching ghép tối ưu query với các vùng ground truth không rỗng.

Chi phí ghép cặp dùng trọng số `(2, 5, 5)` cho lớp, BCE mask và Dice mask:

\[
\mathcal{C}=2\mathcal{C}_{class}+5\mathcal{C}_{BCE}+5\mathcal{C}_{Dice}.
\]

Do U-Net và TransUNet dùng hai định nghĩa loss khác nhau, trị số `train_loss`
giữa hai log **không được so sánh trực tiếp**.

## 2.5. State Space Model và Mamba

Một hệ trạng thái liên tục tuyến tính có thể viết:

\[
h'(t)=Ah(t)+Bx(t), \qquad y(t)=Ch(t)+Dx(t),
\]

trong đó \(h\) là trạng thái ẩn, \(x\) là đầu vào và \(y\) là đầu ra. Sau khi
rời rạc hóa với bước \(\Delta\):

\[
h_t=\bar{A}h_{t-1}+\bar{B}x_t, \qquad y_t=Ch_t+Dx_t.
\]

Mamba [3] làm các tham số chọn lọc phụ thuộc đầu vào, cho phép mô hình quyết
định thông tin nào cần giữ hoặc quên theo từng token. Thuật toán selective scan
được tối ưu cho phần cứng và có độ phức tạp tuyến tính theo chiều dài chuỗi, phù
hợp hơn attention khi số voxel/token rất lớn.

Khi áp dụng cho ảnh hoặc thể tích, tensor không có thứ tự một chiều tự nhiên.
Do đó cần quy định hướng quét, chẳng hạn trước–sau, sau–trước hoặc theo lát cắt.
Nhiều hướng giúp giảm thiên lệch do một phép tuyến tính hóa duy nhất.

## 2.6. Các công trình CNN–Mamba liên quan

| Công trình | Ý tưởng liên quan được kế thừa hoặc tham khảo |
|---|---|
| **U-Mamba** [5] | Kết hợp CNN cục bộ và SSM toàn cục trong khung U-shaped tự cấu hình. |
| **MedMamba** [6] | Khối SS-Conv-SSM chia nhánh CNN/SSM, grouped convolution và channel shuffle; bài toán gốc là phân loại ảnh y khoa. |
| **SegMamba** [7] | Mô hình hóa thể tích 3D, quét theo nhiều hướng và hợp nhất đặc trưng toàn cục. |
| **SegResMamba** [8] | Thiết kế phân đoạn 3D theo hướng hiệu quả, nhấn mạnh khối Mamba gọn và decoder nhẹ. |

## 2.7. Kiến trúc đề xuất DAU-Mamba

![Kiến trúc DAU-Mamba đề xuất](assets/dau_mamba_architecture.png)

**Hình 1.** Kiến trúc DAU-Mamba đề xuất. Hình được dựng mới cho báo cáo từ đặc
tả trong `DAU_Mamba_Kien_truc.md`; không phải kết quả sao chép từ các paper.

### 2.7.1. Luồng tổng thể

Đầu vào là patch 3D bốn modality kích thước `4 × 96 × 96 × 96`. Encoder có năm
mức `[32, 64, 128, 256, 320]`. Hai mức đầu dùng CNN để giữ chi tiết cục bộ;
các mức E2, E3 và bottleneck dùng DSH để bổ sung ngữ cảnh xa với chi phí hợp lý
do độ phân giải đã giảm. Decoder tăng dần kích thước và sử dụng UGSF tại bốn
mức để lọc thông tin từ skip connection trước khi hợp nhất.

### 2.7.2. Direction-Adaptive Split Hybrid (DSH)

Với tensor đầu vào \(F\), DSH chiếu và chia kênh theo tỷ lệ 1:1:

\[
F_l,F_g=Split(Proj(F)).
\]

Nhánh cục bộ dùng 3D CNN:

\[
L=Conv_{3D}(F_l),
\]

trong khi nhánh toàn cục dùng ba phép quét Mamba độc lập:

\[
M_f=Mamba_f(F_g),\quad
M_r=Mamba_r(F_g),\quad
M_s=Mamba_s(F_g).
\]

Ba trọng số hướng được dự đoán theo từng mẫu và chuẩn hóa bằng softmax:

\[
[\alpha_f,\alpha_r,\alpha_s]
=Softmax(MLP(GAP(F_g))).
\]

Đặc trưng toàn cục là:

\[
G=\alpha_fM_f+\alpha_rM_r+\alpha_sM_s.
\]

Hai nhánh được nối, xáo trộn kênh và cộng residual:

\[
F_{out}=F+Proj(ChannelShuffle([L;G])).
\]

Điểm khác biệt so với cộng đều các hướng là mô hình có thể điều chỉnh đóng góp
của từng hướng theo hình thái và vị trí u trong từng ca.

### 2.7.3. Uncertainty-Guided Skip Fusion (UGSF)

Tại mỗi mức decoder, một đầu dự đoán phụ tạo xác suất vùng \(p\). Bản đồ entropy
nhị phân được tính theo:

\[
U=-p\log(p+\epsilon)-(1-p)\log(1-p+\epsilon).
\]

Vùng có entropy cao là vùng mô hình chưa chắc chắn, thường nằm quanh ranh giới
hoặc tổn thương nhỏ. Gate theo voxel nhận đặc trưng encoder \(E\), decoder \(V\)
và entropy \(U\):

\[
G=\sigma(Conv([E;V;U])).
\]

Đặc trưng hợp nhất:

\[
F_{fuse}=G\odot E+(1-G)\odot V.
\]

Thay vì chuyển toàn bộ đặc trưng nông qua skip connection, UGSF học cách ưu tiên
thông tin encoder hoặc decoder theo mức bất định tại từng voxel.

### 2.7.4. Deep supervision và ràng buộc vùng lồng nhau

Các đầu phụ trong decoder được giám sát sau khi nội suy về độ phân giải mục
tiêu. Ở suy luận, quan hệ lồng nhau có thể được bảo đảm trên xác suất:

\[
p_{TC}\leftarrow\max(p_{TC},p_{ET}),\qquad
p_{WT}\leftarrow\max(p_{WT},p_{TC}).
\]

Đây là bước hậu xử lý xác định, không thay thế việc học quan hệ giải phẫu trong
mô hình.

## 2.8. So sánh DAU-Mamba và CGR-Mamba

Hai thiết kế trong thư mục `knowledge` giải quyết cùng bài toán nhưng có chiến
lược khác nhau:

| Tiêu chí | DAU-Mamba | CGR-Mamba |
|---|---|---|
| Vị trí Mamba | E2, E3, bottleneck | Một khối refinement tại F2 |
| Cách dùng CNN | CNN ở tầng nông, CNN–Mamba ở tầng sâu | Một backbone residual CNN dùng chung |
| Tín hiệu hướng dẫn | Bất định tại từng mức decoder | Dự đoán coarse từ backbone |
| Hợp nhất cuối | Decoder UGSF trực tiếp tạo mask | `Z_final = Z_c + G ⊙ ΔZ` |
| Ưu điểm kỳ vọng | Ngữ cảnh đa tỷ lệ, tập trung vùng khó | Nhẹ hơn, sửa dự đoán coarse có kiểm soát |
| Rủi ro | Nhiều khối Mamba, tốn bộ nhớ hơn | Refinement tại một tỷ lệ có thể bỏ lỡ chi tiết |

Trong báo cáo này DAU-Mamba là kiến trúc đề xuất chính; CGR-Mamba là một
baseline kiến trúc hợp lý cho thí nghiệm ablation hoặc nghiên cứu tiếp theo.

---

# 3. TRIỂN KHAI THỬ NGHIỆM VÀ ĐÁNH GIÁ

## 3.1. Cấu trúc hệ thống

```text
XLA/
├── HK7-CV-Class/
│   └── brats-transunet/
│       ├── configs/                 # cấu hình U-Net/TransUNet 2D và 3D
│       ├── data/                    # index và split dữ liệu
│       ├── src/brats/
│       │   ├── datasets.py          # lấy mẫu patch
│       │   ├── preprocess.py        # chuẩn hóa và chuyển nhãn
│       │   ├── losses.py            # Dice+BCE, Hungarian SetCriterion
│       │   ├── inference.py         # suy luận 2D/3D
│       │   ├── metrics.py           # Dice, HD95
│       │   └── models/              # U-Net, TransUNet, ViT decoder
│       ├── train.py
│       ├── evaluate.py
│       ├── runs/                    # checkpoint, log, kết quả test
│       └── figures/                 # đường học và ảnh dự đoán
└── Brain-Tumor-segmentation/
    ├── app.py                       # giao diện Streamlit
    ├── brats_demo/                  # model và pipeline demo độc lập
    ├── scripts/export_demo_models.py
    ├── scripts/smoke_demo.py
    └── knowledge/
        ├── README.md                # báo cáo này
        ├── assets/                  # hình kiến trúc
        ├── *_Kien_truc.md           # đặc tả DAU/CGR-Mamba
        └── *.pdf                    # paper và slide tham khảo
```

## 3.2. Môi trường

Các thư viện chính gồm Python 3.11, PyTorch, NumPy, SciPy, NiBabel, PyYAML,
Matplotlib và Streamlit. Huấn luyện hỗ trợ CUDA AMP; nếu không có GPU vẫn có
thể chạy suy luận/demo trên CPU nhưng chậm hơn.

## 3.3. Phân hoạch dữ liệu

File `data/splits_200.json` chia 200 ca không trùng nhau:

| Tập | Số ca | Mục đích |
|---|---:|---|
| Train | 140 | Cập nhật tham số mô hình |
| Validation | 30 | Chọn checkpoint và siêu tham số |
| Test | 30 | Báo cáo kết quả cuối |

Trong lúc huấn luyện, cấu hình chỉ đánh giá **10 ca validation** để tiết kiệm
thời gian. Kết quả test phía dưới được tính trên toàn bộ 30 ca. Seed cố định là
42. Ca demo `BraTS2021_00495` không thuộc split 200 của hai run chính; do đó chỉ
được dùng như minh họa định tính/kiểm tra giao diện, không cộng vào test mean.

## 3.4. Cấu hình huấn luyện

| Thuộc tính | Giá trị |
|---|---|
| Không gian huấn luyện | 2D axial patch |
| Kích thước patch | `160 × 160` |
| Batch size | 8 |
| Số epoch đã chạy | 20 |
| Iteration/epoch | 250 |
| Optimizer | AdamW |
| Learning rate cực đại | `3 × 10⁻⁴` |
| Weight decay | `3 × 10⁻⁵` |
| Warm-up | 2 epoch |
| AMP | bật |
| Xác suất patch foreground | 0,5 |
| Affine augmentation | xác suất 0,2 |
| Chu kỳ validation | mỗi 5 epoch |
| Ngưỡng nhị phân | 0,5 |

Hai mô hình dùng cùng dữ liệu và lịch học, nhưng khác kiến trúc và loss:

| Mô hình | Tham số | Loss | Checkpoint tốt nhất |
|---|---:|---|---:|
| U-Net 2D | 5,686 triệu | BCE + soft Dice | epoch 19 |
| TransUNet 2D | 8,568 triệu | Hungarian: class + BCE mask + Dice mask | epoch 4 |

## 3.5. Quy trình huấn luyện

```text
NIfTI/.tar/.npz
      ↓
đọc 4 modality + segmentation
      ↓
chuẩn hóa từng modality trên vùng não
      ↓
chuyển nhãn sang WT/TC/ET
      ↓
lấy patch có/không có u + augmentation
      ↓
forward → loss → AMP backward → AdamW
      ↓
validation định kỳ → lưu best.pt và last.pt
```

Các lệnh tái lập điển hình, chạy trong `HK7-CV-Class/brats-transunet`:

```powershell
python train.py --config configs/unet2d.yaml `
  --opts splits=data/splits_200.json output_dir=runs/unet2d_200 train.epochs=20

python train.py --config configs/transunet2d.yaml `
  --opts splits=data/splits_200.json output_dir=runs/tu2d_200 train.epochs=20
```

Đánh giá checkpoint tốt nhất:

```powershell
python evaluate.py --config configs/unet2d.yaml `
  --checkpoint runs/unet2d_200/best.pt --split test

python evaluate.py --config configs/transunet2d.yaml `
  --checkpoint runs/tu2d_200/best.pt --split test
```

Đường học đã lưu trong dự án:

![Đường học U-Net và TransUNet](../../HK7-CV-Class/brats-transunet/figures/training_curves.png)

**Hình 2.** Loss huấn luyện và Dice validation theo log của hai run. Do hai hàm
loss khác thang đo, chỉ nên so xu hướng hội tụ, không so trực tiếp độ lớn loss.

## 3.6. Chỉ số đánh giá

### Dice Similarity Coefficient

Với tập voxel dự đoán \(P\) và ground truth \(G\):

\[
DSC(P,G)=\frac{2|P\cap G|+\epsilon}{|P|+|G|+\epsilon}.
\]

Dice càng gần 1 càng tốt. Báo cáo tính riêng WT, TC, ET rồi lấy trung bình ba
vùng cho từng ca và trung bình trên 30 ca.

### Hausdorff Distance 95%

HD95 là phân vị 95% của khoảng cách biên hai chiều giữa dự đoán và ground truth:

\[
HD95(P,G)=\max\{Q_{0.95}(d(\partial P,\partial G)),
Q_{0.95}(d(\partial G,\partial P))\}.
\]

Giá trị càng thấp càng tốt. So với Hausdorff cực đại, HD95 ít nhạy hơn với một
số ít voxel ngoại lai.

## 3.7. Kết quả validation trong khi huấn luyện

| Mô hình | Epoch tốt nhất | Dice WT | Dice TC | Dice ET | Dice mean |
|---|---:|---:|---:|---:|---:|
| **U-Net 2D** | 19 | 0,9353 | 0,8899 | 0,8904 | **0,9052** |
| TransUNet 2D | 4 | 0,8881 | 0,8556 | 0,8608 | 0,8682 |

Các số này được tính trên 10 ca validation trong lúc train. TransUNet đạt đỉnh
sớm rồi giảm ở TC, cho thấy khả năng overfit hoặc bất ổn ghép query trong cấu
hình 20 epoch hiện tại. U-Net cải thiện đều hơn đến epoch 19.

## 3.8. Kết quả trên 30 ca test

### Dice

| Mô hình | WT ↑ | TC ↑ | ET ↑ | Mean ↑ |
|---|---:|---:|---:|---:|
| **U-Net 2D** | **0,8701** | **0,8566** | **0,8296** | **0,8521** |
| TransUNet 2D | 0,8288 | 0,7820 | 0,7885 | 0,7997 |
| Chênh lệch U-Net − TransUNet | +0,0413 | +0,0747 | +0,0411 | +0,0524 |

### HD95

| Mô hình | WT ↓ | TC ↓ | ET ↓ |
|---|---:|---:|---:|
| **U-Net 2D** | **16,88** | **12,78** | **9,66** |
| TransUNet 2D | 46,32 | 35,07 | 29,62 |

Trong cả Dice và HD95, U-Net vượt TransUNet ở cấu hình hiện tại. Khoảng cách
lớn ở HD95 cho thấy TransUNet có nhiều ca xuất hiện dự đoán xa biên thật hoặc
mask rời rạc. Kết quả không chứng minh CNN luôn tốt hơn Transformer; nó chỉ cho
thấy baseline CNN phù hợp hơn trong ngân sách dữ liệu/epoch và cấu hình decoder
đã dùng.

## 3.9. Kết quả ca demo

Với `BraTS2021_00495`, pipeline demo end-to-end cho kết quả:

| Mô hình | WT | TC | ET | Mean |
|---|---:|---:|---:|---:|
| **U-Net 2D** | 0,9589 | 0,9679 | 0,9230 | **0,9499** |
| TransUNet 2D | 0,8584 | 0,9321 | 0,8976 | 0,8960 |

Đây là một ca minh họa tương đối dễ và **không đại diện** cho phân bố 30 ca
test. Không nên dùng hai con số mean này thay cho kết quả test chính thức.

![Ví dụ trực quan BraTS2021_00495](../../HK7-CV-Class/brats-transunet/figures/BraTS2021_00495.png)

**Hình 3.** Bốn modality, ground truth và kết quả dự đoán của ca minh họa.

## 3.10. Giao diện demo

Ứng dụng `Brain-Tumor-segmentation/app.py` cho phép:

- chọn U-Net 2D hoặc TransUNet 2D;
- dùng ca mẫu đóng gói sẵn hoặc tải lên 4 file NIfTI;
- tùy chọn nạp ground truth để tính Dice;
- duyệt lát cắt axial;
- hiển thị từng modality, mask, overlay và thể tích vùng u;
- tải kết quả dự đoán về ở định dạng NIfTI.

Khởi chạy trong thư mục `Brain-Tumor-segmentation`:

```powershell
python -m streamlit run app.py
```

Ứng dụng phục vụ mục đích nghiên cứu/giảng dạy, không phải thiết bị y tế và
không thay thế kết luận của bác sĩ.

## 3.11. Phân tích kết quả

### Vì sao U-Net đang tốt hơn?

1. **Thiên kiến cục bộ phù hợp:** cấu trúc và biên u được học hiệu quả bằng tích
   chập khi dữ liệu hữu hạn.
2. **Tối ưu đơn giản hơn:** U-Net dự đoán trực tiếp ba kênh; TransUNet phải học
   đồng thời lớp, mask, no-object và phép phân công query.
3. **Ngân sách huấn luyện ngắn:** 20 epoch có thể chưa đủ để decoder query ổn
   định trên các ca khó.
4. **Kích thước mô hình:** TransUNet có nhiều hơn khoảng 2,88 triệu tham số và
   mất khoảng 202–211 giây/epoch, so với khoảng 78–91 giây/epoch của U-Net trong
   log hiện có.
5. **Thông tin 3D bị giới hạn:** cả hai run chính đều huấn luyện theo lát cắt 2D,
   không khai thác trực tiếp liên tục giữa các lát.

### Diễn giải theo vùng

- WT thường dễ hơn vì có kích thước lớn và FLAIR làm vùng phù rõ.
- TC và ET nhỏ hơn, biến thiên hình thái mạnh và nhạy với sai số vài voxel.
- Chênh lệch TC lớn nhất giữa hai mô hình cho thấy decoder query chưa ổn định
  khi lõi u nhỏ hoặc không đồng nhất.

## 3.12. Hạn chế và đe dọa tính hợp lệ

- Chỉ dùng 200 ca và 20 epoch, nhỏ hơn quy mô đầy đủ của BraTS 2021.
- Validation trong train chỉ dùng 10/30 ca; checkpoint tốt nhất có thể phụ thuộc
  vào tập con này.
- Chưa báo cáo khoảng tin cậy, kiểm định thống kê hoặc đánh giá chéo nhiều seed.
- Thực nghiệm 2D bỏ qua ngữ cảnh theo trục z.
- Chưa đánh giá FLOPs, peak VRAM và thời gian suy luận theo cùng một phần cứng.
- Chưa kiểm thử ngoài miền trên dữ liệu từ cơ sở y tế khác.
- Kết quả `Dice mean = 0,7328` trong slide
  `Phân đoạn khối u não bằng TransUNet3D.pdf` là kết quả lịch sử; không có cùng
  lineage log/checkpoint với hai run đang báo cáo nên **không được dùng** trong
  bảng kết quả chính.
- DAU-Mamba và CGR-Mamba mới là đặc tả thiết kế; mọi ưu điểm của chúng hiện là
  giả thuyết cần kiểm chứng.

## 3.13. Kế hoạch kiểm chứng DAU-Mamba

Để chứng minh đóng góp của kiến trúc đề xuất, cần chạy ít nhất các ablation sau
trên cùng split, seed và ngân sách:

| Thí nghiệm | Backbone/khối | Mục đích |
|---|---|---|
| A0 | 3D residual U-Net | Baseline CNN 3D |
| A1 | A0 + DSH, cộng đều ba hướng | Đo lợi ích của Mamba đa hướng |
| A2 | A1 + trọng số hướng thích nghi | Đo đóng góp direction-adaptive |
| A3 | A0 + UGSF | Đo đóng góp uncertainty-guided skip |
| A4 | DSH thích nghi + UGSF | Kiến trúc DAU-Mamba đầy đủ |
| A5 | CGR-Mamba | So sánh decoder tích hợp với refinement residual |

Mỗi thí nghiệm nên báo cáo:

- Dice và HD95 theo WT/TC/ET;
- mean ± standard deviation trên tối thiểu ba seed;
- số tham số, FLOPs, peak VRAM, thời gian train/epoch và thời gian suy luận/ca;
- kết quả riêng cho u nhỏ, u lớn và trường hợp region rỗng;
- ảnh định tính gồm ca tốt, ca trung bình và ca thất bại.

## 3.14. Kết luận và hướng phát triển

Trong thí nghiệm đã hoàn thành, U-Net 2D là mô hình tốt nhất với Dice test trung
bình 0,8521 và biên dự đoán ổn định hơn theo HD95. TransUNet 2D cho thấy tiềm
năng trên một số ca nhưng cần điều chuẩn lâu hơn, kiểm soát query/no-object và
khai thác dữ liệu lớn hơn.

Hướng phát triển tiếp theo là chuyển sang patch 3D và đánh giá DAU-Mamba. DSH
được kỳ vọng cân bằng đặc trưng cục bộ/toàn cục và chọn hướng quét phù hợp theo
từng ca; UGSF dùng entropy để tập trung vào vùng biên chưa chắc chắn. Tuy nhiên,
kiến trúc chỉ nên được xem là đóng góp khoa học sau khi vượt qua ablation, đánh
giá nhiều seed và so sánh công bằng về tài nguyên.

---

# 4. TÀI LIỆU THAM KHẢO

[1] O. Ronneberger, P. Fischer, T. Brox, **“U-Net: Convolutional Networks for
Biomedical Image Segmentation,”** MICCAI, 2015.  
PDF trong thư mục: [`1505.04597v1.pdf`](1505.04597v1.pdf) ·
[arXiv](https://arxiv.org/abs/1505.04597) ·
[DOI](https://doi.org/10.1007/978-3-319-24574-4_28)

[2] A. Hatamizadeh et al., **“UNETR: Transformers for 3D Medical Image
Segmentation,”** WACV, 2022.  
PDF trong thư mục: [`2103.10504v3.pdf`](2103.10504v3.pdf) ·
[arXiv](https://arxiv.org/abs/2103.10504) ·
[MONAI Research](https://monai.io/research/unetr)

[3] A. Gu, T. Dao, **“Mamba: Linear-Time Sequence Modeling with Selective State
Spaces,”** 2023.  
[arXiv](https://arxiv.org/abs/2312.00752) ·
[Mã nguồn chính thức](https://github.com/state-spaces/mamba)

[4] U. Baid et al., **“The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor
Segmentation and Radiogenomic Classification,”** 2021.  
[arXiv](https://arxiv.org/abs/2107.02314) ·
[BraTS](https://www.med.upenn.edu/cbica/brats2021/)

[5] J. Ma, F. Li, B. Wang, **“U-Mamba: Enhancing Long-range Dependency for
Biomedical Image Segmentation,”** 2024.  
[arXiv](https://arxiv.org/abs/2401.04722) ·
[Trang dự án](https://wanglab.ai/u-mamba.html)

[6] Y. Yue, Z. Li, **“MedMamba: Vision Mamba for Medical Image Classification,”**
2024.  
PDF trong thư mục:
[`MedMamba_Vision_Mamba_2024.pdf`](MedMamba_Vision_Mamba_2024.pdf) ·
[arXiv](https://arxiv.org/abs/2403.03849) ·
[Mã nguồn](https://github.com/YubiaoYue/MedMamba)

[7] Z. Xing, T. Ye, Y. Yang, G. Liu, L. Zhu, **“SegMamba: Long-range Sequential
Modeling Mamba for 3D Medical Image Segmentation,”** 2024.  
PDF trong thư mục:
[`SegMamba_3D_Medical_Segmentation_2024.pdf`](SegMamba_3D_Medical_Segmentation_2024.pdf) ·
[arXiv](https://arxiv.org/abs/2401.13560) ·
[Mã nguồn](https://github.com/ge-xing/SegMamba)

[8] D. Das et al., **“SegResMamba: An Efficient 3D Medical Image Segmentation
Architecture with Mamba,”** 2025.  
PDF trong thư mục:
[`SegResMamba_Efficient_3D_Segmentation_2024.pdf`](SegResMamba_Efficient_3D_Segmentation_2024.pdf) ·
[arXiv](https://arxiv.org/abs/2503.07766)

[9] Đặc tả nội bộ, **“DAU-Mamba: Direction-Adaptive and Uncertainty-Guided
CNN–Mamba,”** 2026.  
[`DAU_Mamba_Kien_truc.md`](DAU_Mamba_Kien_truc.md)

[10] Đặc tả nội bộ, **“CGR-Mamba: Coarse-Guided Residual Mamba Network,”** 2026.  
[`CGR_Mamba_Kien_Truc.md`](CGR_Mamba_Kien_Truc.md)

[11] Nhóm dự án, **“Phân đoạn khối u não bằng TransUNet3D,”** slide báo cáo lịch
sử.  
[`Phân đoạn khối u não bằng TransUNet3D.pdf`](Ph%C3%A2n%20%C4%91o%E1%BA%A1n%20kh%E1%BB%91i%20u%20n%C3%A3o%20b%E1%BA%B1ng%20TransUNet3D.pdf)

## Liên kết mã nguồn dự án

- [Kho huấn luyện HK7-CV-Class](https://github.com/nhatminh65537/HK7-CV-Class)
- [Kho giao diện Brain Tumor Segmentation](https://github.com/Phucluu2809/Brain_Tumor_segmentation)
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [NiBabel](https://nipy.org/nibabel/)
- [Streamlit](https://docs.streamlit.io/)

---

## Phụ lục A — Nguồn số liệu trong báo cáo

| Nội dung | File nguồn |
|---|---|
| Log U-Net | `../../HK7-CV-Class/brats-transunet/runs/unet2d_200/log.csv` |
| Test U-Net | `../../HK7-CV-Class/brats-transunet/runs/unet2d_200/eval_test/metrics.csv` |
| Log TransUNet | `../../HK7-CV-Class/brats-transunet/runs/tu2d_200/log.csv` |
| Test TransUNet | `../../HK7-CV-Class/brats-transunet/runs/tu2d_200/eval_test/metrics.csv` |
| Split dữ liệu | `../../HK7-CV-Class/brats-transunet/data/splits_200.json` |
| Cấu hình chung | `../../HK7-CV-Class/brats-transunet/configs/base.yaml` |
| U-Net | `../../HK7-CV-Class/brats-transunet/src/brats/models/unet.py` |
| TransUNet | `../../HK7-CV-Class/brats-transunet/src/brats/models/transunet.py` |
| Loss | `../../HK7-CV-Class/brats-transunet/src/brats/losses.py` |

## Phụ lục B — Quy ước trích dẫn kết quả

- Số có nhãn **validation** lấy từ `log.csv` và chỉ dựa trên 10 ca được đánh giá
  định kỳ trong quá trình train.
- Số có nhãn **test** là trung bình của 30 dòng trong `metrics.csv`.
- Số của ca `00495` chỉ dùng để kiểm thử end-to-end và minh họa demo.
- Không gán bất kỳ kết quả U-Net/TransUNet nào cho DAU-Mamba hoặc CGR-Mamba.
- Khi có thêm thí nghiệm, cần ghi rõ commit, config, checkpoint, split, seed và
  môi trường phần cứng để bảo đảm khả năng tái lập.

---

> **Lưu ý đạo đức và sử dụng:** Sản phẩm hiện tại phục vụ nghiên cứu và giảng
> dạy. Dự đoán của mô hình không được dùng như chẩn đoán y khoa độc lập hoặc thay
> thế đánh giá của bác sĩ chuyên khoa.
