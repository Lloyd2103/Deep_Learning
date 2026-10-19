# Image Deshadowing using Restormer

Thư mục và mã nguồn chuẩn mực cho tác vụ **Image / Document Deshadowing (Loại bỏ bóng trên ảnh và tài liệu)** bằng kiến trúc Restormer kết hợp **Nhánh Tần số cao (High-Frequency Branch)** và **Loss Tần số (Frequency Loss)** nhằm bảo toàn nét chữ sắc cạnh.

Hỗ trợ sẵn cấu hình tối ưu chạy mượt mà trên **Google Colab GPU Tesla T4 (15GB VRAM)** và máy trạm **RTX 5090**.

---

## 1. Cấu trúc thư mục (Directory Structure)

```
Deshadowing/
├── Datasets/
│   ├── README.md                      # Hướng dẫn cấu trúc dataset
│   ├── train/ISTD/
│   │   ├── input/                     # Chứa ảnh có bóng (Shadowed images)
│   │   └── target/                    # Chứa ảnh sạch bóng (Ground-truth shadow-free)
│   └── test/ISTD/
│       ├── input/                     # Chứa ảnh test có bóng
│       ├── target/                    # Chứa ảnh test sạch bóng
│       └── mask/                      # (Tùy chọn) Mặt nạ bóng nhị phân để tính RMSE Shadow/Non-Shadow
├── Options/
│   ├── Deshadowing_Restormer_Colab_T4.yml  # 🌟 Cấu hình tối ưu cho Google Colab T4 (15GB)
│   └── Deshadowing_Restormer.yml           # Cấu hình cho máy trạm (RTX 5090 / Multi-GPU)
├── pretrained_models/                 # Nơi lưu checkpoint (.pth)
├── results/                           # Thư mục chứa kết quả sau khi chạy test.py
├── utils.py                           # Các hàm phụ trợ (load/save ảnh, PSNR, SSIM)
├── test.py                            # Script suy luận (Inference/Testing)
├── evaluate_deshadowing.py            # Script đánh giá PSNR, SSIM, RMSE (All/Shadow/Non-Shadow)
├── train.sh                           # Bash script tiện ích chạy huấn luyện trên server
├── colab_run.py                       # 🌟 Script 1-click kiểm tra môi trường và chạy trên Google Colab
└── README.md
```

---

## 2. Các cải tiến bảo toàn nét chữ (High-Frequency Preservation)

1. **High-Frequency Branch (`use_hf_branch: true`)**:
   - Nhánh CNN siêu nhẹ (~3,000 tham số, không tốn thêm VRAM) nằm tại `basicsr/models/archs/high_freq_branch.py`.
   - Sử dụng bộ lọc High-pass trích xuất cạnh viền/nét chữ, loại trừ ảnh hưởng của mảng bóng tối tần số thấp.
2. **Frequency Domain Loss (`freq_opt`)**:
   - Tính toán sai khác phổ biên độ thông qua 2D Real FFT (`torch.fft.rfft2`, chuẩn `norm='ortho'`), giúp khôi phục các chi tiết gai/nét chữ mà loss L1 thông thường hay làm mờ mịn.

---

## 3. Huấn luyện trên Google Colab (GPU Tesla T4 - 15GB VRAM)

Dành cho các thành viên trong nhóm khi chạy trên Google Colab:

### Bước 1: Chuẩn bị môi trường trên Notebook Colab
```bash
# 1. Cài đặt các thư viện cần thiết
!pip install -q lmdb addict yapf scikit-image tensorboard natsort einops

# 2. Đảm bảo runtime là T4 GPU (Runtime -> Change runtime type -> T4 GPU)
```

### Bước 2: Khởi động huấn luyện 1-Click
```bash
# Kiểm tra GPU và dataset trước:
!python Deshadowing/colab_run.py --check_only

# Khởi chạy huấn luyện với cấu hình an toàn cho T4:
!python Deshadowing/colab_run.py --config Deshadowing/Options/Deshadowing_Restormer_Colab_T4.yml
```
> **Đặc điểm cấu hình Colab T4**:
> - `num_worker_per_gpu: 2` (ngăn ngừa lỗi tràn 12.7GB CPU RAM của Colab).
> - Giới hạn kích thước patch tối đa `256x256`, batch size điều chỉnh từ 4 xuống 1.
> - VRAM tiêu thụ duy trì an toàn ở mức **6GB - 11GB**, không bao giờ bị OOM.

---

## 4. Huấn luyện trên máy trạm cục bộ (RTX 5090)

```bash
# Huấn luyện 1 GPU (RTX 5090):
./Deshadowing/train.sh Deshadowing/Options/Deshadowing_Restormer.yml 1

# Hoặc chạy trực tiếp qua basicsr:
python basicsr/train.py -opt Deshadowing/Options/Deshadowing_Restormer.yml
```

### Theo dõi tiến trình với TensorBoard
```bash
tensorboard --logdir ./tb_logger --port 6006
```

---

## 5. Kiểm thử & Đánh giá (Inference & Evaluation)

### A. Chạy suy luận (Testing)
Đặt checkpoint vào `Deshadowing/pretrained_models/deshadowing.pth`:
```bash
python Deshadowing/test.py \
    --input_dir ./Deshadowing/Datasets/ \
    --result_dir ./Deshadowing/results/ \
    --weights ./Deshadowing/pretrained_models/deshadowing.pth \
    --dataset ISTD
```

### B. Tính toán PSNR, SSIM và RMSE
```bash
python Deshadowing/evaluate_deshadowing.py \
    --res_dir ./Deshadowing/results/ISTD \
    --gt_dir ./Deshadowing/Datasets/test/ISTD/target \
    --mask_dir ./Deshadowing/Datasets/test/ISTD/mask
```
Xuất ra kết quả:
- **Average PSNR (dB)**
- **Average SSIM**
- **RMSE (All)**
- **RMSE (Shadow & Non-Shadow)** (nếu có mask)
