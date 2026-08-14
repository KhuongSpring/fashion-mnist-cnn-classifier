# Fashion MNIST CNN Classifier

Phân loại ảnh thời trang trên bộ dữ liệu **Fashion-MNIST** bằng **CNN**, so sánh với **MLP** truyền thống. Áp dụng **Data Augmentation**, **Dropout**, **Batch Normalization** để tăng khả năng tổng quát hóa và hạn chế overfitting. Đánh giá qua **Accuracy, Precision, Recall, F1-score, Confusion Matrix**.

## Cấu trúc thư mục

```
BTL_AI/
├── main.py                 # CLI: train + evaluate 1 model (mlp/cnn), lưu kết quả vào outputs/
├── requirements.txt
├── notebooks/
│   └── Fashion_MNIST_CNN.ipynb   # Notebook self-contained, chạy trực tiếp trên Google Colab
├── src/
│   ├── config.py            # đường dẫn, hằng số, hyperparameter mặc định
│   ├── data.py               # DataLoader + augmentation cho Fashion-MNIST
│   ├── models.py              # định nghĩa MLP và CNN
│   ├── engine.py               # vòng lặp train/eval/predict
│   └── utils.py                 # seed, vẽ biểu đồ, confusion matrix, classification report
├── data/                    # dữ liệu Fashion-MNIST tải tự động (gitignored)
└── outputs/
    ├── checkpoints/          # trọng số mô hình đã train (.pt)
    ├── figures/               # biểu đồ loss/accuracy, confusion matrix (.png)
    └── logs/                   # lịch sử training, classification report (.json)
```

## Cách chạy

### Cách 1: Google Colab (khuyến nghị cho việc chạy nhanh, có GPU miễn phí)

1. Mở [Google Colab](https://colab.research.google.com), chọn **File > Upload notebook**, tải lên `notebooks/Fashion_MNIST_CNN.ipynb`.
2. Vào **Runtime > Change runtime type**, chọn **GPU**.
3. Chạy tuần tự từ cell đầu tiên đến cuối (**Runtime > Run all**). Notebook tự cài thư viện và tự tải Fashion-MNIST, không cần clone repo.

### Cách 2: Local với Jupyter Notebook

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/Fashion_MNIST_CNN.ipynb
```

### Cách 3: Chạy bằng CLI (script, không cần notebook)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --model mlp --epochs 20
python main.py --model cnn --epochs 20
```

Kết quả (checkpoint, biểu đồ, log) được lưu vào `outputs/`.

## Kỹ thuật áp dụng

- **Data Augmentation**: random horizontal flip, random rotation (±10°), random translate (±10%) — chỉ áp dụng cho tập train.
- **Batch Normalization**: sau mỗi lớp Conv/Linear (trước activation).
- **Dropout**: sau các khối pooling và các lớp fully-connected.
- **Đánh giá**: Accuracy, Precision/Recall/F1-score (per-class + macro/weighted avg qua `sklearn.metrics.classification_report`), Confusion Matrix (heatmap).

## Nâng cấp báo cáo

Notebook hiện tại cho ra kết quả cơ bản (train + so sánh MLP/CNN). Để báo cáo có chiều sâu hơn (ablation study, phân tích lỗi sai, so sánh chi phí...), xem checklist chi tiết trong [`REPORT_GUIDE.md`](REPORT_GUIDE.md).

## Kiến trúc mô hình

- **MLP**: `Flatten → Linear(784,256) → BN → ReLU → Dropout → Linear(256,128) → BN → ReLU → Dropout → Linear(128,10)`
- **CNN**: 2 khối `[Conv-BN-ReLU]×2 → MaxPool → Dropout` (32 rồi 64 kênh) → `Flatten → Linear(3136,256) → BN → ReLU → Dropout → Linear(256,10)`
