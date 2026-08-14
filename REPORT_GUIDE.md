# Hướng dẫn nâng cấp báo cáo (từ kết quả cơ bản → báo cáo có chiều sâu)

Notebook `notebooks/Fashion_MNIST_CNN.ipynb` hiện đã cho ra: train MLP, train CNN, so sánh Accuracy/Precision/Recall/F1 + Confusion Matrix. Đây là **khung sườn**. Để báo cáo có chiều sâu — đúng tinh thần đề tài "áp dụng kỹ thuật cải thiện hiệu suất... hạn chế overfitting" — cần **chứng minh** từng kỹ thuật thực sự có tác dụng, không chỉ liệt kê là có dùng.

Checklist bên dưới xếp theo độ ưu tiên. Làm tới đâu, tick tới đó.

---

## ☐ 1. Ablation study — chứng minh Augmentation/Dropout/BatchNorm có tác dụng (ưu tiên cao nhất)

**Mục đích:** Trả lời câu hỏi "nếu bỏ kỹ thuật này đi thì sao?" — đây là bằng chứng thực nghiệm cho phần "hạn chế overfitting" trong đề tài, thay vì chỉ nói suông là có dùng.

**Cách làm:** Thêm hàm `CNN` có thể bật/tắt từng phần (xem `src/models.py`, class `CNN` hiện dropout đang cố định — sửa lại để nhận `use_batchnorm`, hoặc đơn giản hơn là train 4 phiên bản trong notebook):

1. CNN đầy đủ (augment + dropout + batchnorm) — đã có sẵn, là mô hình `cnn_model` trong notebook
2. CNN **không** Data Augmentation — trong bước tạo `train_loader`, dùng `eval_transform` thay vì `train_transform` cho `train_set`
3. CNN **không** Dropout — tạo class CNN mới với `dropout=0.0`
4. CNN **không** BatchNorm — cần sửa `CNN` bỏ các dòng `nn.BatchNorm2d`/`nn.BatchNorm1d`

Với mỗi phiên bản, ghi lại 2 con số:
- **Test Accuracy**
- **Overfitting gap** = `train_acc` (epoch cuối) − `val_acc` (epoch cuối). Gap càng lớn → càng overfit.

**Output cần có trong báo cáo:** 1 bảng dạng:

| Cấu hình | Test Accuracy | Train-Val Gap (overfitting) |
|---|---|---|
| CNN đầy đủ | ... | ... |
| CNN không Augmentation | ... | ... (thường gap lớn hơn) |
| CNN không Dropout | ... | ... (thường gap lớn hơn) |
| CNN không BatchNorm | ... | ... (thường học chậm/kém ổn định hơn) |

Kèm nhận xét: kỹ thuật nào ảnh hưởng nhiều nhất, có đúng như lý thuyết dự đoán không.

---

## ☐ 2. Phân tích lỗi sai (Error Analysis)

**Mục đích:** Cho thấy bạn hiểu *tại sao* mô hình sai, không chỉ đo được sai bao nhiêu — giám khảo đánh giá cao phần này.

**Cách làm:** Sau khi có `cnn_pred` và `cnn_true` từ bước đánh giá:

```python
import numpy as np

wrong_idx = np.where(cnn_pred != cnn_true)[0]
print(f"Số lượng ảnh đoán sai: {len(wrong_idx)} / {len(cnn_true)}")

# Hiển thị 12 ảnh sai ngẫu nhiên kèm nhãn thật vs nhãn đoán
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
sample_idx = np.random.choice(wrong_idx, 12, replace=False)
for ax, idx in zip(axes.flat, sample_idx):
    img, _ = test_set[idx]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"Thật: {CLASS_NAMES[cnn_true[idx]]}\nĐoán: {CLASS_NAMES[cnn_pred[idx]]}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
plt.show()
```

**Output cần có trong báo cáo:** Ảnh minh họa các trường hợp sai + nhận xét, ví dụ: "CNN hay nhầm Shirt với T-shirt/top và Coat — nhìn vào confusion matrix (mục đã có sẵn) thấy đây là cặp có số nhầm lẫn cao nhất, hợp lý vì các lớp này có hình dáng tương đồng khi ảnh nhỏ và đen trắng."

---

## ☐ 3. So sánh độ phức tạp / chi phí tính toán

**Mục đích:** CNN chính xác hơn nhưng "đắt" hơn — đây là trade-off nên đưa vào phần kết luận.

**Cách làm:** Đo thời gian train và số tham số:

```python
import time

start = time.time()
# train 1 epoch để đo (hoặc lấy trung bình thời gian mỗi epoch trong lúc fit() ở trên)
elapsed = time.time() - start

mlp_params = sum(p.numel() for p in mlp_model.parameters())
cnn_params = sum(p.numel() for p in cnn_model.parameters())
```

**Output cần có trong báo cáo:** Bảng: Model | Số tham số | Thời gian train/epoch | Test Accuracy — để thấy CNN đánh đổi chi phí lấy độ chính xác như thế nào.

---

## ☐ 4. (Tùy chọn, nếu còn thời gian) Trực quan hóa CNN "nhìn thấy" gì

**Mục đích:** Giải thích trực quan *tại sao* CNN tốt hơn MLP, không chỉ bằng con số.

**Cách làm:** Lấy 1 ảnh test, cho qua lớp Conv đầu tiên của `cnn_model.features[0]`, vẽ các feature map ra:

```python
sample_img, _ = test_set[0]
sample_img = sample_img.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    feature_maps = cnn_model.features[0](sample_img)  # sau Conv2d đầu tiên

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[0, i].cpu(), cmap="viridis")
    ax.axis("off")
fig.suptitle("Feature maps sau lớp Conv đầu tiên")
plt.show()
```

**Output cần có trong báo cáo:** Ảnh feature map + nhận xét: CNN học được các bộ lọc phát hiện cạnh/viền/hoa văn, đây là lý do nó vượt trội hơn MLP (MLP không có khái niệm "không gian" của ảnh).

---

## Cấu trúc báo cáo gợi ý (tổng hợp)

1. **Giới thiệu bài toán** — Fashion-MNIST là gì, mục tiêu đề tài
2. **Dữ liệu** — số lượng ảnh, 10 lớp, ảnh mẫu (đã có trong notebook mục 2)
3. **Phương pháp** — kiến trúc MLP/CNN (đã có trong `README.md`), giải thích ngắn gọn Dropout/BatchNorm/Augmentation là gì và vì sao dùng
4. **Thực nghiệm & Kết quả**
   - So sánh MLP vs CNN (đã có — notebook mục 7-8)
   - Ablation study (mục 1 ở trên)
   - Phân tích lỗi sai (mục 2 ở trên)
   - So sánh chi phí (mục 3 ở trên)
   - (Tùy chọn) Feature map (mục 4 ở trên)
5. **Thảo luận** — kỹ thuật nào hiệu quả nhất, hạn chế của mô hình, hướng cải thiện (vd: thử ResNet nhỏ, thử thêm augmentation khác)
6. **Kết luận**

---

## Ghi chú

- Tất cả code mẫu ở trên viết để chèn thêm vào **cuối** `notebooks/Fashion_MNIST_CNN.ipynb` (sau mục 8 "So sánh CNN vs MLP"), tái sử dụng các biến đã có sẵn (`cnn_model`, `test_set`, `CLASS_NAMES`, `DEVICE`, `plt`...).
- Nếu muốn mình code sẵn các phần này vào notebook thay vì tự làm, cứ nói — hiện tại để bạn tự follow theo checklist này trước.
