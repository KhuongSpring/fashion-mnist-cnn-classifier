import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# 1. Tải dữ liệu gốc (chưa chuẩn hóa)
raw_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
raw_pixels = raw_dataset.data.numpy().flatten()

# 2. Tải dữ liệu đã chuẩn hóa
MEAN, STD = (0.2860,), (0.3530,)
norm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
norm_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=norm_transform)
# Trích xuất 1 batch ngẫu nhiên để vẽ cho nhanh
norm_pixels = next(iter(torch.utils.data.DataLoader(norm_dataset, batch_size=1000)))[0].numpy().flatten()

# 3. Vẽ biểu đồ so sánh
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(raw_pixels, bins=50, color='gray', alpha=0.7)
axes[0].set_title("Phân phối Pixel (Ảnh gốc 0-255)")
axes[0].set_xlabel("Giá trị Pixel")
axes[0].set_ylabel("Tần suất")

axes[1].hist(norm_pixels, bins=50, color='blue', alpha=0.7)
axes[1].set_title("Phân phối Pixel (Sau Normalization)")
axes[1].set_xlabel("Giá trị Pixel chuẩn hóa")
axes[1].set_ylabel("Tần suất")

plt.tight_layout()
plt.savefig("histogram_pixel.png", dpi=300)
print("Đã lưu ảnh histogram_pixel.png")