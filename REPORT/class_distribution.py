import matplotlib.pyplot as plt
import seaborn as sns

# Dữ liệu phân bố của Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
counts = [6000] * 10  # Mỗi lớp có 6000 ảnh

plt.figure(figsize=(10, 6))
sns.barplot(x=counts, y=class_names, palette="viridis")

plt.title("Phân bố số lượng ảnh theo từng lớp (Tập Huấn luyện)", fontsize=14)
plt.xlabel("Số lượng ảnh", fontsize=12)
plt.ylabel("Lớp trang phục", fontsize=12)

# Thêm số liệu trực tiếp lên cột
for i, v in enumerate(counts):
    plt.text(v - 400, i + 0.1, str(v), color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
print("Đã lưu ảnh class_distribution.png")