import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from train_fashion_mnist import CNN, CLASS_NAMES  # Import class CNN từ file code chính của bạn

# 1. Khởi tạo môi trường và model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("outputs/checkpoints/cnn.pt", map_location=device))
model.eval()

# 2. Tải tập Test
MEAN, STD = (0.2860,), (0.3530,)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

# 3. Tìm các ảnh đoán sai
misclassified_images = []
misclassified_trues = []
misclassified_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        
        # Lọc ra các vị trí dự đoán sai
        wrong_indices = (preds != labels).nonzero(as_tuple=True)[0]
        for idx in wrong_indices:
            misclassified_images.append(images[idx].cpu().squeeze().numpy())
            misclassified_trues.append(labels[idx].item())
            misclassified_preds.append(preds[idx].item())
            if len(misclassified_images) >= 6: # Chỉ lấy 6 ảnh làm ví dụ
                break
        if len(misclassified_images) >= 6:
            break

# 4. Vẽ lưới 6 ảnh đoán sai
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for i, ax in enumerate(axes.flatten()):
    # Khôi phục lại ảnh để hiển thị (Denormalize)
    img = misclassified_images[i] * STD[0] + MEAN[0]
    ax.imshow(img, cmap='gray')
    true_label = CLASS_NAMES[misclassified_trues[i]]
    pred_label = CLASS_NAMES[misclassified_preds[i]]
    ax.set_title(f"Thực tế: {true_label}\nDự đoán: {pred_label}", color="red", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig("misclassified_examples.png", dpi=300)
print("Đã lưu ảnh misclassified_examples.png")