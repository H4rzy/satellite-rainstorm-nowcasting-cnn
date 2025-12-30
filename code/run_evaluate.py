import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend để save file

from dataset import SatelliteDataset
from imagetransform import ImageTransform

def evaluate_and_save(model, dataloader, device, class_names, save_path):
    """Evaluate model và lưu confusion matrix ra file PNG"""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Đang đánh giá model...")
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tạo confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Vẽ Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Confusion Matrix - Rain Prediction Model", fontsize=14, fontweight='bold')
    plt.xlabel("Dự đoán (Predicted)", fontsize=12)
    plt.ylabel("Thực tế (Actual)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu Confusion Matrix: {save_path}")
    plt.close()
    
    return cm, all_labels, all_preds

def calculate_metrics(cm, class_names):
    """Tính và in các metrics"""
    num_classes = len(class_names)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp + 1e-8)
        recall[i] = tp / (tp + fn + 1e-8)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)

    # Tổng accuracy
    accuracy = np.trace(cm) / cm.sum() * 100
    
    print("\n" + "="*60)
    print("ĐÁNH GIÁ MODEL - RAIN PREDICTION")
    print("="*60)
    print(f"\n📊 TỔNG QUAN:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Tổng mẫu test: {cm.sum()}")
    
    print(f"\n📈 CHI TIẾT TỪNG LỚP:")
    print("-"*60)
    for name, p, r, f in zip(class_names, precision, recall, f1):
        print(f"   {name:12s} | Precision: {p*100:6.2f}% | Recall: {r*100:6.2f}% | F1: {f*100:6.2f}%")
    
    print("-"*60)
    print(f"   Macro Avg     | Precision: {precision.mean()*100:6.2f}% | Recall: {recall.mean()*100:6.2f}% | F1: {f1.mean()*100:6.2f}%")
    print("="*60)
    
    return accuracy, precision, recall, f1

def plot_metrics_bar(class_names, precision, recall, f1, save_path):
    """Vẽ biểu đồ bar cho metrics"""
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x, recall * 100, width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x + width, f1 * 100, width, label='F1-score', color='#FF9800')

    ax.set_xlabel('Lớp dự đoán', fontsize=12)
    ax.set_ylabel('Phần trăm (%)', fontsize=12)
    ax.set_title('Precision / Recall / F1-score theo từng lớp', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 100)

    # Thêm giá trị trên bar
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ Metrics: {save_path}")
    plt.close()

def main():
    # Cấu hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    class_names = ['not_rain', 'medium_rain', 'heavy_rain']
    
    # Load model
    print("📦 Đang load model...")
    model = models.resnet34(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(num_features, 3)
    )
    
    # Thử load model từ các path có thể
    model_paths = [
        "model/model_076.pth",
        "model/model_082.pth",
        "model/satellite_model.pth"
    ]
    
    model_loaded = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"   Loading from: {path}")
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model_loaded = True
            break
    
    if not model_loaded:
        print("❌ Không tìm thấy file model! Vui lòng kiểm tra lại.")
        print("   Các đường dẫn đã thử:", model_paths)
        return
    
    model = model.to(device)
    model.eval()
    print("✅ Model loaded thành công!")
    
    # Load dataset
    print("\n📂 Đang load dataset...")
    test_dir = 'Data/val'  # Dùng validation set để test
    if not os.path.exists(test_dir):
        test_dir = 'Data/train'
        print(f"   Dùng {test_dir} để đánh giá")
    
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(resize, mean, std)
    
    test_dataset = SatelliteDataset(test_dir, transform=transform.data_transform['val'])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"✅ Dataset loaded: {len(test_dataset)} mẫu")
    
    # Evaluate và lưu confusion matrix
    cm, labels, preds = evaluate_and_save(
        model, test_loader, device, class_names, 
        save_path="confusion_matrix_result.png"
    )
    
    # Tính metrics
    accuracy, precision, recall, f1 = calculate_metrics(cm, class_names)
    
    # Vẽ biểu đồ metrics
    plot_metrics_bar(class_names, precision, recall, f1, 
                     save_path="metrics_bar_chart.png")
    
    print("\n" + "="*60)
    print("🎉 HOÀN TẤT! Các file đã tạo:")
    print("   - confusion_matrix_result.png")
    print("   - metrics_bar_chart.png")
    print("="*60)

if __name__ == "__main__":
    main()
