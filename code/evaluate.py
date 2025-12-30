from lib import *
from dataset import *
from imagetransform import *
from train_model import *

def evaluate_confusion_matrix(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tạo confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Vẽ heatmap
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

    return cm

def per_class_accuracy(cm, class_names):
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    for name, acc in zip(class_names, acc_per_class):
        print(f"{name:10s}: {acc*100:.2f}%")
        
def per_class_f1(cm, class_names):
    """
    Tính Precision, Recall, F1-score cho từng lớp và vẽ biểu đồ.
    """
    num_classes = len(class_names)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / (tp + fp + 1e-8)
        recall[i]     = tp / (tp + fn + 1e-8)
        f1[i]         = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8)

    # ---- In số liệu từng lớp ----
    for name, p, r, f in zip(class_names, precision, recall, f1):
        print(f"{name:10s} | Precision: {p*100:.2f}% | Recall: {r*100:.2f}% | F1-score: {f*100:.2f}%")

    # ---- Macro avg ----
    print(f"\nMacro Avg | Precision: {precision.mean()*100:.2f}% "
          f"| Recall: {recall.mean()*100:.2f}% "
          f"| F1-score: {f1.mean()*100:.2f}%")

    # ---- Micro avg ----
    total_tp = np.trace(cm)
    total_fp = cm.sum(axis=0) - np.diag(cm)
    total_fn = cm.sum(axis=1) - np.diag(cm)

    micro_precision = total_tp / (total_tp + total_fp.sum())
    micro_recall    = total_tp / (total_tp + total_fn.sum())
    micro_f1        = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

    print(f"Micro Avg | Precision: {micro_precision*100:.2f}% "
          f"| Recall: {micro_recall*100:.2f}% "
          f"| F1-score: {micro_f1*100:.2f}%")

    # ======================================================
    #                    VẼ BIỂU ĐỒ BAR
    # ======================================================
    x = np.arange(num_classes)
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision * 100, width, label='Precision')
    plt.bar(x,         recall * 100,    width, label='Recall')
    plt.bar(x + width, f1 * 100,        width, label='F1-score')

    plt.xticks(x, class_names, fontsize=12)
    plt.ylabel("Percentage (%)")
    plt.title("Per-Class Precision / Recall / F1-score")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return precision, recall, f1, micro_f1, f1.mean()



#Test
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

# 2. Freeze toàn bộ params nếu muốn fine-tune nhẹ
for param in model.parameters():
    param.requires_grad = False

# 3. Chỉnh fc mới + Dropout
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.6),
    nn.Linear(num_features, 3)  # 3 lớp
)

checkpoint_path = "model_076.pth"
state_dict = torch.load(checkpoint_path, map_location='cuda')  # hoặc map_location='cpu'
model.load_state_dict(state_dict)

# 3. Chuyển model sang device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # quan trọng để đánh giá

test_dir = 'Data/train'
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = ImageTransform(resize, mean, std)
class_names = ['not_rain', 'medium_rain', 'heavy_rain']
test_dataset = SatelliteDataset(test_dir, transform=transform.data_transform['test'])

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
cm = evaluate_confusion_matrix(model, test_loader, device, class_names)
# per_class_accuracy(cm, class_names)
per_class_f1(cm, class_names)