from lib import *
from dataset import *
from imagetransform import *
from train_model import *
from loss import FocalLoss
# from evaluate import evaluate_confusion_matrix, per_class_accuracy
torch.manual_seed(1234)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    # ==== 1️Cấu hình dữ liệu ====
    train_dir = 'Data/train'
    val_dir = 'Data/val'

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(resize, mean, std)

    train_dataset = SatelliteDataset(train_dir, transform=transform.data_transform['train'])
    val_dataset = SatelliteDataset(val_dir, transform=transform.data_transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    dataloader_dict = {'train': train_loader, 'val': val_loader}

    # ==== 2️. Chuẩn bị mô hình ====
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
    
    print(model)
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model = model.to(device)
    
    labels_train = train_dataset.labels  

    classes = np.unique(labels_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels_train)
    # class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    #Sử dụng focal loss
    BOOST_FACTOR = 1.5

    class_weights_boosted = class_weights.copy()
    class_weights_boosted[2] *= BOOST_FACTOR # Tăng trọng số cho lớp 2 (heavy rain)
    weights = torch.tensor(class_weights_boosted, dtype=torch.float).to(device)
    
    print("Trọng số CƠ SỞ (L0, L1, L2):", class_weights)
    print(f"Trọng số TĂNG CƯỜNG (L0, L1, L2): {class_weights_boosted}")
    print(f"Lớp 2 (Heavy Rain) trọng số mới: {weights[2].item():.4f}")
    # # ==== Cấu hình huấn luyện ====
    criterion = FocalLoss(alpha=weights, gamma=2.5)
    #Thử nghiệm với cross entropy loss
    # criterion = nn.CrossEntropyLoss()
    
    params1, params2, params3 = update_params(model)
    # print(len(params1), len(params2), len(params3))
    #Điều chỉnh lr theo từng lớp
    param_groups = []
    if params1:  # features đầu
        param_groups.append({'params': params1, 'lr': 5e-5})
    if params2:  # features cuối
        param_groups.append({'params': params2, 'lr': 2e-4})
    if params3:  # fc cuối
        param_groups.append({'params': params3, 'lr': 3e-4})
        
    optimizer = optim.Adam(param_groups, weight_decay=1e-4)

    # ==== Huấn luyện ====
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-7) 
    num_epochs = 40
    train_losses, val_losses, train_accs, val_accs = train_model(model, dataloader_dict, criterion, optimizer, num_epochs, scheduler)

    # ====  Lưu mô hình ====
    save_dir = r"D:\Documents\BaiTap\AI\DoAnAI\Satelitte"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), 'model/satellite_model.pth')

    # ==== Vẽ biểu đồ ====
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss per Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy per Epoch')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()