import torch
import numpy as np
from PIL import Image
import tifffile as tiff
from torchvision import transforms
from imagetransform import ImageTransform
from lib import *

def load_model(model_path, device='cuda'):
    # Load pretrained ResNet50
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    # Freeze toàn bộ params (nếu muốn fine-tune nhẹ)
    for param in model.parameters():
        param.requires_grad = False

    # Chỉnh fc mới (3 lớp) + dropout
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(num_features, 3)
    )

    # Load state_dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device='cuda'):
    # Transform chuẩn ResNet34
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load ảnh
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)  # thêm batch dimension

    # Dự đoán
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    class_names = ['not_rain', 'medium_rain', 'heavy_rain']
    return class_names[pred_class], probs.squeeze().cpu().numpy()


    
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_path = 'best_model.pth'  # state_dict, không save cả model
# image_path = 'Dataset/Dataset_split_ResNet50/train/medium_rain/Sentinel2_HCMC_20170329.tif'

# model = load_model(model_path, device)
# label, probs = predict_image(model, image_path, device)

# print("Dự đoán:", label)
# print("Xác suất các lớp:", probs)