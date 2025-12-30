from lib import *
from imagetransform import ImageTransform
# Cấu hình GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset class đọc file .tif
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile as tiff

class SatelliteDataset(Dataset):
    """
    Dataset Sentinel-2 với 3 kênh: B11, B8, B4
    Labels: 0=not_rain, 1=medium_rain, 2=heavy_rain
    """
    def __init__(self, base_dir, transform=None):
        self.files = []
        self.labels = []
        self.transform = transform

        # Duyệt qua 3 class: not_rain, medium_rain, heavy_rain
        for label, folder in enumerate(['not_rain', 'medium_rain', 'heavy_rain']):
            folder_path = os.path.join(base_dir, folder)
            for file in os.listdir(folder_path):
                if file.lower().endswith('.tif'):
                    self.files.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        try:
            img = tiff.imread(path).astype(np.float32)

            if img.ndim == 2:  # ảnh xám
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3:  # ảnh 3 kênh trở lên
                img = img[..., :3]  # lấy 3 kênh đầu (hoặc giữ nguyên thứ tự B11,B8,B4)
            else:  # nếu lỗi kích thước hoặc chiều khác
                img = np.zeros((224, 224, 3), dtype=np.float32)

        except Exception as e:
            print(f"Lỗi đọc ảnh {path}: {e}")
            img = np.zeros((224, 224, 3), dtype=np.float32)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale dữ liệu MODIS từ 0-255 về 0-1
        img = np.clip(img / 255.0, 0, 1)

        # Đưa về kiểu uint8 (0–255) để dùng được với PIL
        img = (img * 255).astype(np.uint8)

        # Chuyển sang ảnh PIL
        img = Image.fromarray(img)

        # Áp dụng transform (nếu có)
        if self.transform:
            img = self.transform(img)

        return img, label
