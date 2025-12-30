from lib import *

class ImageTransform():
  def __init__(self, resize, mean, std):
      self.data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=(0.5, 1.0), ratio=(0.8, 1.2)),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            
            transforms.RandomAffine(
                degrees=45,         # Xoay thêm +/- 45 độ (kết hợp với 90 ở dưới)
                translate=(0.1, 0.1), # Dịch chuyển 10%
                shear=10            # Bóp nghiêng 10 độ
            ),

            transforms.RandomRotation(degrees=90),

            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.05
                )
            ], p=0.8),

            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.Normalize(mean, std)
        ]),

        'val': transforms.Compose([
          transforms.Resize(resize),
          transforms.CenterCrop(resize),
          transforms.ToTensor(),
          transforms.Normalize(mean, std)
      ]),
        
        'test': transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
      }
  def __call__(self, img, phase ='train'):
    return self.data_transform[phase](img)