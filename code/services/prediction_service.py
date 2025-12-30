import io
import base64
import numpy as np
import torch
import tifffile as tiff
from PIL import Image
from torchvision import transforms
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class PredictionService:

    def __init__(self, model, device='cpu', class_names=None):
        self.model = model
        self.device = device
        self.class_names = class_names or ['not_rain', 'medium_rain', 'heavy_rain']

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])

    def load_image(self, file_path: str) -> Image.Image:
        try:
            if file_path.lower().endswith((".tif", ".tiff")):
                img_array = tiff.imread(file_path).astype(np.uint8)
                img_array = np.nan_to_num(img_array, nan=0, posinf=0, neginf=0).astype(np.uint8)

                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[2] > 3:
                    img_array = img_array[..., :3]

                pil_img = Image.fromarray(img_array)
            else:
                pil_img = Image.open(file_path).convert("RGB")

            logger.info(f"Successfully loaded image from {file_path}")
            return pil_img

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise

    def image_to_base64(self, pil_img: Image.Image) -> str:
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def predict(self, pil_img: Image.Image) -> Tuple[int, np.ndarray]:
        try:
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class_idx = torch.argmax(probs, dim=1).item()

            probs_np = probs.squeeze().cpu().numpy()
            logger.info(f"Prediction: {self.class_names[pred_class_idx]} with confidence {probs_np[pred_class_idx]:.2%}")

            return pred_class_idx, probs_np

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def get_risk_assessment(self, pred_class_idx: int) -> Dict[str, str]:
        risk_mapping = {
            0: {"level": "Rất Thấp (An toàn)", "color": "success"},
            1: {"level": "Trung Bình (Cần chú ý)", "color": "warning"},
            2: {"level": "CAO (Cần cảnh báo)", "color": "danger"}
        }

        return {
            "risk_level": risk_mapping[pred_class_idx]["level"],
            "risk_color": risk_mapping[pred_class_idx]["color"]
        }

    def process_prediction(self, file_path: str) -> Dict:
        try:
            pil_img = self.load_image(file_path)

            img_data = self.image_to_base64(pil_img)

            pred_class_idx, probs = self.predict(pil_img)

            risk = self.get_risk_assessment(pred_class_idx)

            result = {
                "pred_label": self.class_names[pred_class_idx],
                "pred_class_idx": pred_class_idx,
                "confidence": float(probs[pred_class_idx]),
                "probabilities": {name: float(prob) for name, prob in zip(self.class_names, probs)},
                "risk_level": risk["risk_level"],
                "risk_color": risk["risk_color"],
                "img_data": img_data,
                "label_text": f"Dự đoán: {self.class_names[pred_class_idx]} | Nguy cơ: {risk['risk_level']} (Độ tự tin: {probs[pred_class_idx]*100:.2f}%)"
            }

            return result

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            raise
