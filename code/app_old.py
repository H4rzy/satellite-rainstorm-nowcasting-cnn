from flask import Flask, request, render_template, send_from_directory
import io
import torch
from torchvision import transforms
from flask_cors import CORS
import os, base64
from predict import load_model, predict_image
import tifffile as tiff
from PIL import Image
from lib import *

device = 'cpu'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
CORS(app)

# Đọc CSV
df = pd.read_csv("rainfall_data.csv")

df['date'] = pd.to_datetime(df['date'], format='mixed')

df['year'] = df['date'].dt.year

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.6),
    nn.Linear(num_features, 3) 
)

checkpoint_path = "model/model_076.pth"
model = load_model(checkpoint_path, device='cpu')

class_names = ['not_rain', 'medium_rain', 'heavy_rain']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def predict():
    label_final = ""
    img_data = None
    risk_color = "light"
    risk_level = "Chưa có dự đoán" 
    
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == "":
            label_final = "Vui lòng chọn file ảnh."
        else:
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

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

                # Convert Pillow image sang Base64 để hiển thị
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_data = base64.b64encode(buffered.getvalue()).decode()
                
                # CHẠY DỰ ĐOÁN
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_class_idx = torch.argmax(probs, dim=1).item()
                
                pred_label = class_names[pred_class_idx]
                pred_prob = probs.squeeze().cpu().numpy()[pred_class_idx]
                
                # === ÁP DỤNG LOGIC RỦI RO CUỐI CÙNG ===
                if pred_class_idx == 0:
                    risk_level = "Rất Thấp (An toàn)"
                    risk_color = "success"
                elif pred_class_idx == 1:
                    risk_level = "Trung Bình (Cần chú ý)"
                    risk_color = "warning"
                else:
                    risk_level = "CAO (Cần cảnh báo)"
                    risk_color = "danger"

                label_final = f"Dự đoán: {pred_label} | Nguy cơ: {risk_level} (Độ tự tin: {pred_prob*100:.2f}%)"

            except Exception as e:
                label_final = f"Lỗi xử lý ảnh: {e}"
                print("Lỗi xử lý ảnh:", e)
                # Đảm bảo các biến rủi ro được set cho thông báo lỗi
                risk_color = "secondary" 
                risk_level = "Lỗi kỹ thuật"


    return render_template("index.html", 
                           label=label_final, 
                           risk_color=risk_color, 
                           risk_level=risk_level, 
                           img_data=img_data) 
#Trang xem dữ liệu mưa
@app.route('/rainfall', methods=['GET', 'POST'])
def getRain():
    years = sorted(df['year'].unique())
    selected_year = None
    year_data = None

    if request.method == "POST":
        selected_year = int(request.form.get("year"))
        year_data = df[df['year'] == selected_year].sort_values('date')

    return render_template("rainfall.html", years=years, selected_year=selected_year, year_data=year_data)

app.run()

    