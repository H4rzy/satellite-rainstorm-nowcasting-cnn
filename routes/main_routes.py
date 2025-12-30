import os
from flask import Blueprint, request, render_template, send_from_directory, current_app
import logging

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)


@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@main_bp.route('/', methods=['GET', 'POST'])
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
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            try:
                prediction_service = current_app.prediction_service

                result = prediction_service.process_prediction(file_path)

                label_final = result['label_text']
                img_data = result['img_data']
                risk_color = result['risk_color']
                risk_level = result['risk_level']

                logger.info(f"Prediction successful: {result['pred_label']} with {result['confidence']:.2%} confidence")

            except Exception as e:
                label_final = f"Lỗi xử lý ảnh: {e}"
                logger.error(f"Error processing prediction: {e}")
                risk_color = "secondary"
                risk_level = "Lỗi kỹ thuật"

    return render_template("index.html",
                         label=label_final,
                         risk_color=risk_color,
                         risk_level=risk_level,
                         img_data=img_data)


@main_bp.route('/rainfall', methods=['GET', 'POST'])
def rainfall_data():
    data_service = current_app.data_service

    years = data_service.get_available_years()
    selected_year = None
    year_data = None

    if request.method == "POST":
        selected_year = int(request.form.get("year"))
        year_data = data_service.get_data_by_year(selected_year)
        logger.info(f"Retrieved rainfall data for year {selected_year}")

    return render_template("rainfall.html",
                         years=years,
                         selected_year=selected_year,
                         year_data=year_data)
