# KẾT QUẢ VÀ ĐÁNH GIÁ

## Môi trường thử nghiệm

| Thành phần | Chi tiết |
|------------|----------|
| **Phần cứng** | GPU Nvidia có hỗ trợ CUDA nhằm huấn luyện |
| **Ngôn ngữ & Framework** | Python, PyTorch |
| **Thư viện xử lý dữ liệu** | pandas, tifffile, NumPy, Pillow |

---

## Kịch bản thử nghiệm

Nhóm đã huấn luyện hai mô hình trên cùng một tập dữ liệu để chứng minh việc sử dụng hàm Focal Loss hiệu quả hơn hàm thông thường.

### Bộ dữ liệu

Tập ảnh từ vệ tinh Sentinel-2 đã lọc bỏ ảnh kém chất lượng, được chia thành 2 lớp:

| Tập dữ liệu | Tỷ lệ | Số lượng ảnh |
|-------------|-------|--------------|
| Tập huấn luyện (Train) | 70% | 1,470 |
| Tập kiểm định (Validation) | 15% | 315 |
| Tập kiểm thử (Test) | 15% | 316 |
| **Tổng cộng** | **100%** | **2,101** |

**Phân bố nhãn:**
| Nhãn | Tỷ lệ | Số lượng ảnh |
|------|-------|--------------|
| Low Risk (An toàn) | 64.2% | 1,349 |
| High Risk (Nguy cơ) | 35.8% | 752 |

---

## Kịch bản 1: Mô hình độ chính xác 81.96%

**Mục tiêu:** Cho thấy việc không dùng kỹ thuật xử lý mất cân bằng nhãn sẽ khiến mô hình không tập trung vào lớp hiếm (High Risk)

**Hàm loss:** Cross-Entropy Loss thông thường

### Confusion Matrix - Kịch bản 1

|  | Dự đoán: Low Risk | Dự đoán: High Risk |
|--|-------------------|-------------------|
| **Thực tế: Low Risk** | 185 | 17 |
| **Thực tế: High Risk** | 40 | 74 |

**Hình 5.1:** Confusion Matrix của mô hình có độ chính xác 81.96%

### Các chỉ số đánh giá - Kịch bản 1

| Chỉ số | Low Risk | High Risk |
|--------|----------|-----------|
| Precision | 82.22% | 81.32% |
| Recall | 91.58% | 64.91% |
| F1-Score | 86.65% | 72.20% |

**Hình 5.2:** Biểu đồ so sánh các chỉ số Precision, Recall và F1-score

### Nhận xét Kịch bản 1:

Tuy mô hình có chỉ số **Accuracy cao (81.96%)** và **Precision cao (81.32%)** khi dự đoán lớp High Risk. Nhưng **Recall ở lớp High Risk chỉ đạt 64.91%**. 

Điều này có nghĩa là mô hình đã **bỏ sót 40 trên 114 ảnh High Risk** thực tế, một tỉ lệ bỏ sót lên tới **35.1%**.

Nếu áp dụng mô hình này vào thực tế, khả năng bỏ sót hơn 35% các trường hợp nguy cơ cao sẽ gây ra hậu quả nghiêm trọng và không thể chấp nhận được trong một ứng dụng cảnh báo thời tiết.

---

## Kịch bản 2: Mô hình độ chính xác 81.65%

**Mục tiêu:** Cho thấy mô hình dự đoán đúng và ít bỏ sót lớp hiếm (High Risk)

**Hàm loss:** Focal Loss với alpha = trọng số của lớp (lớp hiếm được tăng trọng số) và gamma = 2.5

### Confusion Matrix - Kịch bản 2

|  | Dự đoán: Low Risk | Dự đoán: High Risk |
|--|-------------------|-------------------|
| **Thực tế: Low Risk** | 167 | 35 |
| **Thực tế: High Risk** | 23 | 91 |

**Hình 5.3:** Confusion Matrix của mô hình sử dụng Focal Loss

### Các chỉ số đánh giá - Kịch bản 2

| Chỉ số | Low Risk | High Risk |
|--------|----------|-----------|
| Precision | 87.89% | 72.22% |
| Recall | 82.67% | 79.82% |
| F1-Score | 85.20% | 75.83% |

**Hình 5.4:** Biểu đồ so sánh các chỉ số Precision, Recall và F1-score

### Nhận xét Kịch bản 2:

Mô hình sử dụng Focal Loss đã thực hiện thành công sự đánh đổi chiến lược: 

- **Recall cho lớp High Risk tăng từ 64.91% lên 79.82%** (tăng 14.91%)
- Số ảnh High Risk bỏ sót **giảm từ 40 xuống còn 23** (giảm 42.5%)
- Tỉ lệ bỏ sót giảm từ **35.1% xuống còn 20.2%**

Điều này chứng minh rằng việc sử dụng hàm Focal Loss đã thành công trong việc buộc mô hình **ưu tiên an toàn, giảm thiểu lỗi bỏ sót**.

---

## Bảng so sánh 2 kịch bản

| Chỉ số | Cross-Entropy | Focal Loss | Thay đổi |
|--------|---------------|------------|----------|
| **Accuracy** | 81.96% | 81.65% | -0.31% |
| **Recall (High Risk)** | 64.91% | 79.82% | **+14.91%** |
| **Số ảnh bỏ sót** | 40/114 | 23/114 | **-17 ảnh** |
| **Tỉ lệ bỏ sót** | 35.1% | 20.2% | **-14.9%** |

---

## Kết luận

Việc hy sinh một phần nhỏ Accuracy (0.31%) là cái giá cần thiết và hoàn toàn hợp lý để:
- **Tăng 14.91% khả năng phát hiện High Risk**
- **Giảm 42.5% số trường hợp bỏ sót**

Focal Loss là lựa chọn phù hợp cho các ứng dụng cảnh báo thời tiết nơi việc bỏ sót một trường hợp nguy hiểm có thể gây ra hậu quả nghiêm trọng.
