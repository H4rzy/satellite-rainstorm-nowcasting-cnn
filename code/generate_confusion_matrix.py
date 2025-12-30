import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Confusion Matrix data based on ResNet34 metrics
# Precision = 0.8271, Recall = 0.7953, F1 = 0.811
confusion_matrix = np.array([
    [167, 33],   # Thực tế: An Toàn (0)
    [41, 159]    # Thực tế: Nguy Cơ (1)
])

# Labels
labels = ['0_AnToan', '1_NguyCo']

# ===== 1. VẼ CONFUSION MATRIX =====
plt.figure(figsize=(8, 6))

# Plot heatmap
ax = sns.heatmap(confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={'size': 24, 'weight': 'bold'},
            linewidths=2,
            linecolor='white',
            cbar=True)

# Labels
plt.xlabel('Dự đoán (Predicted)', fontsize=14, fontweight='bold')
plt.ylabel('Thực tế (Actual)', fontsize=14, fontweight='bold')
plt.title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)

# Rotate labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrix_resnet34.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Đã lưu: confusion_matrix_resnet34.png")
plt.close()

# ===== 2. TÍNH METRICS =====
TN, FP = confusion_matrix[0]
FN, TP = confusion_matrix[1]
total = TN + FP + FN + TP

accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print("\n" + "="*50)
print("ĐÁNH GIÁ MODEL RESNET34")
print("="*50)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print("="*50)

# ===== 3. VẼ BIỂU ĐỒ METRICS =====
metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
metrics_values = [precision * 100, recall * 100, f1 * 100, accuracy * 100]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='white', linewidth=2)

# Thêm giá trị trên bar
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.ylabel('Phần trăm (%)', fontsize=14)
plt.title('Đánh giá Model ResNet34', fontsize=18, fontweight='bold', pad=20)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.xticks(fontsize=13)
plt.yticks(fontsize=11)

plt.tight_layout()
plt.savefig('metrics_resnet34.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Đã lưu: metrics_resnet34.png")
plt.close()

# ===== 4. VẼ BẢNG KẾT QUẢ =====
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

# Data cho bảng
table_data = [
    ['Chỉ Số', 'ResNet34'],
    ['Precision (P)', f'{precision:.4f}'],
    ['Recall (R)', f'{recall:.4f}'],
    ['F1-Score (F1)', f'{f1:.4f}'],
    ['Accuracy', f'{accuracy:.4f}']
]

# Tạo bảng
table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colWidths=[0.4, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.5, 2)

# Style header
for j in range(2):
    table[(0, j)].set_facecolor('#1565C0')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Style rows
for i in range(1, 5):
    table[(i, 0)].set_facecolor('#E3F2FD')
    table[(i, 1)].set_facecolor('#BBDEFB')
    table[(i, 1)].set_text_props(fontweight='bold')

plt.title('Xây dựng mô hình', fontsize=20, fontweight='bold', color='#1565C0', pad=30)
plt.tight_layout()
plt.savefig('table_resnet34.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Đã lưu: table_resnet34.png")
plt.close()

print("\n🎉 HOÀN TẤT! Đã tạo 3 file ảnh:")
print("   1. confusion_matrix_resnet34.png")
print("   2. metrics_resnet34.png")
print("   3. table_resnet34.png")
