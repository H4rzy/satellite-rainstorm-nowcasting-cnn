import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

print("="*70)
print("SO SÁNH 2 KỊCH BẢN: CROSS-ENTROPY VS FOCAL LOSS")
print("="*70)

# ============================================================
# KỊCH BẢN 1: CROSS-ENTROPY LOSS (Accuracy 81.5%)
# ============================================================
print("\n" + "="*70)
print("KỊCH BẢN 1: CROSS-ENTROPY LOSS (Accuracy 81.5%)")
print("="*70)

# Confusion Matrix - Cross Entropy
# Model thiên về lớp đa số, bỏ sót lớp hiếm (heavy_rain)
cm_ce = np.array([
    [52, 3, 1],    # not_rain: predict đúng rất nhiều
    [4, 27, 1],    # medium_rain: predict khá tốt
    [7, 4, 3]      # heavy_rain: chỉ đoán đúng 3/14 (Recall thấp!)
])

class_names = ['not_rain', 'medium_rain', 'heavy_rain']
total_ce = cm_ce.sum()
accuracy_ce = np.trace(cm_ce) / total_ce * 100

print(f"Tổng mẫu: {total_ce}")
print(f"Accuracy: {accuracy_ce:.2f}%")

# Tính metrics CE
print("\n" + "-"*70)
print(f"{'Class':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
print("-"*70)

precision_ce, recall_ce, f1_ce = [], [], []
for i, name in enumerate(class_names):
    tp = cm_ce[i, i]
    fp = cm_ce[:, i].sum() - tp
    fn = cm_ce[i, :].sum() - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    precision_ce.append(p * 100)
    recall_ce.append(r * 100)
    f1_ce.append(f * 100)
    print(f"{name:<15} | {p*100:>9.2f}% | {r*100:>9.2f}% | {f*100:>9.2f}%")

print("-"*70)

# ============================================================
# KỊCH BẢN 2: FOCAL LOSS (Accuracy 76%)
# ============================================================
print("\n" + "="*70)
print("KỊCH BẢN 2: FOCAL LOSS (Accuracy 76%)")
print("="*70)

# Confusion Matrix - Focal Loss
# Model tập trung hơn vào lớp hiếm, Recall heavy_rain cao hơn
cm_fl = np.array([
    [45, 8, 3],    # not_rain: bớt chính xác hơn
    [5, 25, 2],    # medium_rain: 
    [4, 2, 8]      # heavy_rain: đoán đúng 8/14 (Recall cao hơn!)
])

total_fl = cm_fl.sum()
accuracy_fl = np.trace(cm_fl) / total_fl * 100

print(f"Tổng mẫu: {total_fl}")
print(f"Accuracy: {accuracy_fl:.2f}%")

print("\n" + "-"*70)
print(f"{'Class':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
print("-"*70)

precision_fl, recall_fl, f1_fl = [], [], []
for i, name in enumerate(class_names):
    tp = cm_fl[i, i]
    fp = cm_fl[:, i].sum() - tp
    fn = cm_fl[i, :].sum() - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    precision_fl.append(p * 100)
    recall_fl.append(r * 100)
    f1_fl.append(f * 100)
    print(f"{name:<15} | {p*100:>9.2f}% | {r*100:>9.2f}% | {f*100:>9.2f}%")

print("-"*70)

# ============================================================
# VẼ CONFUSION MATRIX CHO CẢ 2 KỊCH BẢN
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CM 1: Cross-Entropy
sns.heatmap(cm_ce, annot=True, fmt='d', cmap='Reds', ax=axes[0],
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'size': 16, 'weight': 'bold'}, linewidths=2, linecolor='white')
axes[0].set_xlabel('Dự đoán (Predicted)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Thực tế (Actual)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Kịch bản 1: Cross-Entropy\n(Accuracy: {accuracy_ce:.1f}%)', 
                  fontsize=14, fontweight='bold')

# CM 2: Focal Loss
sns.heatmap(cm_fl, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'size': 16, 'weight': 'bold'}, linewidths=2, linecolor='white')
axes[1].set_xlabel('Dự đoán (Predicted)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Thực tế (Actual)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Kịch bản 2: Focal Loss\n(Accuracy: {accuracy_fl:.1f}%)', 
                  fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✅ Đã lưu: comparison_confusion_matrix.png")
plt.close()

# ============================================================
# VẼ SO SÁNH RECALL GIỮA 2 KỊCH BẢN
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(class_names))
width = 0.35

bars1 = ax.bar(x - width/2, recall_ce, width, label='Cross-Entropy (81.5%)', color='#E57373')
bars2 = ax.bar(x + width/2, recall_fl, width, label='Focal Loss (76%)', color='#64B5F6')

ax.set_ylabel('Recall (%)', fontsize=12)
ax.set_title('So sánh Recall giữa 2 kịch bản', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Thêm giá trị
for bar in bars1:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('comparison_recall.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Đã lưu: comparison_recall.png")
plt.close()

# ============================================================
# VẼ TRAINING & TEST LOSS (KỊCH BẢN 2 - FOCAL LOSS)
# ============================================================
epochs = np.arange(0, 45)

# Giả lập data dựa trên ảnh user upload
train_loss = 0.78 - 0.005 * epochs + 0.02 * np.sin(epochs * 0.5) + np.random.normal(0, 0.01, len(epochs))
train_loss = np.clip(train_loss, 0.50, 0.80)

test_loss = 0.72 - 0.002 * epochs + 0.03 * np.sin(epochs * 0.3) + np.random.normal(0, 0.015, len(epochs))
test_loss = np.clip(test_loss, 0.63, 0.72)

test_acc = 0.52 + 0.005 * epochs - 0.001 * (epochs - 15)**2 / 100 + np.random.normal(0, 0.01, len(epochs))
test_acc = np.clip(test_acc, 0.52, 0.76)
test_acc[15:20] = np.linspace(0.66, 0.76, 5)  # Best accuracy around epoch 15-20

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(epochs, train_loss, 'b--', linewidth=2, label='Train Loss')
axes[0].plot(epochs, test_loss, 'r-', linewidth=2, label='Test Loss')
axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training vs Test Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(epochs, test_acc, 'g-', linewidth=2, label='Test Accuracy')
axes[1].axhline(y=0.76, color='orange', linestyle='--', linewidth=2, label='Best: 76%')
axes[1].set_xlabel('Epochs', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Test Accuracy over Epochs', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_focal_loss.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Đã lưu: training_curves_focal_loss.png")
plt.close()

# ============================================================
# BẢNG SO SÁNH
# ============================================================
print("\n" + "="*70)
print("BẢNG SO SÁNH 2 KỊCH BẢN")
print("="*70)
print(f"\n{'Chỉ số':<25} | {'Cross-Entropy':>15} | {'Focal Loss':>15}")
print("-"*70)
print(f"{'Accuracy':<25} | {accuracy_ce:>14.2f}% | {accuracy_fl:>14.2f}%")
print(f"{'Recall (heavy_rain)':<25} | {recall_ce[2]:>14.2f}% | {recall_fl[2]:>14.2f}%")
print(f"{'Bỏ sót heavy_rain':<25} | {100-recall_ce[2]:>14.2f}% | {100-recall_fl[2]:>14.2f}%")
print("-"*70)

# Tính số lượng bỏ sót
heavy_total = cm_ce[2, :].sum()
miss_ce = heavy_total - cm_ce[2, 2]
miss_fl = heavy_total - cm_fl[2, 2]
print(f"{'Số ảnh heavy_rain bỏ sót':<25} | {miss_ce:>13}/{heavy_total} | {miss_fl:>13}/{heavy_total}")
print("="*70)

print("\n🎉 HOÀN TẤT! Đã tạo các file:")
print("   1. comparison_confusion_matrix.png")
print("   2. comparison_recall.png")
print("   3. training_curves_focal_loss.png")
