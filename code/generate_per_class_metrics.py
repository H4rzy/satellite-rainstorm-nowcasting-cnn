import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Dữ liệu Per-Class metrics cho 2 lớp
class_names = ['AnToan', 'NguyCo']

# Metrics cho từng lớp (%) - dựa trên confusion matrix đã tính
# TN=167, FP=33, FN=41, TP=159
precision = [80.29, 82.81]  # AnToan: 167/(167+41), NguyCo: 159/(159+33)
recall = [83.50, 79.50]     # AnToan: 167/(167+33), NguyCo: 159/(159+41)
f1_score = [81.86, 81.12]   # 2*P*R/(P+R)

# Vị trí x cho các nhóm
x = np.arange(len(class_names))
width = 0.25

# Tạo figure
fig, ax = plt.subplots(figsize=(8, 6))

# Vẽ 3 nhóm bar
bars1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
bars2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
bars3 = ax.bar(x + width, f1_score, width, label='F1-score', color='#2ca02c')

# Cấu hình trục
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Per-Class Precision / Recall / F1-score', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=12)
ax.legend(loc='upper right', fontsize=10)

# Giới hạn trục y
ax.set_ylim(0, 100)

# Grid
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('per_class_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Đã lưu: per_class_metrics.png")
plt.close()

# In kết quả
print("\n" + "="*60)
print("PER-CLASS METRICS - RESNET34")
print("="*60)
print(f"{'Class':<10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
print("-"*60)
for i, name in enumerate(class_names):
    print(f"{name:<10} | {precision[i]:>9.2f}% | {recall[i]:>9.2f}% | {f1_score[i]:>9.2f}%")
print("-"*60)
print(f"{'Macro Avg':<10} | {np.mean(precision):>9.2f}% | {np.mean(recall):>9.2f}% | {np.mean(f1_score):>9.2f}%")
print("="*60)
