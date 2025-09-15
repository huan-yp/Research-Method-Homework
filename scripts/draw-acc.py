import numpy as np
import matplotlib.pyplot as plt

# 数据
models = ['CLIP', 'ALBEF', 'BLIP']
r1_acc = np.array([88.0, 94.1, 96.0])
r5_acc = np.array([85.5, 82.8, 68.7][::-1])

# 组合标签：R1 和 R5 分开，中间插入空格
labels = (
    ['CLIP-TR1', 'ALBEF-TR1', 'BLIP-TR1'] +
    [''] * 2 +  # 中间插入 2 个空字符串作为间隔
    ['CLIP-IR1', 'ALBEF-IR1', 'BLIP-IR1']
)

# 组合准确率数据，中间用 NaN 占位（不会画柱子）
acc_combined = np.concatenate([r1_acc, [np.nan]*2, r5_acc])

# 颜色
color_r1 = '#1f77b4'
color_r5 = '#ff7f0e'

# 创建颜色列表
colors = [color_r1]*3 + ['white']*2 + [color_r5]*3

# 绘图
x_pos = np.arange(len(labels))
plt.figure(figsize=(8, 5))
bars = plt.bar(x_pos, acc_combined, color=colors, edgecolor='black')

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, acc_combined)):
    if not np.isnan(val):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=16)

# 坐标轴与标题
plt.ylim(0, 110)
plt.ylabel('Accuracy (%)', fontsize=16)
plt.title('Model Comparison on TR and IR Tasks(R1)', fontsize=16)
plt.xticks(x_pos, labels, rotation=45, ha='right', fontsize=16)
plt.tight_layout()

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_r1, label='R1'),
                   Patch(facecolor=color_r5, label='R5')]
plt.legend(handles=legend_elements, loc='lower right')

plt.show()