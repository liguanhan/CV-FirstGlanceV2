import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = np.array([
[963,   0,  14,   2,   1,   1,   2,   1,  15,   1],
[  4, 978,   0,   0,   0,   0,   0,   0,   0,  18],
[ 15,   0, 936,  15,  10,   8,  11,   3,   2,   0],
[  6,   2,  15, 893,  10,  56,   8,   3,   5,   2],
[  1,   0,   5,  10, 969,   8,   3,   4,   0,   0],
[  1,   1,   9,  45,  12, 925,   1,   4,   0,   2],
[  3,   0,   9,  10,   3,   4, 970,   1,   0,   0],
[  5,   1,   2,   7,   6,   9,   0, 970,   0,   0],
[ 13,   5,   2,   2,   0,   0,   0,   0, 971,   7],
[  2,  19,   2,   2,   0,   0,   0,   0,   5, 970]
])

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (Accuracy: 95.45%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('logs/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩阵已保存到 logs/confusion_matrix.png")