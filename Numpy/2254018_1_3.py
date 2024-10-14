import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [u'simHei']  # 指定中文字体为黑体，防止乱码
plt.rcParams['axes.unicode_minus'] = False     # 使用ASCII字符，保证显示正确

# 创建集合A，包含100个随机点
A = np.random.rand(100, 2)

# 创建集合B，包含10个随机点
B = np.random.rand(10, 2)

# 使用KDTree查找B中每个点在A中的最近邻
tree = KDTree(A)
distances, indices = tree.query(B, k=1)

# 输出结果
for i in range(len(B)):
    print(f"B中的点 {B[i]} 的最近邻在A中的点是 {A[indices[i]]}，距离为 {distances[i]}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(A[:, 0], A[:, 1], c='blue', label='集合A (100个点)', alpha=0.5)
plt.scatter(B[:, 0], B[:, 1], c='red', label='集合B (10个点', alpha=0.7)

# 绘制最近邻连接
for i in range(len(B)):
    plt.plot([B[i][0], A[indices[i]][0]], [B[i][1], A[indices[i]][1]], 'k--', alpha=0.5)

plt.title('集合A和集合B的最近邻可视化')
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.legend()
plt.grid()
plt.show()