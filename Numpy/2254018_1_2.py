import numpy as np

# 生成形状为(10, 4)的随机整数数组，元素在[0, 100)中
data = np.random.randint(0, 100, size=(10, 4))

# 输出原始数据
print("原始数据：")
print(data)

# Z-score标准化
mean = np.mean(data, axis=0)  # 计算每列的均值
std = np.std(data, axis=0)    # 计算每列的标准差
normalized_data = (data - mean) / std  # 标准化

# 输出标准化后的数据
print("\n标准化后的数据：")
print(normalized_data)