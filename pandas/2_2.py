import pandas as pd
import numpy as np

# 自定义基于Z-score法的函数
def detect_outliers_zscore(data, threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    return np.where(np.abs(z_scores) > threshold)

# 读取数据
df = pd.read_csv(r'C:\Users\Administrator\Desktop\AI技术基础作业\pandas\outlier.csv')

# 检测english列的异常值
english_outliers = detect_outliers_zscore(df['english'])[0]  # 提取索引数组
df.loc[english_outliers, 'english'] = 90

# 检测computer列的异常值
computer_outliers = detect_outliers_zscore(df['computer'])[0]  # 提取索引数组
computer_mean = df['computer'][~df['computer'].index.isin(computer_outliers)].mean()

# 显式转换为整数类型
df.loc[computer_outliers, 'computer'] = int(computer_mean)  # 或者使用 round() 进行四舍五入

# 保存处理后的数据
df.to_csv('outlier_processed.csv', index=False)

# 打印处理后的数据
print(df)