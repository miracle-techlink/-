import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = [u'simHei']  # 指定中文字体为黑体，防止乱码
plt.rcParams['axes.unicode_minus'] = False     # 使用ASCII字符，保证显示正确

# 修正文件路径和编码的分开
df = pd.read_csv(r'C:\Users\Administrator\Desktop\AI技术基础作业\sh_air_quality_1.csv', encoding='gbk')
# df.head()  
# df.info()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 将空气质量等级转换为数值
label_encoder = LabelEncoder()
df['质量等级'] = label_encoder.fit_transform(df['质量等级'])

# 特征和标签
X = df[['AQI指数', 'PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3']]
y = df['质量等级']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 选择最佳的k值
k_values = range(1, 21)  # k值范围
mean_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)  # 5折交叉验证
    mean_accuracies.append(scores.mean())

# 找到最佳k值
best_k = k_values[np.argmax(mean_accuracies)]
print(f'最佳k值: {best_k}')

# 使用最佳k值训练kNN分类器
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# 在测试集上测试分类器的准确率
y_pred = knn_best.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率: {accuracy:.2f}')

from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 创建一个 DataFrame 用于可视化
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)

# 添加标题和标签
plt.title('混淆矩阵')
plt.xlabel('预测等级')
plt.ylabel('实际等级')
plt.xticks(rotation=45)  # 旋转x轴标签以便于阅读
plt.yticks(rotation=0)    # 旋转y轴标签以便于阅读
plt.tight_layout()  # 自动调整布局
plt.show()