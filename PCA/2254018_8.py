# 2254018_8.py
# PCA可视化iris数据
# - 利用PCA对iris降维到二维空间
# - 可视化降维后的数据，画出散点图
# - 通过逻辑回归对变换后的样本进行分类，并测试其性能

# 先导入数据集并且查看数据集
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder  # 导入LabelEncoder
from sklearn.model_selection import train_test_split  # 导入训练测试集划分
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归
from sklearn.metrics import accuracy_score, classification_report  # 导入性能评估

# 导入数据
iris = pd.read_csv('iris.csv')

# 可视化数据集情况
sns.pairplot(iris, hue='species')
plt.show()

# 将'species'列转换为数值
label_encoder = LabelEncoder()  # 创建LabelEncoder实例
iris['species'] = label_encoder.fit_transform(iris['species'])  # 将'species'列转换为数值

# 将DataFrame转换为NumPy数组
iris_flat = iris.drop('species', axis=1).values  # 去掉'species'列

# PCA降维
pca = PCA(n_components=2)  # 将成分数量设置为2以降维到二维
X_reduced = pca.fit_transform(iris_flat)  # 使用展平后的数据进行PCA
print(f'成分数量：{pca.n_components_}')
print(f'这些成分的累计方差贡献率:{pca.explained_variance_ratio_.sum()}')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_reduced, iris['species'], test_size=0.2, random_state=42)

# 逻辑回归分类
logistic_model = LogisticRegression()  # 创建逻辑回归模型
logistic_model.fit(X_train, y_train)  # 训练模型

# 预测
y_pred = logistic_model.predict(X_test)  # 进行预测

# 性能评估
accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
print(f'模型准确率: {accuracy:.2f}')
print('分类报告:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))  # 打印分类报告

# 可视化降维后的数据
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=iris['species'])  # 使用转换后的'species'
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')  # 添加y轴标签
plt.title('PCA of Iris Dataset')  # 添加标题
plt.show()