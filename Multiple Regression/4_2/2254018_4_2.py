import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [u'simHei']  # 指定中文字体为黑体，防止乱码
plt.rcParams['axes.unicode_minus'] = False     # 使用ASCII字符，保证显示正确

# 产生带白噪声的正弦波数据集
X = np.linspace(0, 10, 100).reshape(-1, 1)  # 生成100个样本，1个特征
y = np.sin(X) + 0.1 * np.random.randn(100, 1)  # 添加白噪声

# 先用多项式特征变换（degree=10)扩展特征
poly_features = PolynomialFeatures(degree=10)
X_poly = poly_features.fit_transform(X)  # 扩展特征

# 创建管道
pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=10)),  # 多项式特征变换
    ('ridge_reg', Ridge(alpha=1))  # 岭回归估计器
])

# 计算学习曲线
train_sizes, train_scores, test_scores = learning_curve(pipeline, X_poly, y, cv=5)

# 计算平均分数
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

# 绘制学习曲线
plt.plot(train_sizes, train_scores_mean, label='训练集得分')
plt.plot(train_sizes, test_scores_mean, label='测试集得分')
plt.xlabel('训练样本数')
plt.ylabel('得分')
plt.title('学习曲线')
plt.legend()
plt.show()

# 先拟合管道
pipeline.fit(X, y)  # 用训练数据拟合模型

# 从[0,10]等间距取1000数据点构成 Xnew
Xnew = np.linspace(0, 10, 1000).reshape(-1, 1)  # 生成1000个样本

# 使用训练好的模型进行预测
y_pred = pipeline.predict(Xnew)  # 预测输出

# 绘制拟合结果
plt.scatter(X, y, label='原始数据', color='blue', s=10)  # 原始数据点
plt.plot(Xnew, y_pred, label='拟合结果', color='red')  # 拟合曲线
plt.xlabel('X')
plt.ylabel('y')
plt.title('模型拟合结果')
plt.legend()
plt.show()