# 导入必需库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
plt.rcParams['font.sans-serif'] = [u'simHei']  # 指定中文字体为黑体，防止乱码
plt.rcParams['axes.unicode_minus'] = False     # 使用ASCII字符，保证显示正确

# 导入数据集
df = pd.read_csv(r'C:\Users\Administrator\Desktop\AI homework\Multiple Regression\dataset\weather (1).csv')

# 对Description列进行独热编码
df = pd.get_dummies(df, columns=['Description'], drop_first=True)

# 定义特征和目标
X = df.drop('Temperature_c', axis=1)  # 特征
y = df['Temperature_c']                 # 目标

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X_train, y_train)

# 对测试数据进行预测
y_pred = model.predict(X_test)

# 输出线性回归方程
coefficients = model.coef_  # 获取特征的系数
intercept = model.intercept_  # 获取截距

# 打印方程
equation = "y = {:.4f}".format(intercept)
for i, coef in enumerate(coefficients):
    equation += " + {:.4f} * {}".format(coef, X.columns[i])

print("线性回归方程:", equation)

# 计算指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
r2 = r2_score(y_test, y_pred)

# 将结果存入数据框
metrics_df = pd.DataFrame({
    '指标': ['MAE', 'MSE', 'RMSE', 'R2'],
    '值': [mae, mse, rmse, r2]
})

print("误差指标为:",metrics_df)

# 绘制预测值与实际值的散点图
plt.scatter(y_test, y_pred)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('预测值与实际值的散点图')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # 添加对角线
plt.show()

