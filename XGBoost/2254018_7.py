import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import json

# 读取数据
data = pd.read_csv(r'D:\2024\课内\AI homework\XGBoost\telco-churn.csv')

# 步骤 1: 删除 'customerID' 列
data = data.drop(columns=['customerID'])

# 步骤 2: 将 'TotalCharges' 转换为数值，强制错误为 NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# 处理转换后 'TotalCharges' 中可能的缺失值
data['TotalCharges'].fillna(0, inplace=True)

# 步骤 3: 编码分类变量
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 步骤 4: 拆分特征 (X) 和目标 (y)
X = data.drop(columns=['Churn'])
y = data['Churn']

# 随机打乱并拆分数据 (80% 训练集, 20% 测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 调整模型参数以加快训练速度
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=42, use_label_encoder=False)

# 训练模型
xgb_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = xgb_model.predict(X_test)

# 生成分类报告
report = classification_report(y_test, y_pred, output_dict=True)

# 将模型保存为 JSON 文件
model_json = xgb_model.get_booster().save_config()
with open('D:/2024/课内/AI homework/XGBoost/xgboost_model.json', 'w') as f:
    json.dump(model_json, f)

# 打印分类报告
print(report)
