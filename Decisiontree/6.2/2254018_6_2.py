import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 读取数据
df = pd.read_csv(r'D:\2024\课内\AI homework\Decisiontree\6.2\ad.data', header=None)

# 替换缺失值为-1
df.replace(' ?', -1, inplace=True)

# 将所有列转换为数值型
df = df.apply(pd.to_numeric, errors='coerce')

# 假设最后一列是标签
X = df.iloc[:, :-1]  # 特征
y = df.iloc[:, -1]   # 标签

# 将标签也转换为数值型
y = y.apply(lambda x: 1 if x == 'ad.' else 0)

# 切分数据集
X_train,X_test,y_train,y_test=train_test_split(X, y,stratify=y,random_state=42)

# 设置决策树分类器
tree = DecisionTreeClassifier(random_state=0)

# 设置参数网格
params = {
    'max_depth': [140, 150, 155, 160],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2, 3]
}

# 创建网格搜索对象
grid_search = GridSearchCV(estimator=tree, param_grid=params, cv=5, scoring='accuracy')

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 获取最优模型
best_tree = grid_search.best_estimator_

# 预测测试集
y_pred = best_tree.predict(X_test)

# 输出分类性能报告
print("分类性能报告：")
print(classification_report(y_test, y_pred))