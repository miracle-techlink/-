from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 加载wine数据集
wine = load_wine()
X, y = wine.data, wine.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建一个softmax回归分类器
softmax_clf = LogisticRegression(multi_class='multinomial', solver='saga', random_state=42)

# 创建一个管道，包括特征标准化和softmax回归
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('softmax', softmax_clf)
])

# 定义超参数的网格
param_grid = {
    'softmax__max_iter': [100, 200, 300, 400, 500],
    'softmax__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

# 使用网格搜索来找到最优的超参数
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


# 使用最优的超参数训练模型
best_softmax_clf = grid_search.best_estimator_
best_softmax_clf.fit(X_train, y_train)

# 预测测试集
y_pred = best_softmax_clf.predict(X_test)
y_pred_proba = best_softmax_clf.predict_proba(X_test)

# 计算测试准确率
test_accuracy = accuracy_score(y_test, y_pred)

# 为了计算ROC AUC值，我们需要将标签二值化
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# 计算ROC AUC值
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')

# 输出分类性能报告
report = classification_report(y_test, y_pred, target_names=wine.target_names)

print(classification_report(y_test, y_pred))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
print("ROC AUC: {:.2f}%".format(roc_auc * 100))