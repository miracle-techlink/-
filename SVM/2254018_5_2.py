from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# 加载wine数据集
wine = load_wine()
X, y = wine.data, wine.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义SVC分类器
svc = SVC(probability=True, random_state=42)

# 定义超参数网格
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1],
    'svc__kernel': ['rbf', 'linear']
}

# 创建一个管道，包括特征缩放和SVC
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', svc)
])

# 使用网格搜索来找到最优的超参数
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 使用grid_search中找到的最优参数来训练模型
best_svc_clf = grid_search.best_estimator_

# 使用最优分类器预测测试集
y_pred = best_svc_clf.predict(X_test)    # 返回预测的类别标签。
y_pred_proba = best_svc_clf.predict_proba(X_test)  # 返回预测的概率值。

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
