# knn函数的修改
def knn(X_train, y_train, X_test, k=1, p=2):
    # 计算闵可夫斯基距离
    distances = np.linalg.norm(X_train[:, np.newaxis] - X_test, ord=p, axis=2)
    # 选择最近的k个邻居
    nearest_indices = np.argsort(distances, axis=0)[:k]
    # 进行投票
    nearest_labels = y_train[nearest_indices].flatten()  # 确保是1D数组
    return np.argmax(np.bincount(nearest_labels))

# 导入所需的库与数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_p = 1
best_accuracy = 0

# 比较不同p值的准确率
for p in range(1, 8):
    y_pred = [knn(X_train, y_train, x_test, k=1, p=p) for x_test in X_test]
    accuracy = accuracy_score(y_test, y_pred)
    print(f'p={p}, 准确率={accuracy:.2f}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_p = p

print(f'最佳p值: {best_p}, 最佳准确率: {best_accuracy:.2f}')

# 使用最佳p值重新训练模型
y_pred_best = [knn(X_train, y_train, x_test, k=1, p=best_p) for x_test in X_test]
final_accuracy = accuracy_score(y_test, y_pred_best)
print(f'使用最佳p值的最终准确率: {final_accuracy:.2f}')