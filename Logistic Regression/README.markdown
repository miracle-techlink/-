## 提交说明
Scikit-learn内置数据集：wine, 178个样本，13个特征，分3个类别（0、1、2）
**提示:**
from sklearn.datasets import load_wine
wine = load_wine()
（1）用softmax回归创建一个分类器，考虑是否需要数据预处理，以及最优超参数(max_iter,C)选择问题。
（2）评估最优模型的测试准确率和ROC AUC值。
（3）输出测试数据属于3个类别的概率、类别，注意要求构成dataframe对象