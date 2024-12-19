## 提交说明
Scikit-learn内置数据集：wine，178个样本，13个特征，分3个类别（0、1、2）
from sklearn.datasets import load_wine
wine = load_wine()
(1) 先对红酒数据进行缩放处理 StandardScaler，再建立SVC分类器，用网格搜索工具GridSearchCV，自动寻找SVC最优超参数组合，输出测试集的分类性能报告。
(2) 将（1）中步骤构建管道，重新训练,输出最优分类器在测试集上的分类准率和ROC AUC值。