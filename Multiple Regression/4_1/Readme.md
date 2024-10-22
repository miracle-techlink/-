# 作业要求
（1）读入给定的天期测量数据weather.csv，存入一个DataFrame中。
 注意：Description列是分类变量，需要进行编码转换，转为n-1个二进制特征。
（2）以Temperature_c列为目标，整理数据（X,y），以7：3划分成训练集和测试集。
（3）利用sklearn的LinearRegression，创建线性回归模型，拟合上述数据。
（4）对测试数据进行预测，绘制预测值与实际值的散点图。
（5）计算MAE、MSE、RMSE、R2指标，并存入数据框中。