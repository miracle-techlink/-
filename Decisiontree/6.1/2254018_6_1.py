import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

file_path = r"D:\2024\课内\AI homework\Decisiontree\6.1\buyCar.csv"
df = pd.read_csv(file_path,
                 encoding='gbk',        # 指定编码
)

# 删除列'CustomerID'
df = df.drop('CustomerID', axis=1)

# 对'Sex'列进行数值编码
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# 对'Income'列进行数值编码
df['Income'] = LabelEncoder().fit_transform(df['Income'])

# 对Bought列进行数值编码
df['Bought'] = LabelEncoder().fit_transform(df['Bought'])


car_data = df[["Age","Income","Sex"]].values
car_target = df["Bought"].values

# 切分数据集
X_train,X_test,y_train,y_test=train_test_split(car_data, car_target,
stratify=car_target,random_state=42)

# 训练决策树模型
tree=DecisionTreeClassifier(max_depth=4,random_state=0) #设置max_depth=4，限制树的最大深度
tree.fit(X_train,y_train)
print("剪枝后训练集上的准确度：{:.3f}".format(tree.score(X_train,y_train)))
print("剪枝后测试集上的准确度：{:.3f}".format(tree.score(X_test,y_test)))

# 绘制决策树
plt.figure(figsize=(55,20))
plot_tree(tree, filled=True, rounded=True, 
          feature_names=["Age", "Income", "Sex"],
          class_names=["不买", "买"], 
          fontsize=20)
plt.show()

# 创建新的测试数据
new_data = [[28, 2, 1]]  # 28岁，高收入(2)，男性(1)

# 使用训练好的模型进行预测
prediction = tree.predict(new_data)

# 输出预测结果
if prediction[0] == 1:
    print("预测结果：会购买汽车")
else:
    print("预测结果：不会购买汽车")

# 如果想要查看预测的概率
probabilities = tree.predict_proba(new_data)
print(f"购买概率: {probabilities[0][1]:.2%}")
print(f"不购买概率: {probabilities[0][0]:.2%}")