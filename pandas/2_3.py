import pandas as pd

# 读取train.csv文件，以PassengerId列为行索引
df = pd.read_csv(r'C:\Users\Administrator\Desktop\AI技术基础作业\pandas\train.csv', index_col='PassengerId')

# 提取Survived列的数据作为目标向量y，其余为X
y = df['Survived']
X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])

# 将X中缺失数据用0填充
X.fillna(0, inplace=True)

# 处理X中的性别数据，将male用0替换，female用1替换
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# 对X中的Embarked进行one-hot编码转换
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

# 打印X和y的前5行数据
print(X.head(5))
print(y.head(5))
