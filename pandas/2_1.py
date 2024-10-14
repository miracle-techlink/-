import pandas as pd

# 读取文件
df = pd.read_csv(r'C:\Users\Administrator\Desktop\AI技术基础作业\score_test.csv')

# 用所在列的均值填充缺失值
df.fillna(df.mean(), inplace=True)

# 计算每个考生的总分
df['总分'] = df.iloc[:, 1:4].sum(axis=1)

# 将总分转换为三个档次的成绩
def grade(score):
    if score >= 85:
        return '优秀'
    elif score >= 60:
        return '合格'
    else:
        return '不合格'

df['总评'] = df['总分'].apply(grade)

# 将结果存入score2.csv
df.to_csv('score2.csv', index=False)