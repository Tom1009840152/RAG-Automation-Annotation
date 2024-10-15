import pandas as pd
import numpy as np
import os

# 定义文件路径
file_path = os.path.join('..', 'data', 'raw', 'my_log_data.xlsx')
df = pd.read_excel(file_path)

# 首先将日期列转换为日期格式
df['every_date'] = pd.to_datetime(df['every_date'])

# 按日期降序排序
df_sorted = df.sort_values(by='every_date', ascending=False)

# 去除重复值，保留日期最近的一行
df_unique = df_sorted.drop_duplicates(subset=['query', 'answer', 'boundary', 'correct'], keep='first')

duplicates_removed = len(df) - len(df_unique)
# 打印去重后的数据框和去除的重复值数量
print(f"去除了 {duplicates_removed} 个重复值")

#分类----------------------------------------
# 计算每个 query 中 boundary 的众数
boundary_mode = df_unique.groupby('query')['boundary'].transform(lambda x: x.mode()[0])

# 标记 boundary 值等于众数的行为 1，其他为 0
df_unique['boundary_mark'] = (df_unique['boundary'] == boundary_mode).astype(int)

# 提取 boundary_mark 为 1 的行
df_cat = df_unique[df_unique['boundary_mark'] == 1][['query', 'boundary']]

# 去重
df_cat = df_cat.drop_duplicates()

#日志入库----------------------------------------
df_unique_right = df_sorted.drop_duplicates(subset=['query', 'answer', 'correct'], keep='first')

def filter_by_correct(df, n):
    # 定义一个空列表来存储结果
    result = []

    # 按 query 分组
    grouped = df.groupby('query')

    for name, group in grouped:
        # 对于 correct 为 0 的行，保留最近的 n 行
        correct_0 = group[group['correct'] == 0].head(n)
        # 对于 correct 为 1 的行，保留最近的 n 行
        correct_1 = group[group['correct'] == 1].head(n)

        # 将结果添加到列表中
        result.append(correct_0)
        result.append(correct_1)

    # 将结果合并为一个数据框
    return pd.concat(result)

# 使用函数进行筛选
n = 3# 例如，保留最近的 2 行
filtered_df = filter_by_correct(df_unique, n)