import pandas as pd
import numpy as np
import os

# 定义文件路径
file_path = os.path.join('..', 'data', 'raw', 'sample_data.xlsx')
df = pd.read_excel(file_path)

# 首先将日期列转换为日期格式
df['every_date'] = pd.to_datetime(df['every_date'])

# 按日期降序排序
df_sorted = df.sort_values(by='every_date', ascending=False)

# 去除重复值，保留日期最近的一行
df_unique = df_sorted.drop_duplicates(subset=['query', 'answer', 'boundary', 'correct'], keep='first').copy()

duplicates_removed = len(df) - len(df_unique)
# 打印去重后的数据框和去除的重复值数量
print(f"去除了 {duplicates_removed} 个重复值")


# 分类----------------------------------------
df_cat_use = df_unique.copy()
# 计算每个 query 中 boundary 的众数
boundary_mode = df_cat_use.groupby('query')['boundary'].transform(lambda x: x.mode()[0])

# 标记 boundary 值等于众数的行为 1，其他为 0
df_cat_use['boundary_mark'] = (df_cat_use['boundary'] == boundary_mode).astype(int)

# 提取 boundary_mark 为 1 的行
df_cat = df_cat_use[df_cat_use['boundary_mark'] == 1][['query', 'boundary']]

# 去重
df_cat = df_cat.drop_duplicates()

# 输出check
boundary_counts = df_cat_use.groupby('query')['boundary'].nunique()  # 检查每个 query 是否有多个不同的 boundary
queries_with_multiple_boundaries = boundary_counts[boundary_counts > 1].index  # 过滤出有多个不同 boundary 的 query
df_cat_check = df_cat_use[df_cat_use['query'].isin(queries_with_multiple_boundaries)]  # 将这些 query 的行复制到 df_cat_check
df_cat_check.to_csv(os.path.join('..', 'data', 'processed', 'check', 'df_cat_check.csv'), index=False)  # 保存 df_cat_check 到 data/processed 文件夹


# 保存1----------------------------------------
# 保存 df_cat 到 data/processed 文件夹
df_cat.to_csv(os.path.join('..', 'data', 'processed', 'use', 'df_boundary.csv'), index=False)


# 日志入库----------------------------------------
df_correct_use = df_unique.copy()
df_unique_right = df_correct_use.drop_duplicates(subset=['query', 'answer', 'correct'], keep='first')

# 输出check
correct_counts = df_correct_use.groupby(['query', 'answer'])['correct'].nunique()   # 检查每个 (query, answer) 组合是否有多个不同的 correct
queries_with_multiple_corrects = correct_counts[correct_counts > 1].index   # 过滤出有多个不同 correct 的 (query, answer) 组合
df_correct_check = df_correct_use[df_correct_use.set_index(['query', 'answer']).index.isin(queries_with_multiple_corrects)]  # 将这些组合的行复制到 df_correct_check
df_correct_check.to_csv(os.path.join('..', 'data', 'processed',  'check', 'df_correct_check.csv'), index=False)  # 保存 df_correct_check 到 data/processed 文件夹


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
n = 3  # 例如，保留最近的 2 行
df_correct = filter_by_correct(df_correct_use, n)



# 保存2----------------------------------------
# 保存 filtered_df 到 data/processed 文件夹
df_correct.to_csv(os.path.join('..', 'data', 'processed', 'use','df_right.csv'), index=False)

#%%
