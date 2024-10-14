import pandas as pd
import numpy as np

df = pd.read_excel("new_data_m_clean.xlsx")

# 先去除重复值
df_unique = df.drop_duplicates().copy()

# 再去除冲突值(先标记再导出)
df_unique['B_unique_count_per_A'] = df_unique.groupby('query')['category'].transform('nunique')

# 标记违反规则的行：如果B_unique_count_per_A大于1，则标记为True
df_unique['Violation'] = df_unique['B_unique_count_per_A'] > 1

#导出文件，手工处理后再导入
df_unique.to_excel("new_data_m_clean.xlsx",index = False)