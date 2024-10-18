import pandas as pd
from sklearn.cluster import KMeans
import dashscope
from http import HTTPStatus
import os

# 设置 Dashscope API 密钥
dashscope.api_key = 'sk-aa60aeabc333475fa3679f899c509704'

# 嵌入函数
def embed_with_str(text):
    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v3,
        input=text)
    if resp.status_code == HTTPStatus.OK:
        return resp.output['embeddings'][0]['embedding']
    else:
        raise Exception(f"Error in embedding: {resp.output}")

def extract_questions_by_clustering(df, n_clusters):
    """
    根据聚类从问题列表中抽取问题。

    参数：
    df: 包含问题的 DataFrame，要求有一列名为 'query'
    n_clusters: 聚类的数量

    返回：
    每个聚类中随机抽取的一条问题的 DataFrame
    """
    # 将问题转换为嵌入向量
    embeddings = df['query'].apply(embed_with_str).tolist()

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)

    # 将聚类标签添加到 DataFrame
    df['cluster'] = kmeans.labels_

    # 从每个聚类中随机抽取一条问题
    sampled_queries = df.groupby('cluster').apply(lambda x: x.sample(1)).reset_index(drop=True)

    return sampled_queries[['query', 'cluster']]

# 读取数据
file_path = os.path.join('data', 'raw', 'sample_data.xlsx')
df = pd.read_excel(file_path)
# 提取问题
result = extract_questions_by_clustering(df, n_clusters=10)
result.to_csv(os.path.join('data', 'processed', 'use','extract_questions.csv'), index=False)
df.to_csv(os.path.join('data', 'processed', 'use','extract_questions_clusterALL.csv'), index=False)

