import dashscope
from http import HTTPStatus
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
dashscope.api_key = 'sk-aa60aeabc333475fa3679f899c509704'

class Classifier(nn.Module):
    def __init__(self, num_classes=1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        X = F.relu(self.fc1(inputs))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        X = self.fc3(X)
        X = self.sigmoid(X)
        return X


### 2. 加载模型状态
model = Classifier()
model.load_state_dict(torch.load('model\classifier_5epoch.pth', map_location=torch.device('cpu')))
model.eval()   # 将模型置于评估模式

def embed_with_str(text):
    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v3,
        input=text)
    if resp.status_code == HTTPStatus.OK:
        return resp.output['embeddings'][0]['embedding']
    else:
        raise Exception(f"Error in embedding: {resp.output}")


# 单个测试
input_s = "皇玺集团在2023年第一季度的毛利是多少？"
input_tk = embed_with_str(input_s)
input_tk_tensor = torch.tensor(input_tk)

import time
start_time = time.time()
with torch.no_grad():  # 在推理阶段不计算梯度
    prediction = model(input_tk_tensor)
end_time = time.time()
print(f"Model prediction took {end_time - start_time} seconds. {input_s} : {prediction} ")

# 批量使用---------------------------------------------------------------------------------------------------------------
df = pd.read_excel('工作簿1.xlsx')  # 读取Excel文件
df['query'] = df['query'].astype(str)  # 确保query列中的所有数据都是字符串类型

predictions = []
for query in df['query']:
    input_tk = embed_with_str(query)
    input_tk_tensor = torch.tensor(input_tk)  # 将嵌入向量转换为 PyTorch 张量并移动到 GPU

    with torch.no_grad():  # 在推理阶段不计算梯度
        prediction = model(input_tk_tensor)
    predictions.append(prediction.item())  # 将预测结果添加到列表中

# 将预测结果添加到DataFrame中
df['prediction'] = predictions
df['result'] = df['prediction'].apply(lambda x: '范围内' if x > 0.5 else '范围外')

# 保存结果到新的Excel文件
df.to_excel('predictions2.xlsx', index=False)  # 保存预测结果