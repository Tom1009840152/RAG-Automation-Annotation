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

# 嵌入函数
def embed_with_str(text):
    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v3,
        input=text)
    if resp.status_code == HTTPStatus.OK:
        return resp.output['embeddings'][0]['embedding']
    else:
        raise Exception(f"Error in embedding: {resp.output}")


# 标签映射
labels = {1: 1, 0: 0}

class CustomDataset(Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['boundary'].tolist()]
        self.texts = df['query'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        embedding = embed_with_str(text)
        embedding_tensor = torch.tensor(embedding)
        batch_y = self.labels[idx]
        return embedding_tensor, batch_y


# 定义文件路径
file_path = os.path.join('data', 'processed', 'use', 'data_splits.xlsx' )
test_data = pd.read_excel(file_path, sheet_name='Test')
test_dataset = CustomDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)


# 初始化一个列表来保存所有的embeddings
all_embeddings = []
y_true = []

# 遍历DataLoader
for batch in test_dataloader:
    embeddings, labels = batch  # 假设你的CustomDataset返回的是(embedding_tensor, label)
    all_embeddings.append(embeddings)
    y_true.extend(labels.numpy())

# 使用torch.cat将列表中的所有张量连接成一个张量
all_embeddings_tensor = torch.cat(all_embeddings, dim=0)


### 4. 使用模型进行预测.现在，你可以使用模型对新数据进行预测了。假设你有一个名为`input_tensor`的张量，你想要用模型来预测它：
with torch.no_grad():  # 在推理阶段不计算梯度
    prediction = model(all_embeddings_tensor)

y_score = prediction.numpy().flatten()  # 将预测结果转换为numpy数组，并确保它是一维的，以匹配y_true的形状




# Calculate the ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Simple (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Save the plot to a file
plt.savefig('model/roc_curve.png')
plt.show()