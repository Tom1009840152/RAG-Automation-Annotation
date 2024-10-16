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


# 定义文件路径
file_path = os.path.join('data', 'processed', 'use', 'df_boundary.csv')
df_all = pd.read_csv(file_path)

# 拆分数据集
train_df, temp_df = train_test_split(df_all, test_size=0.3, random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 输出数据集大小
print(f"训练集大小: {train_df.shape}")
print(f"测试集大小: {test_df.shape}")
print(f"验证集大小: {val_df.shape}")

# 标签映射
labels = {1: 1, 0: 0}


# 自定义数据集类
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

# 简化的分类器模型
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

# 训练模型函数
def train_model(model, train_dataloader, optimizer, criterion, device):
    model.to(device)
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    tk0 = tqdm(train_dataloader, smoothing=0, mininterval=1.0)
    for batch in tk0:
        optimizer.zero_grad()
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predicted = outputs.round()
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(train_dataloader)
    avg_acc = total_correct / total_samples
    print(f'Average Loss: {avg_loss}, Average Accuracy: {avg_acc}')


def validate_model(model, dataloader, device):
    model.eval()  # 将模型设置为评估模式
    predictions = []
    true_labels = []
    with torch.no_grad():  # 在评估阶段不计算梯度
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().cpu().numpy())  # 将预测结果保存到列表中
            true_labels.extend(labels.squeeze().cpu().numpy())  # 将真实标签保存到列表中

    # 计算AUC
    auc_score = roc_auc_score(true_labels, predictions)
    print(f'Validation AUC: {auc_score}')
    return auc_score


def main():
    device = torch.device('cpu')
    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)
    test_dataset = CustomDataset(test_df)
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=20)
    test_dataloader = DataLoader(test_dataset, batch_size=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier().to(device)  # 将模型移动到正确的设备上
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        train_model(model, train_dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}')
        validate_model(model, val_dataloader, device)

    # 保存模型
    torch.save(model.state_dict(), 'model\classifier_5epoch.pth')
    print("模型已保存。")

if __name__ == "__main__":
    main()

with pd.ExcelWriter(r'data\processed\use\data_splits.xlsx', engine='xlsxwriter') as writer:
    # 将每个数据集写入不同的工作表
    train_df.to_excel(writer, sheet_name='Train', index=False)
    test_df.to_excel(writer, sheet_name='Test', index=False)
    val_df.to_excel(writer, sheet_name='Validation', index=False)


