# RAG 自动化标注项目

## 项目简介

本项目旨在通过利用历史标注数据和大模型，实现自动化标注，从而减少人工标注的工作量。通过相似度匹配，项目可以自动标注新数据，达到节省人力的效果。

## 项目结构

```
RAG-Automation-Annotation/
├── README.md
├── requirements.txt
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_inference.py
│   └── utils.py
└── data/
    ├── raw/
    │   └── historical_data.csv
    └── processed/
```

## 安装指南

1. 克隆本仓库到本地：

   ```bash
   git clone https://github.com/yourusername/RAG-Automation-Annotation.git
   ```

2. 进入项目目录：

   ```bash
   cd RAG-Automation-Annotation
   ```

3. 安装项目依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

### 数据预处理

运行 `data_preprocessing.py` 以加载和清洗原始数据：

```bash
python src/data_preprocessing.py
```

### 模型训练

使用 `model_training.py` 训练相似度模型：

```bash
python src/model_training.py
```

### 模型推理

使用 `model_inference.py` 查找与新问题相似的历史标注：

```bash
python src/model_inference.py
```

## 贡献指南

欢迎对本项目的贡献！如果你有任何建议或发现问题，请提交 issue 或 pull request。

## 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。

## 联系方式

如有任何问题或建议，请联系 [li_zhichao2022@163.com]。
