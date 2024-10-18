# RAG Automation Annotation Project

## Overview

This project aims to automate the annotation process for RAG (Retrieval-Augmented Generation) applications by leveraging historical log data. The goal is to reduce manual annotation efforts by using a model to find similar previously annotated data and automatically annotate new data based on these similarities.

![image](https://github.com/user-attachments/assets/3883519f-9963-49b1-bc89-4b7c9ac992d3)


## Getting Started

### Prerequisites

- Python 3.10
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG-Automation-Annotation.git
   cd RAG-Automation-Annotation
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Data Preprocessing**:
![未命名文件 (2)](https://github.com/user-attachments/assets/36a67c76-fcd3-4e39-9db6-6ff7591235c0)



   - Run `data_preprocessing.py` to clean and preprocess the raw data.
   ```bash
   python src/data_preprocessing.py
   ```

2. **Model Training**:
   - Train the similarity model using `model_training.py`.
   ```bash
   python src/model_training.py
   ```

3. **Model Inference**:
   - Use `model_inference.py` to find similar historical annotations for new queries.
   ```bash
   python src/model_inference.py
   ```

## Code Explanation

- **`data_preprocessing.py`**: Handles loading and cleaning of raw data.
- **`model_training.py`**: Trains a simple similarity model using TF-IDF and Nearest Neighbors.
- **`model_inference.py`**: Finds similar historical annotations for new queries using the trained model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [li_zhichao2022@163.com].
