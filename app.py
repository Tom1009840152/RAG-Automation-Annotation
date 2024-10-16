from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# 设置上传文件的保存路径
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # 渲染 HTML 文件
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 保存文件
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 在这里可以调用你的模型训练或标注函数
    # train_model(file_path) 或 annotate_file(file_path)

    return jsonify({'success': True, 'filename': file.filename})

if __name__ == '__main__':
    app.run(debug=True)