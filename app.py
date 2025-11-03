import os
import torch
import librosa
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch.nn.functional as F
from nnmodels import CNN_2D_Attention_Model
from utils import device

app = Flask(__name__, static_folder='static')

# 确保上传文件目录存在
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 加载模型
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result/CNN_2D_Attention_Model.pt')
model = CNN_2D_Attention_Model(10).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # 确保模型处于评估模式

# 流派列表和对应的颜色
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
COLORS = {
    'blues': '#0033cc',     # 深蓝色
    'classical': '#e6b800', # 金色
    'country': '#996633',   # 棕色
    'disco': '#cc00cc',     # 紫色
    'hiphop': '#ff3300',    # 橙红色
    'jazz': '#009999',      # 青色
    'metal': '#333333',     # 深灰色
    'pop': '#ff66b3',       # 粉色
    'reggae': '#00cc00',    # 绿色
    'rock': '#cc0000'       # 红色
}

def process_audio(file_path, duration=30):
    """
    处理音频文件:
    1. 加载音频
    2. 提取中间30秒
    3. 生成梅尔频谱图
    4. 通过模型预测流派
    """
    # 加载音频 - 使用与训练时相同的采样率
    y, sr = librosa.load(file_path, sr=22050)
    
    # 提取中间30秒 - 使用与训练时相同的数据长度 (660000 samples = 30s at 22050Hz)
    data_length = 660000  # 与训练时相同 (与 main.py 中的 config['dataset']['data_length'] 一致)
    if len(y) > data_length:
        mid_point = len(y) // 2
        mid_samp = data_length // 2
        y = y[mid_point-mid_samp:mid_point-mid_samp+data_length]
    
    # 确保音频长度一致
    if len(y) < data_length:
        # 如果音频太短，用零填充
        padding = data_length - len(y)
        y = np.pad(y, (0, padding), 'constant')
    
    # 生成梅尔频谱图 - 使用与训练时相同的参数
    n_mels = 256  # 与训练时相同 (与 main.py 中的 config['dataset']['n_mels'] 一致)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram)  # 不使用 ref=np.max，与训练时保持一致
    
    # 转为张量并预测 - 确保输入格式与模型训练时完全一致
    # CNN_2D_Attention_Model 期望输入形状为 [batch_size, height, width]
    # mel_spectrogram_db 形状为 [n_mels, time]
    mel_tensor = torch.tensor(mel_spectrogram_db, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 获取最后一层softmax之前的原始输出（logits）
        # 我们通过修改模型的调用方式来获取logits
        model.eval()
        # 运行到mlp输出，但不应用softmax
        x = mel_tensor
        B, H, W = x.shape
        x = x.view(B, 1, H, W)
        x = model.conv(x)
        conv_out = x.view(B, -1)
        query = conv_out.unsqueeze(1)
        attn_out, _ = model.attention(query, query, query)
        attn_out = attn_out.squeeze(1)
        x = model.fc(attn_out)
        logits = model.mlp(x)
        
        # 使用一个较高的温度参数的softmax函数来平滑概率分布
        temperature = 5.0  # 较高的温度会使分布更加平滑
        probabilities = F.softmax(logits / temperature, dim=1)[0].cpu().numpy()
    
    # 收集结果
    results = []
    for i, genre in enumerate(GENRES):
        results.append({
            'genre': genre,
            'probability': float(probabilities[i]),
            'color': COLORS[genre]
        })
    
    # 按概率排序
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'predictions': results,
        'mel_spectrogram': mel_spectrogram_db.tolist(),
        'audio_data': y.tolist(),
        'sampling_rate': sr
    }

@app.route('/')
def index():
    return render_template('index.html', genres=GENRES, colors=COLORS)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        try:
            results = process_audio(file_path)
            return jsonify({
                'filename': file.filename,
                'results': results
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
