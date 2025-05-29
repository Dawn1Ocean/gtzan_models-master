import os
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from nnmodels import CNN_2D_Attention_Model
from utils import device, genre_dict

# 流派列表
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# 加载模型
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result/CNN_2D_Attention_Model.pt')
model = CNN_2D_Attention_Model(10).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # 确保模型处于评估模式

def process_audio_test(file_path):
    """测试版本的音频处理函数，严格按照训练时的处理方式"""
    # 加载音频
    y, sr = librosa.load(file_path, sr=22050)
    
    # 提取中间30秒 - 使用与训练时相同的数据长度 (660000 samples = 30s at 22050Hz)
    data_length = 660000
    if len(y) > data_length:
        mid_point = len(y) // 2
        mid_samp = data_length // 2
        y = y[mid_point-mid_samp:mid_point-mid_samp+data_length]
    
    # 确保音频长度一致
    if len(y) < data_length:
        padding = data_length - len(y)
        y = np.pad(y, (0, padding), 'constant')
    
    # 生成梅尔频谱图 - 使用与训练时相同的参数
    n_mels = 256
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram)
    
    # 转为张量并预测
    mel_tensor = torch.tensor(mel_spectrogram_db, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 获取最后一层softmax之前的原始输出（logits）
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
        temperature = 10.0  # 较高的温度会使分布更加平滑
        probabilities = F.softmax(logits / temperature, dim=1)[0].cpu().numpy()
    
    # 打印结果
    print(f"File: {os.path.basename(file_path)}")
    genre_probs = [(GENRES[i], float(probabilities[i])) for i in range(len(GENRES))]
    genre_probs.sort(key=lambda x: x[1], reverse=True)
    
    for genre, prob in genre_probs:
        print(f"{genre}: {prob*100:.2f}%")
    print()
    
    # 返回预测的流派（概率最高的）
    return GENRES[np.argmax(probabilities)]

# 测试几个不同流派的音频文件
test_files = [
    './Data/genres_original/blues/blues.00000.wav',
    './Data/genres_original/classical/classical.00000.wav',
    './Data/genres_original/country/country.00000.wav',
    './Data/genres_original/disco/disco.00000.wav',
    './Data/genres_original/hiphop/hiphop.00000.wav',
    './static/example_blues.wav',
    './static/example_classical.wav',
    './static/example_rock.wav',
]

for file_path in test_files:
    if os.path.exists(file_path):
        predicted_genre = process_audio_test(file_path)
        # 打印真实标签和预测标签
        true_genre = os.path.basename(file_path).split('.')[0]
        if true_genre not in GENRES:
            true_genre = os.path.basename(os.path.dirname(file_path))
        print(f"True: {true_genre}, Predicted: {predicted_genre}")
        print("=" * 50)
    else:
        print(f"File not found: {file_path}")
