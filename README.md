# 音乐流派分类可视化WebApp

这是一个基于深度学习的音乐流派分类项目，使用卷积神经网络和注意力机制来分析音频并将其分类为10种不同的音乐流派。

## 功能特点

- 支持上传WAV音频文件并进行分析
- 自动提取音频中间30秒并转换为Mel频谱图
- 实时显示音频波形和频谱图
- 炫酷的流派预测可视化
    - 根据预测结果创建多彩光晕效果
    - 使用条形图显示各个流派的匹配度
    - 流派预测结果列表显示
- 支持多文件上传和管理
- 交互式音频播放控制

## 支持的音乐流派

- Blues（蓝调）- `#0033cc`
- Classical（古典）- `#e6b800`
- Country（乡村）- `#996633`
- Disco（迪斯科）- `#cc00cc`
- Hip-Hop（嘻哈）- `#ff3300`
- Jazz（爵士）- `#009999`
- Metal（金属）- `#333333`
- Pop（流行）- `#ff66b3`
- Reggae（雷鬼）- `#00cc00`
- Rock（摇滚）- `#cc0000`

## 安装与运行

1. 安装依赖：

    ```bash
    # 使用uv
    uv sync

    # 或使用传统的pip
    python -m venv venv
    source venv/bin/activate.fish  # fish shell
    pip install -r requirements.txt
    ```

2. 运行WebApp：

    对于Fish Shell：

    ```fish
    ./run.fish
    ```

    对于Bash Shell：

    ```bash
    ./run.sh
    ```

    或者直接运行：

    ```
    python app.py
    ```

3. 在浏览器中访问：`http://127.0.0.1:5000`

## 使用方法

1. 在网页中点击上传区域或拖放音频文件
2. 等待分析完成（模型将提取音频特征并进行分类）
3. 查看分析结果，包括波形图、频谱图和流派预测
4. 使用播放控制来听取上传的音频
5. 通过可视化了解音频属于哪种音乐流派

## 项目结构

- `app.py`: Flask应用程序入口
- `nnmodels.py`: 神经网络模型定义
- `utils.py`: 工具函数
- `templates/`: HTML模板
- `static/`: CSS、JavaScript和其他静态资源
- `uploads/`: 上传的音频文件
- `result/`: 训练好的模型

## 原始项目配置

```Python
config = {
        'log_dir': os.path.join(project_path, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        'result_path': os.path.join(project_path, "result"),
        'model': 'CNN_2D_Attention_Model',
        'args': (10,),
        'seed': 1337,        # the random seed
        'test_ratio': 0.2,   # the ratio of the test set
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.0001437,    # initial learning rate
        'isDev': True,       # True -> Train new model anyway
        'dataset': {
            'data_path': './Data/genres_original',
            'feature_path': './Data/features_30_sec.csv',
            'type': 'data',  # 'feature' -> Trainset = features; 'data' -> Trainset = Datas
            'Mel': True,     # Using Mel Spectrogram or not
            'Aug': False,     # Adding random noise before every epoch (May slow down the training) or not
            'data_length': 660000,  # If dataset != 'feature',
            'n_mels': 128,
        },
        'optimizer': torch.optim.AdamW,
        'scheduler': {
            'start_iters': 3,
            'start_factor': 1,
            'end_factor': 0.01,
        },
        'show': False,       # plotting
        'fold': 0,           # 0 -> not k-fold; k>0 -> k-fold
        'summary': False,    # Show summary
    }
```

## 模型结构

```Python
class CNN_2D_Attention_Model(nn.Module):
    def __init__(self, label_d, c=16, k=3, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(1, c, k, s, 1),
            Conv2d(c, 4*c, k, s, 1),
            Conv2d(4*c, 8*c, k, s, 1),
            nn.Conv2d(8*c, 8*c, k, s, 1),
            nn.BatchNorm2d(8*c),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # 注意力机制模块
        self.attention = nn.MultiheadAttention(
            embed_dim=8*c,
            num_heads=4,
            batch_first=True
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.LazyLinear(32*c),
                                 nn.SiLU(),
                                 nn.BatchNorm1d(32*c),)
        self.mlp = MLP(input_dim=32*c, hidden_dim=16*c, output_dim=label_d, dropout=0.6)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        # x.shape = (batch_size, input_d)
        B, H, W = x.shape
        x = x.view(B, 1, H, W)
        x = self.conv(x)
        conv_out = x.view(B, -1)  # [B,8c]
        
        # 注意力机制（添加维度适配）
        query = conv_out.unsqueeze(1)    # [B,1,8c]
        attn_out, _ = self.attention(query, query, query)
        attn_out = attn_out.squeeze(1)  # [B,8c]
        x = self.fc(attn_out)
        x = self.mlp(x)
        x = self.act(x)
        return x
```
