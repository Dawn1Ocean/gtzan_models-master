# gtzan_models

```Python
config = {
        'seed': 1337,       # the random seed
        'test_ratio': 0.2,  # the ratio of the test set
        'epochs': 150,
        'batch_size': 64,
        'lr': 0.0001437,    # initial learning rate
        'data_path': './Data/genres_original',
        'feature_path': './Data/features_30_sec.csv',
        'isDev': True,      # True -> Train new model anyway
        'isFeature': False, # True -> Trainset = features; False -> Trainset = Datas
        'data_length': 160000,  # If isFeature == False
    }
```

```Python
class HybridAudioClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridAudioClassifier, self).__init__()
        
        # 1D CNN特征提取分支
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4, padding=40),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 64, kernel_size=80, stride=4, padding=40),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=80, stride=4, padding=40),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(128, 256, kernel_size=80, stride=4, padding=40),
            nn.BatchNorm1d(256),
            nn.SiLU()
        )
        
        # BiLSTM分支
        self.lstm_branch = nn.Sequential(
            nn.LSTM(
                input_size=256,  # 与 CNN 输出通道数匹配
                hidden_size=128, 
                num_layers=2, 
                batch_first=True, 
                bidirectional=True
            )
        )
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=256,  # 与特征维度匹配 
            num_heads=4,
            dropout=0.3
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.LazyLinear(512),  # BiLSTM 输出是双向的
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        B, L = x.shape
        x = x.view(B, 1, L)
        # CNN特征提取
        cnn_features = self.cnn_branch(x)
        
        # 准备LSTM输入 
        # 从 [batch, channels, time] 转换到 [batch, time, channels]
        lstm_input = cnn_features.permute(0, 2, 1)
        
        # BiLSTM处理
        lstm_out, _ = self.lstm_branch(lstm_input)
        
        # 注意力机制
        # 转换维度以适应多头注意力
        attn_input = lstm_out.permute(1, 0, 2)  # [seq_len, batch, features]
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        
        # 池化和展平
        attn_output = attn_output.permute(1, 0, 2)
        pooled_output = torch.mean(attn_output, dim=1)
        
        # 分类
        output = self.classifier(pooled_output)
        
        return output
```