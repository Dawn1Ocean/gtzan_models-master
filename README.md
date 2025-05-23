# Installation

```bash
uv sync
```

# gtzan_models

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