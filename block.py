import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import device

__all__ = ['Conv', 'Bottleneck', 'PositionalEncoding', 'MLP']

class Conv(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, p='same', act=nn.SiLU):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv1d(cin, cout, kernel, stride, bias=False, padding=p),
            nn.BatchNorm1d(cout),
            act()
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.m(x)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, k, s=1, p='same', e=0.5, shortcut=False, act=nn.SiLU):
        super().__init__()
        if shortcut:
            assert c1 == c2
        
        self.shortcut = shortcut
        self.c = int(c2*e)
        self.cv1 = Conv(c1, self.c, k, s, p, act)
        self.cv2 = Conv(self.c, c2, k, s, p, act)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.cv2(self.cv1(x)) + x if self.shortcut else self.cv2(self.cv1(x))
    

class PositionalEncoding(nn.Module):
    """位置编码模块，支持更长的序列"""
    def __init__(self, d_model, dropout=0.1, max_len=200000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # 注册一个空的位置编码缓冲区
        self.register_buffer('pe', None)
        # 记录最大长度，用于动态创建位置编码
        self.max_registered_len = 0

    def _extend_pe(self, length):
        """动态扩展位置编码以覆盖更长的序列"""
        if self.max_registered_len >= length:
            return

        # 计算新的位置编码
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        
        # 创建新的位置编码矩阵
        pe = torch.zeros(length, self.d_model).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, length, d_model]
        
        # 更新缓冲区
        self.register_buffer('pe', pe)
        self.max_registered_len = length
        print(f"位置编码已扩展至长度: {length}")

    def forward(self, x):
        # x预期形状: [batch_size, seq_length, d_model]
        seq_length = x.size(1)
        
        # 如果需要，动态扩展位置编码
        if self.pe is None or self.max_registered_len < seq_length:
            self._extend_pe(max(seq_length, self.max_registered_len * 2))
            
        x = x + self.pe[:, :seq_length, :]
        return self.dropout(x)
    

class MLP(nn.Module):
    """多层感知机模块"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
    
