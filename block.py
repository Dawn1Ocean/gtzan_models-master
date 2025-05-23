import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import device

__all__ = ['Conv', 'Bottleneck', 'PositionalEncoding', 'MLP', 'Conv2d', 'CBS2d', 'SPPF', 'C2PSA']

class Conv(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, p:int|str='same', act:bool|type[nn.Module]=True):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel, stride, bias=False, padding=p)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.Identity() if act is False else (nn.SiLU() if act is True else act())

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

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

class Bottleneck2d(Bottleneck):
    def __init__(self, c1, c2, k, s=1, p='same', e=0.5, shortcut=False, act=nn.SiLU):
        super().__init__(c1, c2, k, s, p, e, shortcut, act)
        self.cv1 = CBS2d(c1, self.c, k, s, p, act=act)
        self.cv2 = CBS2d(self.c, c2, k, s, p, act=act)

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
            
        x = x + self.pe[:, :seq_length, :] # type: ignore
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
    
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, p:int|str='same', act=nn.SiLU):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(cin, cout, kernel, stride, bias=False, padding=p),
            act(),
            nn.MaxPool2d(kernel, 2),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.m(x)
    
class CBS2d(Conv):
    def __init__(self, c1:int, c2:int, k=1, s=1, p:int|str='same', d=1, g=1, act:bool|type[nn.Module]=True):
        super().__init__(c1, c2, k, s, p, act)
        self.conv = nn.Conv2d(c1, c2, k, s, p, d, g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

class C2f(nn.Module):
    def __init__(self, c1:int, c2:int, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2*e)
        self.cv1 = CBS2d(c1, 2 * self.c, 1)
        self.cv2 = CBS2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck2d(self.c, self.c, k=3, e=1.0, shortcut=shortcut) for _ in range(n))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBS2d(c1, c_, 1, 1)
        self.cv2 = CBS2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = CBS2d(dim, h, 1, act=False)
        self.proj = CBS2d(dim, dim, 1, act=False)
        self.pe = CBS2d(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(CBS2d(c, c * 2, 1), CBS2d(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = CBS2d(c1, 2 * self.c, 1, 1)
        self.cv2 = CBS2d(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))