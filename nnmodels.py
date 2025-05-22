import torch
import torch.nn as nn
import torch.nn.functional as F

from block import Conv, Conv2d, Bottleneck, PositionalEncoding, MLP, C2f, CBS2d, SPPF, C2PSA

from utils import device

__all__ = ['weight_init', 'Data_Model', 'Feature_Model', 'HybridAudioClassifier',
           'TransformerEncoderDecoderClassifier', 'CNNTransformerClassifier', 'Mel_Model',
           'Mel_Attention_Model', 'YOLO11s']

def weight_init(m):
    if isinstance(m, nn.LazyLinear):
        pass
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# build the CNN model
class Data_Model(nn.Module):
    def __init__(self, label_d, c=16, k=3, kb=5):
        super().__init__()
        self.conv1 = Conv(1, c, 11, 4, 1)
        self.conv2 = nn.Sequential(Conv(c, 4*c, k, 4, 1), 
                                   Bottleneck(4*c, 4*c, kb, shortcut=True))
        self.conv3 = nn.Sequential(Conv(4*c, 8*c, k, 2, 1), 
                                   Bottleneck(8*c, 8*c, kb, shortcut=True))
        self.conv4 = nn.Sequential(Conv(8*c, 8*c, k), 
                                   Bottleneck(8*c, 8*c, kb, shortcut=True))
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(nn.LazyLinear(8*c), 
                                 nn.BatchNorm1d(8*c), 
                                 nn.SiLU())
        # self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Sequential(nn.Linear(8*c, 4*c), 
                                 nn.BatchNorm1d(4*c), 
                                 nn.SiLU())
        # self.dropout2 = nn.Dropout(0.2)
        # self.ssm = nn.LSTM(int(2*c), int(8*c), int(4*c), bias=False, dropout=0.2, bidirectional=True)
        self.fc3 = nn.Linear(4*c, label_d)
        self.act = nn.Softmax(dim=1)

    def forward(self, x:torch.Tensor):
        # x.shape = (batch_size, input_d)
        # reshape the tensor with shape (batch_size, input_d) to (batch_size, 1, input_d)
        B, L = x.shape
        x = x.view(B, 1, L)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)
        # x = x.view(256, 32, -1)
        
        # lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, 100)
        
        # lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc1(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        # x = self.dropout2(x)
        # x, _ = self.ssm(x)
        x = self.act(self.fc3(x))
        return x
    
# build the ANN model
class Feature_Model(nn.Module):
    def __init__(self, input_d, label_d):
        super().__init__()
        self.fc1 = nn.Linear(input_d, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, label_d)

    def forward(self, x):
        # x.shape = (batch_size, input_d)
        x = F.silu(self.fc1(x))
        x = self.bn1(x)
        x = F.silu(self.fc2(x))
        x = self.bn2(x)
        x = F.silu(self.fc3(x))
        x = self.bn3(x)
        x = F.silu(self.fc4(x))
        x = self.bn4(x)
        x = F.silu(self.fc5(x))
        x = self.bn5(x)
        x = F.softmax(self.fc6(x), dim=1)
        return x
    
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

class TransformerEncoderDecoderClassifier(nn.Module):
    """使用编码器-解码器结构的Transformer用于序列分类，支持长序列"""
    def __init__(self, input_dim, num_classes=10, d_model=64, nhead=4, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_seq_length=32,
                 use_segment_pooling=True, segment_length=2048):
        super(TransformerEncoderDecoderClassifier, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.use_segment_pooling = use_segment_pooling
        self.segment_length = segment_length
        
        # 输入特征嵌入
        self.emb = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # Transformer解码器
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, 
            num_layers=num_decoder_layers
        )
        
        # 分类标记嵌入（作为解码器的查询）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # MLP分类头
        self.mlp_classifier = MLP(
            input_dim=d_model,
            hidden_dim=dim_feedforward,
            output_dim=num_classes,
            dropout=dropout
        )

        self.act = nn.Softmax(dim=1)
        
    def _input_embedding(self, src:torch.Tensor)->torch.Tensor:
        """输入嵌入：将输入序列嵌入到d_model维度"""
        return self.emb(src)

    def _process_long_sequence(self, src:torch.Tensor)->torch.Tensor:
        """处理长序列：将其分段，分别编码，然后汇集结果"""
        batch_size, seq_length, _ = src.shape
        segment_length = self.segment_length
        
        # 计算需要的段数
        num_segments = (seq_length + segment_length - 1) // segment_length
        
        # 存储每个段的编码输出
        segment_outputs = []
        
        # 逐段处理
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, seq_length)
            
            # 提取当前段
            segment = src[:, start_idx:end_idx, :]
            
            # 编码当前段
            segment_embedded = self._input_embedding(segment)
            segment_encoded = self.pos_encoder(segment_embedded)
            segment_memory = self.transformer_encoder(segment_encoded)
            
            # 对当前段进行池化（平均池化）
            segment_pooled = torch.mean(segment_memory, dim=1, keepdim=True)  # [batch_size, 1, d_model]
            segment_outputs.append(segment_pooled)
        
        # 连接所有段的池化结果
        pooled_memory = torch.cat(segment_outputs, dim=1)  # [batch_size, num_segments, d_model]
        
        # 如果段数超过Transformer的处理能力，可能需要再次池化
        if num_segments > 500:  # 设置一个安全阈值
            print(f"警告：序列过长，分割为 {num_segments} 段后仍需进一步池化。考虑增加segment_length值。")
            # 进一步池化，例如每10个段池化为1个
            pooling_factor = (num_segments + 499) // 500
            reduced_segments = []
            for i in range(0, num_segments, pooling_factor):
                end = min(i + pooling_factor, num_segments)
                reduced_segment = torch.mean(pooled_memory[:, i:end, :], dim=1, keepdim=True)
                reduced_segments.append(reduced_segment)
            pooled_memory = torch.cat(reduced_segments, dim=1)
        
        return pooled_memory
        
    def forward(self, src:torch.Tensor, src_mask=None, src_key_padding_mask=None)->torch.Tensor:
        """
        前向传播
        
        参数:
        - src: 输入序列 [batch_size, seq_length]
        - src_mask: 源序列的注意力掩码
        - src_key_padding_mask: 源序列的填充掩码
        
        返回:
        - logits: 分类logits [batch_size, num_classes]
        """
        batch_size, seq_length = src.shape
        src = src.view(batch_size, seq_length, 1)
        
        # 对于超长序列，使用分段处理
        if self.use_segment_pooling and seq_length > self.segment_length:
            memory = self._process_long_sequence(src)
        else:
            # 1. 输入嵌入
            src = self._input_embedding(src)  # [batch_size, seq_length, d_model]
            
            # 2. 位置编码
            src = self.pos_encoder(src)  # [batch_size, seq_length, d_model]
            
            # 3. Transformer编码器
            memory = self.transformer_encoder(
                src, 
                mask=src_mask, 
                src_key_padding_mask=src_key_padding_mask
            )  # [batch_size, seq_length, d_model]
        
        # 4. 准备解码器输入（分类标记）
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)  # [batch_size, 1, d_model]
        cls_tokens = self.pos_encoder(cls_tokens)  # 添加位置编码
        
        # 5. Transformer解码器
        # 解码器在训练和推理时只使用一个标记，没有掩码
        decoder_output = self.transformer_decoder(
            cls_tokens,  # 目标序列（查询）
            memory,      # 编码器输出（键和值）
            tgt_mask=None, 
            memory_mask=None
        )  # [batch_size, 1, d_model]
        
        # 6. 提取解码器输出的分类标记表示
        cls_representation = decoder_output.squeeze(1)  # [batch_size, d_model]
        
        # 7. 通过MLP分类头
        logits = self.mlp_classifier(cls_representation)  # [batch_size, num_classes]
        
        return self.act(logits)


class CNNTransformerClassifier(TransformerEncoderDecoderClassifier):
    """结合CNN和Transformer的分类器"""
    def __init__(self, input_dim, num_classes=10, d_model=128, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=512, dropout=0.2, max_seq_length=128,
                 use_segment_pooling=True, segment_length=2048):
        super(CNNTransformerClassifier, self).__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length,
            use_segment_pooling=use_segment_pooling,
            segment_length=segment_length
        )
        
        # CNN特征提取分支
        self.emb = nn.Sequential(
            Conv(1, 16, 37, 4, 1),
            Conv(16, 64, 37, 4, 1),
            Conv(64, d_model, 37, 1)
        )
    
    def _input_embedding(self, src:torch.Tensor)->torch.Tensor:
        src = src.permute(0, 2, 1).contiguous() # [batch_size, d_model, seq_length]
        src = self.emb(src)
        src = src.permute(0, 2, 1).contiguous() # [batch_size, seq_length, d_model]
        return src

# build the 2D-CNN model
class Mel_Model(nn.Module):
    def __init__(self, label_d, c=16, k=3, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(1, c, k, s, 1),
            Conv2d(c, 4*c, k, s, 1),
            Conv2d(4*c, 8*c, k, s, 1),
            Conv2d(8*c, 8*c, k, s, 1),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.LazyLinear(32*c),
                                 nn.SiLU(),
                                 nn.BatchNorm1d(32*c),)
        self.mlp = MLP(32*c, 16*c, label_d, dropout=0.75)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        # x.shape = (batch_size, input_d)
        B, H, W = x.shape
        x = x.view(B, 1, H, W)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.mlp(x)
        x = self.act(x)
        return x
    
class Mel_Attention_Model(nn.Module):
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
        self.mlp = MLP(32*c, 16*c, label_d, dropout=0.6)
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

class YOLO11s(nn.Module):
    def __init__(self, num_classes:int=10, d:int=256, n=2):
        super().__init__()
        self.b1 = CBS2d(1, d//16, 3, 2, 1)
        self.b2 = CBS2d(d//16, d//8, 3, 2, 1)
        self.b3 = nn.Sequential(
            CBS2d(d//8, d//4, 3, 2, 1),
            C2f(d//4, d//4, shortcut=True)
        )
        self.b4 = nn.Sequential(
            CBS2d(d//4, d//2, 3, 2, 1),
            C2f(d//2, d//2, n, shortcut=True)
        )
        self.b5 = nn.Sequential(
            CBS2d(d//2, d, 3, 2, 1),
            C2f(d, d, n, shortcut=True),
            SPPF(d, d),
            C2PSA(d, d, 2),
        )
        self.fpn = nn.ModuleList([
            C2f(d+d//2, d//2, n),
            C2f(d//2+d//4, d//4, n),
            CBS2d(d//4, d//4, 3, 2, 1),
            C2f(d//2+d//4, d//2, n),
            CBS2d(d//2, d//2, 3, 2, 1),
            C2f(d+d//2, d, n),])
        self.head = nn.ModuleList(
            [nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(d//2),
            ) for _ in range(3)]+
            [MLP(3*d//2, d, num_classes)]
        )
        self.logits = nn.Softmax(dim=1)

    def _fpn(self, b3:torch.Tensor, b4:torch.Tensor, b5:torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b5u = F.interpolate(b5, size=b4.shape[2:], mode='bilinear') # d
        p4 = torch.cat([b4, b5u], dim=1) # d+d//2
        p4 = self.fpn[0](p4) # d//2
        p4u = F.interpolate(p4, size=b3.shape[2:], mode='bilinear') # d//2
        p3 = torch.cat([b3, p4u], dim=1) # d//4 + d//2
        p3 = self.fpn[1](p3) # d//4
        p4n = self.fpn[2](p3) # d//4
        p4 = torch.cat([p4, p4n], dim=1) # d//2 + d//4
        p4 = self.fpn[3](p4) # d//2
        p5n = self.fpn[4](p4) # d//2
        p5 = torch.cat([b5, p5n], dim=1) # d + d//2
        p5 = self.fpn[5](p5) # d
        return p3, p4, p5

    def _head(self, p3:torch.Tensor, p4:torch.Tensor, p5:torch.Tensor)->torch.Tensor:
        y = torch.cat([self.head[0](p3), self.head[1](p4), self.head[2](p5)], dim=1)
        return self.head[3](y)

    def forward(self, x:torch.Tensor):
        # print(x.shape)
        # raise KeyboardInterrupt
        B, H, W = x.shape
        x = x.view(B, 1, H, W)
        # x = F.interpolate(x, (640, 640))
        b1 = self.b1(x)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        
        p3, p4, p5 = self._fpn(b3, b4, b5)
        out = self._head(p3, p4, p5)
        return self.logits(out)

if __name__ == '__main__':
    # model = CNNTransformerClassifier(1, 10).to(device)
    model = YOLO11s(10).to(device)
    model.apply(weight_init)
    model(torch.randn([128, 512, 1290]).to(device))