import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import accuracy_score
from tqdm import tqdm

# the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

def weight_init(m):
    if isinstance(m, nn.LazyLinear):
        pass
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Conv(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, p='same', act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel, stride, bias=False, padding=p)
        self.act = act()
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.bn(self.act(self.conv(x)))

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

# define the training function and validation function
def train_steps(loop, model, criterion, optimizer):
    train_loss, train_acc = [], []
    model.train()
    for step_index, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss.append(loss)
        pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(y, pred_result)
        train_acc.append(acc)
        loop.set_postfix(loss=loss, acc=acc)

    return {"loss": np.mean(train_loss), "acc": np.mean(train_acc), "lr": optimizer.param_groups[0]['lr']}


def test_steps(loop, model, criterion):
    test_loss, test_acc = [], []
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in loop:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y).item()

            test_loss.append(loss)
            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            acc = accuracy_score(y, pred_result)
            test_acc.append(acc)
            loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(test_loss), "acc": np.mean(test_acc)}

def train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer, scheduler):
    epochs = config['epochs']
    train_loss_ls, train_loss_acc, test_loss_ls, test_loss_acc, train_lr= [], [], [], [], []
    for epoch in range(epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        train_loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
        test_loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')

        train_metrix = train_steps(train_loop, model, criterion, optimizer)
        test_metrix = test_steps(test_loop, model, criterion)
        scheduler.step()

        train_loss_ls.append(train_metrix['loss'])
        train_loss_acc.append(train_metrix['acc'])
        test_loss_ls.append(test_metrix['loss'])
        test_loss_acc.append(test_metrix['acc'])
        train_lr.append(train_metrix['lr'])

        tqdm.write(f'Epoch {epoch + 1}: '
              f'train loss: {train_metrix["loss"]}; '
              f'train acc: {train_metrix["acc"]}; ')
        tqdm.write(f'Epoch {epoch + 1}: '
              f'test loss: {test_metrix["loss"]}; '
              f'test acc: {test_metrix["acc"]}')

        writer.add_scalar('train/loss', train_metrix['loss'], epoch)
        writer.add_scalar('train/accuracy', train_metrix['acc'], epoch)
        writer.add_scalar('validation/loss', test_metrix['loss'], epoch)
        writer.add_scalar('validation/accuracy', test_metrix['acc'], epoch)

    return {'train_loss': train_loss_ls, 'train_acc': train_loss_acc, 'test_loss': test_loss_ls, 'test_acc': test_loss_acc, 'train_lr': train_lr}

if __name__ == '__main__':
    model = Data_Model(10).to(device)
    model.apply(weight_init)
    model(torch.randn([2, 660000]).to(device))