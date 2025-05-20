import seaborn
import torch
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = ['device', 'GenreDataset', 'get_data_set', 'get_feature_set', 'load_data', 'plot_heat_map', 'plot_history']

# the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

genre_dict = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9,
}

def audio_augmentation(audio):
    # 随机缩放
    # scale_factor = 1 + torch.randn(1).item() * 0.1
    # augmented = audio * scale_factor
    
    # 随机高斯噪声
    noise = torch.randn_like(audio) * 0.01
    audio += noise
    
    return audio

# define the dataset class
class GenreDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = audio_augmentation(torch.tensor(self.x[index], dtype=torch.float32)).to(device)
        y = torch.tensor(self.y[index], dtype=torch.long).to(device)
        return x, y

    def __len__(self):
        return len(self.x)
    
def get_data_set(data_path, data_length):
    dataset, labelset = np.zeros((1, data_length)), []
    # dataset: the segment of music
    # labelset: convert blues/classical/country/disco/hiphop/jazz/metal/pop/reggae/rock to 0/1/2/3/4/5/6/7/8/9 in order
    for root, _, files in os.walk(data_path):
        for file in tqdm(files, desc=f'{os.path.basename(root).ljust(10)}'):
            genre = file.split('.')[0]
            try:
                data, sr = librosa.load(os.path.join(root, file))
                data = librosa.resample(data, orig_sr=sr, target_sr=sr)
                dataset = np.vstack((dataset, data[:data_length]))
                labelset.append(genre_dict[genre])
            except RuntimeError as e:
                pass
            except AttributeError as e:
                tqdm.write('There\'s something wrong in ' + file)        
    return dataset[1:,:], labelset

def get_feature_set(data_path):
    # dataset: features of the segment of music
    # labelset: convert blues/classical/country/disco/hiphop/jazz/metal/pop/reggae/rock to 0/1/2/3/4/5/6/7/8/9 in order
    with open(data_path, 'r') as f:
        content = f.read()
    content = content.split('\n')[1:-1]
    dataset, labelset = [], []
    for data in content:
        datas = data.split(',')
        dataset.append(list(map(float, datas[2:-1])))
        labelset.append(genre_dict[datas[-1]])
    return dataset, labelset

# load dataset and preprocess
def load_data(ratio, random_seed, data_path, data_length, type='feature'):
    dataset, labelset = get_feature_set(data_path) if type == 'feature' else get_data_set(data_path, data_length)
    # reshape the data and split the dataset
    dataset = np.array(dataset)
    labelset = np.array(labelset).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labelset, test_size=ratio, random_state=random_seed)
    return X_train, X_test, y_train, y_test

# confusion matrix
def plot_heat_map(y_test, y_pred):
    con_mat = confusion_matrix(y_test, y_pred)
    # normalize
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # plot
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    # plt.xlim(0, con_mat.shape[1])
    # plt.ylim(0, con_mat.shape[0])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_history(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history['train_acc'])
    plt.plot(history['test_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history['train_loss'])
    plt.plot(history['test_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(history['train_lr'])
    plt.title('Learning Rate')
    plt.ylabel('lr')
    plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('lr.png')
    plt.show()