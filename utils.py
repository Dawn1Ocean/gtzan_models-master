import torch
import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = ('device', 'GenreDataset', 'get_data_set', 'get_feature_set', 'load_data')

# the device to use
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
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
    # scale_factor = np.random.rand(0.5, 1.5)
    # audio = audio * scale_factor
    
    # 随机高斯噪声
    noise = np.random.randn(*audio.shape) * 0.01
    audio += noise
    
    return audio

# define the dataset class
class GenreDataset(Dataset):
    def __init__(self, x, y, val=False, mel=False, aug=False, sr=22050, n_mels=512):
        assert len(x) == len(y), "The length of x and y must be the same"
        assert len(x) > 0, "The length of x must be greater than 0"
        assert (val is True and aug is True) is False, "val and aug cannot be True at the same time"

        self.x, self.y = x, y
        self.val, self.mel, self.aug = val, mel, aug
        self.sr, self.n_mels = sr, n_mels

        if self.mel is True and self.aug is False:
            self.x = np.array([self._mel(x) for x in self.x])

    def __getitem__(self, index):
        x = self.x[index]
        if self.val is False and self.aug is True:
            x = audio_augmentation(x)
            if self.mel is True:
                x = self._mel(x)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(self.y[index], dtype=torch.long).to(device)
        return x, y

    def __len__(self):
        return len(self.x)
    
    def _mel(self, x):
        return librosa.amplitude_to_db(librosa.feature.melspectrogram(y=x, sr=self.sr, n_mels=self.n_mels))
    
def get_data_set(data_path, data_length):
    dataset, labelset = [], []
    # dataset: the segment of music
    # labelset: convert blues/classical/country/disco/hiphop/jazz/metal/pop/reggae/rock to 0/1/2/3/4/5/6/7/8/9 in order
    for root, _, files in os.walk(data_path):
        for file in tqdm(files, desc=f'{os.path.basename(root).ljust(10)}'):
            genre = file.split('.')[0]
            try:
                data, sr = librosa.load(os.path.join(root, file))
                mid_data, mid_samp = len(data) // 2, data_length // 2
                dataset.append(data[mid_data-mid_samp:mid_data-mid_samp+data_length])
                labelset.append(genre_dict[genre])
            except RuntimeError as e:
                pass
            except AttributeError as e:
                tqdm.write('There\'s something wrong in ' + file)
    return dataset, labelset

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
def load_data(ratio, random_seed, data_path, data_length=None, type='feature'):
    match type:
        case 'feature':
            dataset, labelset = get_feature_set(data_path)
        case 'data':
            dataset, labelset = get_data_set(data_path, data_length)
        case _:
            raise NotImplementedError(f"Unknown type: {type}")
    # reshape the data and split the dataset
    dataset = np.array(dataset)
    labelset = np.array(labelset).reshape(-1)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labelset, test_size=ratio, random_state=random_seed)
    return X_train, X_test, y_train, y_test