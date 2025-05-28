import torch
import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader
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

def audio_augmentation(audio, noise_factor):
    # 随机缩放
    # scale_factor = np.random.rand(0.5, 1.5)
    # audio = audio * scale_factor
    
    # 随机高斯噪声
    noise = np.random.randn(*audio.shape) * noise_factor
    audio += noise
    
    return audio

# define the dataset class
class GenreDataset(Dataset):
    def __init__(self, x, y, val=False, mel=False, aug=False, noise_factor=0.01, sr=22050, n_mels=512):
        assert len(x) == len(y), "The length of x and y must be the same"
        assert len(x) > 0, "The length of x must be greater than 0"

        self.x, self.y = x, y
        self.val, self.mel, self.aug = val, mel, aug
        self.noise_factor, self.sr, self.n_mels = noise_factor, sr, n_mels

    def __getitem__(self, index):
        x = self.x[index]
        if self.aug:
            if not self.val:
                x = audio_augmentation(x, self.noise_factor)
            if self.mel:
                x = self._mel(x)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(self.y[index], dtype=torch.long).to(device)
        return x, y

    def __len__(self):
        return len(self.x)
    
    def _mel(self, x):
        return librosa.amplitude_to_db(librosa.feature.melspectrogram(y=x, sr=self.sr, n_mels=self.n_mels))
    
def get_data_set(dataset_config):
    dataset, labelset = [], []
    # dataset: the segment of music
    # labelset: convert blues/classical/country/disco/hiphop/jazz/metal/pop/reggae/rock to 0/1/2/3/4/5/6/7/8/9 in order
    for root, _, files in os.walk(dataset_config['data_path']):
        for file in tqdm(files, desc=f'{os.path.basename(root).ljust(10)}'):
            genre = file.split('.')[0]
            try:
                data, sr = librosa.load(os.path.join(root, file))
                mid_data, mid_samp = len(data) // 2, dataset_config['data_length'] // 2
                data = data[mid_data-mid_samp:mid_data-mid_samp+dataset_config['data_length']]
                if dataset_config['Mel'] and not dataset_config['Aug']:
                    data = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=dataset_config['n_mels'])
                    data = librosa.amplitude_to_db(data)
                dataset.append(data)
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
def load_data(ratio, random_seed, dataset_config):
    match dataset_config['type']:
        case 'feature':
            dataset, labelset = get_feature_set(dataset_config['feature_path'])
        case 'data':
            dataset, labelset = get_data_set(dataset_config)
        case _:
            raise NotImplementedError(f"Unknown type: {type}")
    # reshape the data and split the dataset
    dataset = np.array(dataset)
    labelset = np.array(labelset).reshape(-1)
    if ratio:
        X_train, X_test, y_train, y_test = train_test_split(dataset, labelset, test_size=ratio, random_state=random_seed)
        return X_train, X_test, y_train, y_test
    return dataset, [], labelset, []

def getKfoldDataloader(datas:np.ndarray, labels:np.ndarray, config:dict, k:int=5)->list[tuple[DataLoader, DataLoader]]:
    kfold = KFold(n_splits=k, shuffle=True)

    dataloaders = []
    for train_index, test_index in tqdm(kfold.split(datas), desc=f'{k}-fold dataloader'):
        dataloaders.append((DataLoader(GenreDataset(
                x=datas[train_index],
                y=labels[train_index],
                mel=config['dataset']['Mel'],
                aug=config['dataset']['Aug'],
            ),
            batch_size=config['batch_size'], shuffle=True),
            DataLoader(GenreDataset(
                x=datas[test_index],
                y=labels[test_index],
                mel=config['dataset']['Mel'],
                aug=config['dataset']['Aug'],
            ),
            batch_size=config['batch_size'], shuffle=False)))
        
    return dataloaders