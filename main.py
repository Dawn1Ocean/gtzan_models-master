import datetime
import torch
import os

from torch.utils.data import DataLoader
from torchinfo import summary

from nnmodels import (
    weight_init,
    CNN_1D_Model,
    MLP_Model,
    CNN_1D_BiLSTM_Attention_Model, 
    Transformer_Encoder_Decoder_Model, 
    CNN_1D_Transformer_Model, 
    CNN_2D_Model,
    CNN_2D_Attention_Model,
    YOLO11s)
from utils import load_data, GenreDataset, device, getKfoldDataloader
from trainer import training, testing, kFoldVal

# project root path
project_path = "."

if __name__ == '__main__':
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

    if not os.path.exists(config['result_path']):
        os.makedirs(config['result_path'])
    model_path = os.path.join(project_path, "result", config['model'] + ".pt")
    dataset_config = config['dataset']

    if config['fold']:
        config['test_ratio'] = 0.0  # Reading all of the data

    # X_train, y_train is the training set
    # X_test, y_test is the test set
    X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'], dataset_config) # type: ignore
    
    if config['fold']:
        acc = kFoldVal([globals()[config['model']](*config['args']).to(device) for _ in range(config['fold'])], config, getKfoldDataloader(X_train, y_train, config, k=config['fold']))
        print(f'{config['fold']}-Fold Validation Accuracy: {acc}')
    else:
        model = globals()[config['model']](*config['args']).to(device)

        train_dataset = GenreDataset(X_train, y_train, mel=dataset_config['Mel'], aug=dataset_config['Aug'], n_mels=dataset_config['n_mels'])
        test_dataset = GenreDataset(X_test, y_test, val=True, mel=dataset_config['Mel'], n_mels=dataset_config['n_mels'])

        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        dataloaders = (train_dataloader, test_dataloader)
        model.apply(weight_init)
        if os.path.exists(model_path) and not config['isDev']:
            # import the pre-trained model if it exists
            print('Import the pre-trained model, skip the training process')
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            training(model, config, dataloaders, model_path, config['model'], config['result_path'])
            if config['summary']:
                summary(model, (config['batch_size'], *X_train.shape[1:]), col_names=["input_size", "kernel_size", "output_size"], verbose=2)
        testing(model, config, test_dataloader, config['model'], config['result_path'])