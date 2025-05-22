import os
import datetime
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, ChainedScheduler

from nnmodels import (
    weight_init, 
    Data_Model, 
    Feature_Model, 
    HybridAudioClassifier, 
    TransformerEncoderDecoderClassifier, 
    CNNTransformerClassifier, 
    Mel_Model, 
    Mel_Attention_Model,
    YOLO11s)
from utils import load_data, plot_history, plot_heat_map, GenreDataset, device
from trainer import train_epochs

# project root path
project_path = "./"
# define log directory
# must be a subdirectory of the directory specified when starting the web application
# it is recommended to use the date time as the subdirectory name
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "genre_model.pt"

if __name__ == '__main__':
    config = {
        'model': "YOLO11s",
        'args': (10,),
        'seed': 1337,       # the random seed
        'test_ratio': 0.2,  # the ratio of the test set
        'epochs': 10,
        'batch_size': 32,
        'lr': 0.00001437,    # initial learning rate
        'data_path': './Data/genres_original',
        'feature_path': './Data/features_30_sec.csv',
        'isDev': True,      # True -> Train new model anyway
        'dataset': {
            'type': 'augMel',  # 'feature' -> Trainset = features; 'original' -> Trainset = Datas; 'mel', 'augMel' -> Trainset = melspectrogram
            'Mel': True,
            'Aug': True,
        },
        'data_length': 660000,  # If dataset != 'feature'
        'optimizer': torch.optim.AdamW,
        'scheduler': {
            'start_iters': 3,
            'start_factor': 1,
            'end_factor': 0.01,
        },
        'show': False, # plotting
    }

    model = globals()[config['model']](*config['args'])

    # X_train, y_train is the training set
    # X_test, y_test is the test set
    match config['dataset']['type']:
        case 'original'|'augMel':
            X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'], config['data_path'], config['data_length'], type='data')
        case 'feature':
            X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'], config['feature_path'], type='feature')
        case 'mel':
            X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'], config['data_path'], config['data_length'], type='mel')
        case _:
            raise NotImplementedError(f"Dataset type '{config['dataset']['type']}' is not implemented.")
    
    train_dataset = GenreDataset(X_train, y_train, mel=config['dataset']['Mel'], aug=config['dataset']['Aug'])
    test_dataset = GenreDataset(X_test, y_test, val=True, mel=config['dataset']['Mel'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model.apply(weight_init)
    if os.path.exists(model_path) and not config['isDev']:
        # import the pre-trained model if it exists
        print('Import the pre-trained model, skip the training process')
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'])
        scheduler = ChainedScheduler([
            LinearLR(optimizer, 
                    total_iters=config['scheduler']['start_iters']), 
            LinearLR(optimizer, 
                    start_factor=config['scheduler']['start_factor'], 
                    end_factor=config['scheduler']['end_factor'], 
                    total_iters=config['epochs'])]
            , optimizer=optimizer)

        # print the model structure if there is not any lazy layers in Net
        # summary(model, (config['batch_size'], X_train.shape[1]), col_names=["input_size", "kernel_size", "output_size"], verbose=2)

        # define the Tensorboard SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        # train and evaluate model
        history = train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer, scheduler, scaler=torch.GradScaler())
        writer.close()
        # save the model
        torch.save(model.state_dict(), model_path)
        # plot the training history
        plot_history(history, config['show'])

    # predict the class of test data
    y_pred, y_truth = [], []
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y_pred.extend(pred_result)
            y_truth.extend(y.detach().cpu().numpy())
    # plot confusion matrix heat map
    plot_heat_map(y_truth, y_pred, config['show'])