import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils import device

__all__ = ['train_epochs', 'amp_train_epochs']

# define the training function and validation function
def train_steps(loop, model, criterion, optimizer, scaler:torch.GradScaler=None):
    train_loss, train_acc = [], []
    model.train()
    for step_index, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X)
            loss = criterion(pred, y)
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

def amp_train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer, scheduler):
    epochs = config['epochs']
    train_loss_ls, train_loss_acc, test_loss_ls, test_loss_acc, train_lr= [], [], [], [], []
    scaler = torch.GradScaler()
    for epoch in range(epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        train_loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
        test_loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')

        train_metrix = train_steps(train_loop, model, criterion, optimizer, scaler)
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