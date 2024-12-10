#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Training of the modelB using the scale invariance assumption. This training 
consists in only minimizing the reconstruction loss since the perceptual loss 
has no purpose following this approach.

Else
----
@author: Romuald Ait Bachir
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.backends
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import os
from typing import Dict, Tuple
import pickle
import sys
from shutil import copy

from model import ModelB_2
from dataset import ModisDatasetB_scale_invariance
import utils as us

from argparse import ArgumentParser


def train_step(model: torch.nn.Module, 
               train_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim,
               loss_fn: torch.nn.Module,
               device: str):
    """
    Description
    -----------
    Function doing the training process (Forward pass + Backpropagation) for a 
    single epoch for the modelB.
    
    Parameters
    ----------
    model : torch.nn.Module
        model in training (modelB in this particular case).
    pretrained_model : torch.nn.Module
        model used to compare the gradients (pretrained model / modelA).
    train_dataloader : torch.utils.data.DataLoader
        Batchified training dataset.
    optimizer : torch.optim
        Optimizer used to minimize the loss function.
    loss_fn : torch.nn.Module
        Loss function.
    alpha : float
        MixedGradientLoss hyperparameter
    lambda_loss : float
        Hyperparameter used in the perceptual loss function. Note: When giving a single 
        value to all the features, it's simply equal to a mean.
    device : str
        Device on which to perform the training.

    Returns
    -------
    loss_train : float
        Measurement of the mean loss over the current epoch.
    psnr_train : float
        Measurement of the mean PSNR over the current epoch.
    ssim_train : float
        Measurement of the mean SSIM over the current epoch.

    """
    model.train()

    loss_train = 0
    psnr_train = 0
    ssim_train = 0
    
    pbar = tqdm(train_dataloader, position = 1, desc = "Training Step", unit = " Batch", leave = False)
        
    for _, data in enumerate(pbar): 

        # FORWARD PASS
        lst_4km_up, ndvi, lst_1km = data[0].to(device), data[1].to(device), data[2].to(device) 
        
        optimizer.zero_grad()
        
        # For modelB_2
        lst_ndvi = torch.cat((lst_4km_up, ndvi), dim = 1)
        
        lst_SR = model(lst_ndvi)
        
        ### Final loss
        loss = loss_fn(lst_SR, lst_1km)

        loss.backward()
        
        optimizer.step()
        
        loss_train += loss.item()
        psnr_train += us.psnr_skimage(lst_SR.detach().cpu().numpy(), lst_1km.detach().cpu().numpy())
        ssim_train += us.ssim_skimage(lst_SR.detach().cpu().numpy(), lst_1km.detach().cpu().numpy())
        
        pbar.set_description('Loss over the batch: loss: {}'.format(np.format_float_scientific(loss.item(), precision = 2)))
    
    # Computing the mean of the metrics over the whole epoch.
    loss_train /= len(train_dataloader)
    psnr_train /= len(train_dataloader)
    ssim_train /= len(train_dataloader)

    return loss_train, psnr_train, ssim_train


@torch.inference_mode()
def test_step(model: torch.nn.Module, 
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str):
    """
    Description
    -----------
    Function performing the validation/testing step to evaluate the performances
    of the modelB on unseen data.
    
    Parameters
    ----------
    model : torch.nn.Module
        model in testing (modelB in this particular case).
    pretrained_model : torch.nn.Module
        model used to compare the gradients (pretrained model / modelA).
    test_dataloader : torch.utils.data.DataLoader
        Batchified test dataset.
    loss_fn : torch.nn.Module
        Loss function.
    alpha : float
        MixedGradientLoss hyperparameter
    lambda_loss : float
        Hyperparameter used in the perceptual loss function. Note: When giving a single 
        value to all the features, it's simply equal to a mean.
    device : str
        Device on which the model is train/tested.

    Returns
    -------
    loss_test : float
        Measurement of the mean loss over the current epoch.
    psnr_test : float
        Measurement of the mean PSNR over the current epoch.
    ssim_test : float
        Measurement of the mean SSIM over the current epoch.

    """
    model.eval()
    
    loss_test = 0
    psnr_test = 0
    ssim_test = 0
    
    pbar = tqdm(test_dataloader, position = 1, desc = "Testing Step", unit = " Batch", leave = False)
    
    for _, data in enumerate(pbar): 

        # FORWARD PASS
        lst_4km_up, ndvi, lst_1km = data[0].to(device), data[1].to(device), data[2].to(device) 
        
        optimizer.zero_grad()
        
        # For modelB_2
        lst_ndvi = torch.cat((lst_4km_up, ndvi), dim = 1)
        
        lst_SR = model(lst_ndvi)
        
        ### Final loss
        loss = loss_fn(lst_SR, lst_1km)
        
        loss_test += loss.item()
        psnr_test += us.psnr_skimage(lst_SR.detach().cpu().numpy(), lst_1km.detach().cpu().numpy())
        ssim_test += us.ssim_skimage(lst_SR.detach().cpu().numpy(), lst_1km.detach().cpu().numpy())
        
        pbar.set_description('Loss over the batch: loss: {}'.format(np.format_float_scientific(loss.item(), precision = 2)))
    
    # Computing the mean of the metrics over the whole epoch.
    loss_test /= len(test_dataloader)
    psnr_test /= len(test_dataloader)
    ssim_test /= len(test_dataloader)

    return loss_test, psnr_test, ssim_test


def train(model: torch.nn.Module,
          train_dataset: torch.utils.data.Dataset,
          val_dataset: torch.utils.data.Dataset,
          n_Epochs: int,
          batch_size: int,
          learning_rate: float,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim,
          device: str,
          checkpoint: type(us.model_checkpoint)) -> Tuple[torch.nn.Module, Dict]:
    """
    Description
    -----------
    Function implementing the training loop.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to train (modelB).
    pretrained_model : torch.nn.Module
        Model used to compare the gradients (modelA).
    train_dataset : torch.utils.data.Dataset
        Training dataset.
    val_dataset : torch.utils.data.Dataset
        Validation dataset.
    n_Epochs : int
        Number of times the model sees the whole dataset.
    batch_size : int
        Number of image inside a mini-batch.
    learning_rate : float
        Step size taken in the opposite direction of the gradient.
    loss_fn : torch.nn.Module
        Loss function to minimize.
    optimizer : torch.optim
        Optimization algorithm used to reach the minimum of the loss.
    alpha : float
        Hyperparameter used in the perceptual loss function.
    lambda_loss : float
        hyperparameter used in the perceptual loss function.
    device : str
        Device on which to perform the training.
    checkpoint : type(us.model_checkpoint)
        Implementation of the early stopping to avoid overfitting.

    Returns
    -------
    model : torch.nn.Module
        Model trained (modelB).
    metrics : dict
        Dictionnary containing all the measurements made during the training/testing.
    """
    
    # Preparing the data.
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size= batch_size, shuffle = True)
    
    # Preparing the dictionnary containing the metrics.
    metrics = {}
    metrics['train_loss'] = []
    metrics['train_psnr'] = []
    metrics['train_ssim'] = []
    metrics['val_loss'] = []
    metrics['val_psnr'] = []
    metrics['val_ssim'] = []
        
    for i in tqdm(range(1,n_Epochs+1,1), position = 0, desc = "Training", unit = " Epochs"):
            
            tl, tp, ts = train_step(model, 
                           train_dataloader,
                           optimizer,
                           loss_fn,
                           device)
            metrics['train_loss'].append(tl)
            metrics['train_psnr'].append(tp)
            metrics['train_ssim'].append(ts)
            
            tl, tp, ts = test_step(model, 
                           val_dataloader,
                           loss_fn,
                           device)
            metrics['val_loss'].append(tl)
            metrics['val_psnr'].append(tp)
            metrics['val_ssim'].append(ts)

            checkpoint.test_update(model, metrics, 'val_loss', i)
            
            # This part should probably be reviewed since in the end, if there is no 
            # early stopping, we'll save the last model...
            if checkpoint.train_state == 'continue' and i == n_Epochs:
                
                metrics['best_epoch'] = n_Epochs
            
            if checkpoint.train_state == 'break':
                
                metrics['best_epoch'], state_dict = checkpoint.best_epoch, checkpoint.saved_state
            
                model.load_state_dict(state_dict)
                
                break
            
    return model, metrics


def plot_loss(metrics_dict, savepath = './results/modelB', modelname = 'modelB'):
    """
    Description
    -----------
    Function plotting the metrics stored inside the variable metrics
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionnary containing the training/testing metrics.
    savepath : string, optional
        Path where to save the figures. The default is './results/modelA'.
    modelname : string, optional
        Name used to title the figures. The default is 'modelB'.
    """
    
    long = len(metrics_dict['train_loss'])
    
    plt.figure(1, figsize = (10,7))
    l1, = plt.plot(range(long), metrics_dict['train_loss'], color = 'blue')
    l2, = plt.plot(range(long), metrics_dict['val_loss'], color = 'orange')
    plt.legend(loc ='upper right', handles = [l1, l2], labels = ['Train Loss', 'Test Loss'])
    plt.title("loss = f(epoch)")
    plt.xlabel('epoch')
    plt.ylabel('loss_value')
    plt.savefig(os.path.join(savepath, modelname+'_loss.png'))

    plt.figure(2, figsize = (10,7))
    l1, = plt.plot(range(long), metrics_dict['train_psnr'], color = 'blue')
    l2, = plt.plot(range(long), metrics_dict['val_psnr'], color = 'orange')
    plt.legend(loc ='upper right', handles = [l1, l2], labels = ['Train PSNR', 'Test PSNR'])
    plt.title("PSNR = f(epoch)")
    plt.xlabel('epoch')
    plt.ylabel('PSNR value')
    plt.savefig(os.path.join(savepath, modelname+'_psnr.png'))
    
    plt.figure(3, figsize = (10,7))
    l1, = plt.plot(range(long), metrics_dict['train_ssim'], color = 'blue')
    l2, = plt.plot(range(long), metrics_dict['val_ssim'], color = 'orange')
    plt.legend(loc ='upper right', handles = [l1, l2], labels = ['Train SSIM', 'Test SSIM'])
    plt.title("SSIM = f(epoch)")
    plt.xlabel('epoch')
    plt.ylabel('SSIM value')
    plt.savefig(os.path.join(savepath, modelname+'_ssim.png'))
    

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, default="./paramsB.json")
    args = parser.parse_args()
    
    file = args.params 
    
    # Reading the parameters.
    dataset_parameter, _, modelB_parameters, hyperparameters, save_parameters, device = us.read_JsonB(file)
    
    print("Loading the ModisDatasetB...")
    train_ds = ModisDatasetB_scale_invariance('data/ModisDatasetB.csv', transf = dataset_parameter['transf'], split = 'Train', time = dataset_parameter['time'])
    val_ds = ModisDatasetB_scale_invariance('data/ModisDatasetB.csv', transf = dataset_parameter['transf'], split = 'Val', time = dataset_parameter['time'])
    print("Finished loading the dataset")
    
    print('Using device: {}'.format(device))
    
    # Not overwriting an older model
    if os.path.isdir(save_parameters['save_path']):
        print('The model chosen already exists.')
        print('Stopping the training.')
        sys.exit(0)
    
    modelB = ModelB_2(in_channels = modelB_parameters['in_channels'], 
                    downchannels = modelB_parameters['downchannels'],
                    padding_mode = modelB_parameters['padding_mode'],
                    activation = modelB_parameters['activation'],
                    bilinear = modelB_parameters['bilinear'],
                    n_bridge_blocks = modelB_parameters['n_bridge_blocks']).to(device) 
    
    # Initialization of the optimizer and of the loss function.
    optimizer = torch.optim.Adam(modelB.parameters(), lr=hyperparameters['learning_rate'])

    loss_fn = nn.HuberLoss(reduction = 'mean', delta = 1.0).to(device)
    
    checkpoint = us.model_checkpoint(hyperparameters['n_epochs'], hyperparameters['patience'])

    print("Starting the training.")
    # Training the model
    modelB, metrics = train(modelB,
                            train_ds, 
                            val_ds, 
                            hyperparameters['n_epochs'], 
                            hyperparameters['batch_size'], 
                            hyperparameters['learning_rate'],
                            loss_fn, 
                            optimizer,
                            device,
                            checkpoint)

    print("Training finished")
    
    ## Saving the experiment
    print("Saving the data in {}".format(save_parameters['save_path']))
    
    # Making the saving directory
    os.makedirs(save_parameters['save_path'], exist_ok=True)

    # Saving the model
    us.save_model(modelB, save_parameters['save_path'], save_parameters['model_name'])
        
    # Saving the results
    plot_loss(metrics, save_parameters['save_path'], save_parameters['model_name'])
        
    # Copying the parameter file in the results to keep track of the experiments.
    copy(file, os.path.join(save_parameters['save_path'], save_parameters['model_name'] + '_train_params.json'))
        
    # Saving the metrics
    with open(os.path.join(save_parameters['save_path'], save_parameters['model_name'] + '_lossdata.pkl'),'wb') as f:
        pickle.dump(metrics, f)




