# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:48:45 2021

@author: JEK
edited by KK 2021-2022
edited by ET: 2022-2024
"""

from __future__ import print_function
from __future__ import division
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import copy
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn import manifold
import pandas as pd
# import dataset
#from data.dataset import Dataset
from combination_dataset import CombinationDataset
import optuna



# Training, validation and test datasets
def create_datasets(img_dir, data_dir, test_img_dir, test_data_dir, val_img_dir, val_data_dir, img_size, data_augmentation=True, mode='rgb'):
    # very basic transforms, modify as needed
    if data_augmentation:
        transformations = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-180, 180)),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #ImageNet (uncomment if using ImageNet pretrained model)
                ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #ImageNet (uncomment if using ImageNet pretrained model)
                ])
                }
    else:
        transformations = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                ])
                }

    if mode == 'rgb':
        train_dataset = CombinationDataset(img_dir, None, None, data_dir, channels=3, img_size=img_size, transform=transformations['train'])
        val_dataset = CombinationDataset(val_img_dir, None, None, val_data_dir,  channels=3, img_size=img_size, transform=transformations['val'])
        test_dataset = CombinationDataset(test_img_dir, None, None, test_data_dir, channels=3, img_size=img_size, transform=transformations['val'])

    elif mode == 'hsi':
        train_dataset = CombinationDataset(None, img_dir, None, data_dir, channels=46, img_size=img_size, transform=transformations['train'])
        val_dataset = CombinationDataset(None, val_img_dir, None, val_data_dir, channels=46, img_size=img_size, transform=transformations['val'])
        test_dataset = CombinationDataset(None, test_img_dir, None, test_data_dir, channels=46, img_size=img_size, transform=transformations['val'])

    elif mode == 'msi':
        train_dataset = CombinationDataset(None, None, img_dir, data_dir, channels=5, img_size=img_size, transform=transformations['train'])
        val_dataset = CombinationDataset(None, None, val_img_dir, val_data_dir, channels=5, img_size=img_size, transform=transformations['val'])
        test_dataset = CombinationDataset(None, None, test_img_dir, test_data_dir, channels=5, img_size=img_size, transform=transformations['val'])

    
    else:
        print('CAUTION: Incorrect mode')

    return train_dataset, test_dataset, val_dataset


def load_data(train_dataset, val_dataset, train_batch_size, pin_memory=True, num_workers=0):

    val_batch_size = 4
    # split the data for training and validation
    #valid_size = 0.2
    #num_train = len(train_dataset)
    #indices = list(range(num_train))
    #split = int(np.floor(valid_size * num_train))

    #np.random.shuffle(indices) # shuffle indices for random train-val split
    #train_idx, val_idx = indices[split:], indices[:split] # indices for train and val

    # PyTorch Sampler for training and validation
    #train_sampler = SubsetRandomSampler(train_idx)
    #val_sampler = SubsetRandomSampler(val_idx)

    # PyTorchDataloader for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size,
                                num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size,
                                    num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    # Dataloaders dict for training-validation-loop
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    return dataloaders

# Train and evaluate the accuracy of neural network model
def train_model(param, model, device, dataloaders, num_epochs, num_classes, trial=None):
    
    since = time.time()
    
    loss_history = []
    val_losses = []
    best_val_loss = np.inf
    best_val_acc = 0
    best_model_loss_wts = copy.deepcopy(model.state_dict())
    best_model_acc_wts = copy.deepcopy(model.state_dict())

     # Calculate class weights for weighted cross entropy loss
    all_labels = []
    for images, labels, ids in dataloaders['train']:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels)
    class_counts = torch.bincount(all_labels)
    class_weights = 1.0/class_counts
    normalized_weights = class_weights / torch.sum(class_weights) * len(class_counts)
    weights = normalized_weights.to(device)

    try:
        criterion = nn.CrossEntropyLoss(weight = weights) 
    except:
        criterion = nn.CrossEntropyLoss() # fails if data doesn't include samples from all classes

    optimizer = torch.optim.AdamW(model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])

    for epoch in tqdm(range(num_epochs)):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            phase_size = 0

            # iterate over data
            for images, labels, ids in dataloaders[phase]:
                phase_size += len(labels)
                images = images.to(device)
                labels = labels.type(torch.LongTensor)                 
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
            
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(images)#.squeeze(1)
                    loss = criterion(outputs, labels)                  
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

               
                # statistics
                running_loss += loss.item() * images.size(0)            

            epoch_loss = running_loss / phase_size

            if phase == 'train':
                loss_history.append((epoch, epoch_loss))
                print(f"\nTraining loss = {loss_history[-1]}")
	
            if phase == 'val':
                oa, precision, recall, f1 = val_model(model, device, dataloaders[phase], num_classes)
                epoch_acc = oa
                if trial != None:
                    trial.report(epoch_loss, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                if epoch_loss <= best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_loss_wts = copy.deepcopy(model.state_dict())
                    print(f"Best validation loss updated at epoch {epoch}, loss: {best_val_loss}")
                if epoch_acc >= best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_acc_wts = copy.deepcopy(model.state_dict())
                val_losses.append((epoch, epoch_loss))
                print(f"Validation loss = {val_losses[-1]}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # save weights with the least training loss
    optim_wts = copy.deepcopy(model.state_dict())
    # load best model weights
    model.load_state_dict(best_model_loss_wts)
    
    return model, loss_history, val_losses, optim_wts, best_model_loss_wts, best_model_acc_wts


def val_model(model, device, val_dl, num_classes):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for idx, (images, labels, ids) in enumerate(val_dl):
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          outputs = outputs.cpu().detach().numpy()
          actual = labels.cpu().numpy()
          predicted = np.argmax(outputs, axis=1)
          actual = actual.reshape((len(actual), 1))
          predicted = predicted.reshape((len(predicted), 1))
          predictions.append(predicted)
          ground_truths.append(actual)
          total += labels.size(0)
          correct += accuracy_score(actual, predicted, normalize=False)

    predictions, ground_truths = np.vstack(predictions), np.vstack(ground_truths)
    oa = correct / total # overall accuracy
    precision = precision_score(ground_truths, predictions, average=None, labels=list(range(num_classes)), zero_division = 0)
    recall = recall_score(ground_truths, predictions, average=None, labels =list(range(num_classes)), zero_division = 0)
    f1 = np.divide(2*np.multiply(precision, recall), precision+recall+1)

    return oa, precision, recall, f1   


# This function computes the accuracy on the test dataset
def test_model(model, device, test_dl, num_classes):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for idx, (images, labels, ids) in enumerate(test_dl):
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          outputs = outputs.cpu().detach().numpy()
          actual = labels.cpu().numpy()
          predicted = np.argmax(outputs, axis=1)
          actual = actual.reshape((len(actual), 1))
          predicted = predicted.reshape((len(predicted), 1))
          predictions.append(predicted)
          ground_truths.append(actual)
          total += labels.size(0)
          correct += accuracy_score(actual, predicted, normalize=False)

    predictions, ground_truths = np.vstack(predictions), np.vstack(ground_truths)
    oa = correct / total # overall accuracy
    precision = precision_score(ground_truths, predictions, average=None, labels=list(range(num_classes)))
    recall = recall_score(ground_truths, predictions, average=None, labels =list(range(num_classes)))
    f1 = np.divide(2*np.multiply(precision, recall), precision+recall)


    return oa, precision, recall, f1




def imshow(inp, n, folder, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title, loc='center', wrap=True)
    #plt.pause(0.001)            # pause a bit so that plots are updated
    plt.savefig(folder + '/train_data_sample.jpg')
    plt.close()




# Model visualization (RGB)
def visualize_model(model, device, folder, dataloaders, mode = 'rgb', num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(10, 10))
    
    with torch.no_grad():
        for (images, labels) in iter(dataloaders['val']):
            images = images.to(device)
            labels = labels.to(device)
            print('\nLabels:')
            print(labels)
            outputs = model(images)[:num_images]
            print('\nOutputs: ')
            print(outputs)
            
            # Only for RGB
            if mode == 'rgb':
                for j in range(images.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    outputs = outputs.cpu()
                    labels = labels.cpu()
                    ax.set_title('predicted: {}'.format(np.where(np.array(outputs[j] == max(outputs[j])))[0])+ '\n ground truth: {}'.format(round((float(labels[j])))))
                    plt.imshow(images.cpu().data[j].numpy().transpose((1,2,0)))
                                        
                    if images_so_far == num_images:
                        plt.savefig(folder+'/vismodel.jpg')
                        plt.clf()
                        model.train(mode=was_training)
                        return
    
    model.train(mode=was_training)


def get_representations(model, dl, device):

    model.eval()

    outputs = []
    predictions = []
    labels = []
    ids = []

    with torch.no_grad():
        
        for (x, y, id) in iter(dl):

            x = x.to(device)

            y_pred = model(x)
            predicted = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
            #predicted = predicted.reshape((len(predicted), 1))
            outputs.append(y_pred.cpu())
            for p in predicted:
                predictions.append(p)
            labels.append(y)
            ids.append(id)
        
    outputs = torch.cat(outputs, dim = 0)
    labels = torch.cat(labels, dim = 0)
    ids = torch.cat(ids, dim=0)

    return outputs, predictions, labels, ids


def plot_representations(data, labels, classes, folder, fig_name, n_images = None):
    
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
        
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c = labels, cmap = 'tab10')
    handles, labels = scatter.legend_elements()
    ax.legend(handles = handles, labels = classes)
    plt.savefig(folder+'/t-sne' + fig_name + '.jpg')
    #plt.show()
    

def get_tsne(data, n_components = 2, n_images = None):
    
    if n_images is not None:
        data = data[:n_images]
        
    tsne = manifold.TSNE(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data


        

