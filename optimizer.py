# -*- coding: utf-8 -*-
"""
@author: JEK
edited by KK 2021-2022
edited by ET 2022-2024
"""

from __future__ import print_function
from __future__ import division
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from cnns import CNN1_3D, CNN2D, CNN2_2D3D, CNN3_3D, CNN4_3D, CNN2_3D, CNN1_2D3D, Hyper3DNet
import optuna
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from torchvision.models import VGG16_BN_Weights
import logging
from model import create_datasets, load_data, train_model, val_model, test_model
import yaml
import timm
from utils.get_args import get_args
from combination_dataset import CombinationDataset

def clock():
    now = datetime.now()
    print("Time: ", now)

def create_exp_dir(args):
    # user defined experiment name
    if args.exp:
        exp_number = 1
        exp = args.exp
        while True:
            # Construct the new directory name
            result_path = os.path.join(args.optimizer_output_dir, exp)
            try:
                # Attempt to create the directory
                os.mkdir(result_path)
                print(f"Created directory: {result_path}")
                return result_path
            except FileExistsError:
                # If the directory already exists, increment the number and try again
                exp = args.exp + str(exp_number)
                exp_number+=1

    # default experiment name (exp1, exp2, ...)
    else:
        exp_number = 1
        while True:
            # Construct the new directory name
            result_path = os.path.join(args.optimizer_output_dir, f"exp{exp_number}")
            try:
                # Attempt to create the directory
                os.mkdir(result_path)
                print(f"Created directory: {result_path}")
                return result_path
            except FileExistsError:
                # If the directory already exists, increment the number and try again
                exp_number += 1

def objective(trial, result_path):

    args = get_args()
    print(args)

    seed = int(args.seed)
    datatype = args.datatype
    model_str = args.model
    weights = args.weights

    datadir = args.datadir
    with open('data.yaml', 'r') as file:
        content = yaml.safe_load(file)
        num_classes = content.get('nc')

    # use GPU if available
    if torch.cuda.is_available():
        device = args.device
    else:
        print("Cuda not available. Using cpu instead.")
        device = "cpu"

    #HERE fix ALL seeds!
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False   # benchmark mode is good whenever your input sizes for the network do not vary. If input size varies, it may negatively effect the runtime performance.

    torch.backends.cudnn.deterministic = True # only uses deterministic convolution algorithms

    if datatype == 'rgb':
        in_channels = 3
        img_size = 150
    elif datatype == 'msi':
        in_channels = 5
        img_size = 150
    elif datatype == 'hsi':
        in_channels = 46
        img_size = 50
    else:
        print("Incorrect datatype")

    # Initialize model
    if model_str == 'vgg':
        model = models.vgg16_bn()
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3,3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = torch.nn.Linear(4096, num_classes) # for vgg

    elif model_str == 'vit':
        img_size = 224
        model = timm.create_model('vit_small_patch16_224', num_classes=num_classes, pretrained=False, in_chans=in_channels) 

    else: 
        model = eval(model_str)(in_channels=in_channels, num_classes=num_classes)
    
    model.to(device)

    train_img_dir = os.path.join(datadir, "train")
    train_label_dir = os.path.join(datadir, "train_labels.csv")

    test_img_dir = os.path.join(datadir, "test")
    test_label_dir = os.path.join(datadir, "test_labels.csv")

    val_img_dir = os.path.join(datadir, "val")
    val_label_dir = os.path.join(datadir, "val_labels.csv")

    if weights:
        # Load model
        model.load_state_dict(torch.load(weights))
    else:
        print("Training model from scratch.")

    epochs = int(args.epochs)

    train_dataset, test_dataset, val_dataset = create_datasets(train_img_dir, train_label_dir, test_img_dir, test_label_dir, val_img_dir, val_label_dir, img_size, mode=datatype)

    # Number of input channels
    in_channels = train_dataset[0][0].shape[0]


    # Use multiple GPU's if available
    if torch.cuda.device_count() > 1:
        print('Use multiple GPUs')
        model = nn.DataParallel(model)

    params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-4, log=True),
            'train_batch_size': trial.suggest_int('train_batch_size', 12, 64, step=4),
            'weight_decay': trial.suggest_float('weight_decay', 1e-3, 1e-1, log=True),
            }

    dataloaders = load_data(train_dataset, val_dataset, params['train_batch_size'])
    # Number of trainable parameters
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model, loss_hist, val_loss, optim_wts, best_model_loss_wts, best_model_acc_wts = train_model(params, model, device, dataloaders, epochs, num_classes, trial=trial)
    
    # Use multiple GPU's if available
    if torch.cuda.device_count() > 1:
        print('Use multiple GPUs')
        model = nn.DataParallel(model)
    
    test_dl = DataLoader(test_dataset, batch_size=4)
    oa, precision, recall, f1 = test_model(model, device, test_dl, num_classes)
    #score = np.mean(f1) # optimize mean f1 score between classes
    epoch, score = val_loss[-1]
    
    plt.figure()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(*zip(*val_loss))
    plt.plot(*zip(*loss_hist))
    plt.legend(['Validation loss', 'Training loss'])
    #plt.show()
    plt.savefig(result_path+ "/batchsize" + str(params["train_batch_size"]) + "_LR" + str(params["learning_rate"]) + "_WD" + str(params["weight_decay"]) + '_curve.png')
    plt.close()

    torch.save(model.state_dict(), result_path+ "/batchsize" + str(params["train_batch_size"]) + "_LR" + str(params["learning_rate"]) + "_WD" + str(params["weight_decay"]) + 'trained_model.pth')
    
    with open('data.yaml', 'r') as file:
        content = yaml.safe_load(file)
        classes = content.get('names')

    results_dict = {}
    results_dict['overall accuracy'] = oa
    results_dict['precision'] = precision
    results_dict['recall'] = recall
    results_dict['F1-score'] = f1

    results = pd.DataFrame(results_dict, classes)
    results.to_csv(result_path + "/batchsize" + str(params["train_batch_size"]) + "_LR" + str(params["learning_rate"]) + "_WD" + str(params["weight_decay"]) + "_results.csv", sep=';')

    return score

def main():

    print("Script started at: ")
    clock()
    args = get_args()

    if not os.path.exists(args.optimizer_output_dir):
        os.mkdir(args.optimizer_output_dir)
   
    result_path = create_exp_dir(args)
        
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.HyperbandPruner(min_resource=3,  max_resource=args.epochs))
    study.optimize(lambda trial: objective(trial, result_path), n_trials=200)

    # Create a dataframe from the study.
    search_df = study.trials_dataframe()

    search_df.to_csv(result_path+"/search_results.csv", sep=';')

    # Write args to text file
    with open(result_path + '/args.txt', 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")

    print("Script ended at: ")
    clock()


if __name__ == "__main__":
    main()