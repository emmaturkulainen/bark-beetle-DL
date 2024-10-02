from model import create_datasets, load_data, train_model, test_model, imshow, visualize_model, get_representations, plot_representations, get_tsne
from datetime import datetime
import torch
from torch import nn
import sys
import os
import numpy as np
from cnns import CNN2D, CNN1_3D, CNN2_3D, CNN3_3D, CNN4_3D, CNN1_2D3D, CNN2_2D3D, Hyper3DNet
import pandas as pd
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
from torchvision.models import VGG16_BN_Weights
from torch.utils.data import DataLoader
import yaml
import timm
from utils.get_args import get_args


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
            result_path = os.path.join(args.output_dir, exp)
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
            result_path = os.path.join(args.output_dir, f"exp{exp_number}")
            try:
                # Attempt to create the directory
                os.mkdir(result_path)
                print(f"Created directory: {result_path}")
                return result_path
            except FileExistsError:
                # If the directory already exists, increment the number and try again
                exp_number += 1
    

def main():
    print("Script started at: ")
    clock()

    args = get_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
   
    result_path = create_exp_dir(args)
        
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

    # Parameters for training the model
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)

    train_dataset, test_dataset, val_dataset = create_datasets(train_img_dir, train_label_dir, test_img_dir, test_label_dir, val_img_dir, val_label_dir, img_size, mode=datatype)

    # Number of input channels
    in_channels = train_dataset[0][0].shape[0]


    # Use multiple GPU's if available
    if torch.cuda.device_count() > 1:
        print('Use multiple GPUs')
        model = nn.DataParallel(model)

    param = {
                'learning_rate': learning_rate,
                'train_batch_size': batch_size,
                'weight_decay': weight_decay,
                }

    dataloaders = load_data(train_dataset, val_dataset, batch_size)
    # Number of trainable parameters
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model, loss_hist, val_loss, optim_wts, best_model_loss_wts, best_model_acc_wts = train_model(param, model, device, dataloaders, epochs, num_classes)

    test_dl = DataLoader(test_dataset, batch_size=4)
    best_loss_oa, best_loss_precision, best_loss_recall, best_loss_f1 = test_model(model, device, test_dl, num_classes)


    # Visualize images (ONLY RGB)
    if datatype == 'rgb':
        # Get a batch of training data
        images, labels = next(iter(dataloaders['train']))[0:2][0:4]    
        # Make a grid from batch
        out = torchvision.utils.make_grid(images)    
        imshow(out, 1, result_path, title=[round(float(x)) for x in labels])

    print(model)
    # Load model to device
    model = model.to(device)

    # plot validation and training loss
    plt.figure()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(*zip(*val_loss))
    plt.plot(*zip(*loss_hist))
    plt.legend(['Validation loss', 'Training loss'])
    #plt.show()
    plt.savefig(result_path +'\curve.png')
    plt.close()

    if not os.path.exists(os.path.join(result_path, 'weights')):
        os.mkdir(os.path.join(result_path, 'weights'))

    # Save model with best validation loss 
    torch.save(model.state_dict(), result_path+'/weights/best_loss.pth')

    # Save model with best validation accuracy
    model.load_state_dict(best_model_acc_wts)
    torch.save(model.state_dict(), result_path+'/weights/best_acc.pth')

    # Save model at last training iteration
    model.load_state_dict(optim_wts)
    torch.save(model.state_dict(), result_path+'/weights/last.pth')

    # Print the accuracies on the test set
    with open('data.yaml', 'r') as file:
        content = yaml.safe_load(file)
        classes = content.get('names')

    for i, name in enumerate(classes):
        print(f"Precision for {name} trees is {best_loss_precision[i]:.3f}")
        print(f"Recall for {name} trees is {best_loss_recall[i]:.3f}")
        print(f"F1-score for {name} trees is {best_loss_f1[i]:.3f}")
        print()  # For better readability


    results_dict = {}

    results_dict['overall accuracy'] = best_loss_oa
    results_dict['precision'] = best_loss_precision
    results_dict['recall'] = best_loss_recall
    results_dict['F1-score'] = best_loss_f1
    results = pd.DataFrame(results_dict, classes)
    results.to_csv(result_path+"/best_val_loss_results.csv", sep=';')

    outputs, predictions, labels, id = get_representations(model, test_dl, device)
    output_tsne_data = get_tsne(outputs)
    plot_representations(output_tsne_data, labels, classes, result_path, "val_loss")

    model.load_state_dict(best_model_acc_wts)
    best_acc_oa, best_acc_precision, best_acc_recall, best_acc_f1 = test_model(model, device, test_dl, num_classes)

    results_dict = {}
    results_dict['overall accuracy'] = best_acc_oa
    results_dict['precision'] = best_acc_precision
    results_dict['recall'] = best_acc_recall
    results_dict['F1-score'] = best_acc_f1

    results = pd.DataFrame(results_dict, classes)
    results.to_csv(result_path+"/best_val_acc_results.csv", sep=';')

    outputs, predictions, labels, id = get_representations(model, test_dl, device)
    output_tsne_data = get_tsne(outputs)
    plot_representations(output_tsne_data, labels, classes, result_path, "val_acc")

    model.load_state_dict(optim_wts)
    optim_oa, optim_precision, optim_recall, optim_f1 = test_model(model, device, test_dl, num_classes)

    results_dict = {}
    results_dict['overall accuracy'] = optim_oa
    results_dict['precision'] = optim_precision
    results_dict['recall'] = optim_recall
    results_dict['F1-score'] = optim_f1

    results = pd.DataFrame(results_dict, classes)
    results.to_csv(result_path+"/end_of_training_results.csv", sep=';')

    outputs, predictions, labels, id = get_representations(model, test_dl, device)
    output_tsne_data = get_tsne(outputs)
    plot_representations(output_tsne_data, labels, classes, result_path, "end_of_training")

    # Write args to text file
    with open(result_path + '/args.txt', 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")

    print("Script ended at: ")
    clock()
    

if __name__ == "__main__":
    main()