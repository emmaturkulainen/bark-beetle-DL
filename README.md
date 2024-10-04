# Bark beetle DL

Implementations of classifier models for spruce health classification and codes for their training and testing.
The models take as input .tif images of individual tree crowns and classify them into separate categories.

# Installation

The code has been developed in Python 3.10 environment. For other Python versions package conflicts may occur. To create the environment with Conda (To install Conda follow the official instructions):

```cd bark-beetle-DL```

```conda create -n bark-beetle-DL python=3.10```

```conda activate bark-beetle-DL```

Please follow the instructions [here] (https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install rest of the requirements with 

```pip install -r requirements.txt```

# Dataset

Download the [Dataset](https://drive.google.com/file/d/1zzKB3auHGvp3Nx_3BWEKw-HJzI5S7GfE/view?usp=drive_link) and extract it in the project directory. Dataset includes references tree samples from Paloheinä and Ruokolahti areas from years 2019 and 2020.

# Custom data

Organize the data accordingly:
```
├── data
| ├── train
| ├── test
| ├── val
| ├── train_labels.csv
| ├── test_labels.csv
| ├── val_labels.csv
```

Input images are to be named such that the image name includes a string containing the id of the image. E.g. 'id10', 'id12', etc. Image names can contain other information as well, as long the id is included. The corresponding class labels for the images are stored in .csv files with columns 'tree_id' and 'class'. 

Modify data.yaml to have the correct classes.

# Running the models

You can use the script ```run_model.py``` to perform training and testing of the models. To run the script with default arguments, use command
```python run_model.py```

You can specify the wanted network model using parameter ```--model```. The available models can be seen in ```cnns.py```. In addition, the code supports the use of VGG16 and ViT. Examples: 

```python run_model.py --model CNN2D```

```python run_model.py --model vgg```

```python run_model.py --model vit```

# Optimization

Optimization of the hyperparameters is implemented in ```optimizer.py```. The code runs trials with different configurations of batch size, learning rate and weight decay to achieve optimized training of the models. The hyperparameter values can have a large impact on the results. The optimizer results are stored by default in folder ```optimizer_outputs```. The results include the classification results and loss graphs of each completed trial as well as a search_results.csv file that contains a summary of the trial results. Example usage: 

```python optimizer.py --model CNN2D --datatype msi```
