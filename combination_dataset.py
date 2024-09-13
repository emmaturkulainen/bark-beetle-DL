# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:41:11 2021

@author: JEK
edited by KK 2021-2022
"""

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import math
from PIL import Image, ImageEnhance
import cv2
import rasterio
import torchvision

# data frame to csv for creating a torch Dataset
class CombinationDataset(Dataset):
    def __init__(self, rgb_dir, hsi_dir, msi_dir, data_file, channels, img_size, transform=None, target_transform=None, normalize=True):
              
        
        self.data_file = data_file
        self.normalize = normalize
        self.channels = channels
        self.img_size = img_size 
        
        # Load labels to a data frame from a csv
        try:
            labels = pd.read_csv(self.data_file, delimiter=',', usecols=['tree_id', 'class']) 
        except:
            labels = pd.read_csv(self.data_file, delimiter=';', usecols=['tree_id', 'class']) 
        # Create dataframe for images
        df =  pd.DataFrame(columns=['id', 'rgb', 'msi', 'hsi', 'image', 'class'])

        # Map hyperspectral images as arrays to 'hsi' column
        if hsi_dir:
            hsi_images = hsi_df(hsi_dir, self.channels, img_size)
            df['hsi'] = hsi_images['image']
            df['id'] = hsi_images['id']
            
        # Map RGB images as arrays to 'rgb' column
        if rgb_dir:
            rgb_images = rgb_df(rgb_dir, img_size)
            df['rgb'] = rgb_images['image']
            df['id'] = rgb_images['id']
            
        # Map msi images as arrays to 'msi' column
        if msi_dir:
            msi_images = msi_df(msi_dir, self.channels, img_size)
            df['msi'] = msi_images['image']
            df['id'] = msi_images['id']
            
        # Combine RGB, MSI and HSI arrays to one array
        if rgb_dir and hsi_dir and msi_dir:
            df['image'] = df['rgb'].combine(df['msi'], func=lambda img1, img2: np.concatenate((img1, img2), axis=2)).combine(df['hsi'], func=lambda img1, img2: np.concatenate((img1, img2), axis=2))
        elif rgb_dir and msi_dir:
            df['image'] = df['rgb'].combine(df['msi'], func=lambda img1, img2: np.concatenate((img1, img2), axis=2))
        elif rgb_dir and hsi_dir:
            df['image'] = df['rgb'].combine(df['hsi'], func=lambda img1, img2: np.concatenate((img1, img2), axis=2))
        elif hsi_dir and msi_dir:
            df['image'] = df['msi'].combine(df['hsi'], func=lambda img1, img2: np.concatenate((img1, img2), axis=2))
        elif hsi_dir:
            df['image'] = df['hsi']
        elif rgb_dir:
            df['image'] = df['rgb']
        elif msi_dir:
            df['image'] = df['msi']    

        df['class'] = df['id'].map(lambda i: labels.loc[labels['tree_id'] == i]['class'].item())
            
        self.data = df
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data['class'])
     
    def __getitem__(self, idx):
        image = self.data['image'][idx]
        label = self.data['class'][idx]
        id = self.data['id'][idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, id
    
    
def groupById(dataframe): return pd.DataFrame(dataframe.groupby(dataframe.applymap(lambda filename: filename.split('id')).iloc[:,0].map(lambda string: string[1]))) 

def groupByN(dataframe): return pd.DataFrame(dataframe.groupby(dataframe.applymap(lambda filename: filename.split('id')[0].split('n')[1]).iloc[:,0]))

# Aux function for filtering integers from a string for ordering by ID
def filter_ints(string): 
            ints = [int(i) for i in filter(lambda s: s.isdigit(), string)]
            return int("".join(map(str, ints)))
    
# convert images to correct format (float32) and resize images to square shapes
def rescale(datadir, filename, img_size):
    dataset = rasterio.open(os.path.join(datadir, filename))
    im = dataset.read([1,2,3]) # some images have extra channel in addition to RGB-channels. Only read RGB
    # Handle nodata values if they exist
    nodata = dataset.nodata
    if nodata is not None:
        # Set nodata values to a value that fits within the float32 range
        im = np.where(im == nodata, 0, im)

    # Normalize the image data to float32 range
    im = np.interp(im, (im.min(), im.max()), (0, 1)).astype(np.float32)
    
    im = np.transpose(im, axes=[1, 2, 0])
    im_resized = cv2.resize(im, (img_size, img_size))
    return im_resized

# convert multispectral images to correct format (float32) and resize images to square shapes
def rescale_msi(datadir, filename, img_size):
    dataset = rasterio.open(os.path.join(datadir, filename))
    im = dataset.read([1,2,3,4,5]) # remove thermal band
    # Handle nodata values if they exist
    nodata = dataset.nodata
    if nodata is not None:
        # Set nodata values to a value that fits within the float32 range
        im = np.where(im == nodata, 0, im)

    # Normalize the image data to float32 range
    im = np.interp(im, (im.min(), im.max()), (0, 1)).astype(np.float32)
    
    im = np.transpose(im, axes=[1, 2, 0])
    im_resized = cv2.resize(im, (img_size, img_size))
    return im_resized
    
# convert hyperspectral images to correct format (float32) and resize images to square shapes
def rescale_hsi(datadir, filename, img_size):
    dataset = rasterio.open(os.path.join(datadir, filename))
    im = dataset.read()
    # Handle nodata values if they exist
    nodata = dataset.nodata
    if nodata is not None:
        # Set nodata values to a value that fits within the float32 range
        im = np.where(im == nodata, 0, im)

    # Normalize the image data to float32 range
    im = np.interp(im, (im.min(), im.max()), (0, 1)).astype(np.float32)
    
    im = np.transpose(im, axes=[1, 2, 0])
    im_resized = cv2.resize(im, (img_size, img_size))
    return im_resized


  
def hsi_df(hsi_dir, channels, img_size): 
    
    # define dataframe for hsi filenames
    hsi_data = pd.DataFrame(columns=['id', 'filename', 'image'])

    # filenames of the hyperspectral images
    hsi_data['filename'] = os.listdir(hsi_dir)

    # id's of each image
    hsi_data['id'] = hsi_data['filename'].map(lambda fn: int(filter_ints(fn.split('id')[1]))) 

    #print(hsi_data)

    # stack filenames to hyperspectral image arrays
    hsi_data['image'] = stack_hsi(hsi_dir, hsi_data['filename'], img_size)

    # pad to size
    hsi_data['image'] = pad_to_size(hsi_data['image'], img_size)

    # add missing first spectral band (if missing)
    hsi_data['image'] = hsi_data['image'].map(lambda array: np.pad(array, ((0, 0), (0,0), (channels-array.shape[2], 0))))

    hsi_data = hsi_data.sort_values(by=['id']).reset_index(drop=True)

    return hsi_data
    
def msi_df(msi_dir, channels, img_size):

    msi_images = pd.DataFrame(columns=['id', 'filename', 'image'])

    # msi image filenames
    msi_images['filename'] = os.listdir(msi_dir)

    # id's of each image
    msi_images['id'] = msi_images['filename'].map(lambda fn: int(filter_ints(fn.split('id')[1])))

    
    # Map corresponding images to arrays with values between [0, 1]
    msi_images['image'] = msi_images['filename'].map(lambda filename: np.array(rescale_msi(msi_dir, filename, img_size)))
    #msi_images['image'] = msi_images['filename'].map(lambda filename: np.array(Image.open(os.path.join(msi_dir, filename))))
    msi_images['image'] = msi_images['image'].map(lambda array: (np.nan_to_num(np.where(array<0, 0, array))))
    #msi_images['image'] = msi_images['image'].map(lambda array: np.expand_dims(array, axis=2))

    # pad to size
    msi_images['image'] = pad_to_size(msi_images['image'], img_size)

    msi_images['image'] = msi_images['image'].map(lambda array: np.pad(array, ((0, 0), (0,0), (channels-array.shape[2], 0))))

    msi_images = msi_images.sort_values(by=['id']).reset_index(drop=True)

    return msi_images

def rgb_df(rgb_dir, img_size):
    
    rgb_images = pd.DataFrame(columns=['id', 'filename', 'image'])

    # rgb image filenames
    rgb_images['filename'] = os.listdir(rgb_dir)

    # id's of each image
    rgb_images['id'] = rgb_images['filename'].map(lambda fn: int(filter_ints(fn.split('id')[1])))
    
    
    # Map corresponding images to arrays with values between [0, 1]
    rgb_images['image'] = rgb_images['filename'].map(lambda filename: np.array(rescale(rgb_dir, filename, img_size)))
 
    #rgb_images['image'] = rgb_images['filename'].map(lambda filename: np.array(Image.open(os.path.join(rgb_dir, filename)).convert('RGB'))/255)
    rgb_images['image'] = rgb_images['image'].map(lambda array: (np.nan_to_num(np.where(array<0, 0, array))))
    
    # pad to size
    rgb_images['image'] = pad_to_size(rgb_images['image'], img_size)
    
    rgb_images = rgb_images.sort_values(by=['id']).reset_index(drop=True)
    
    return rgb_images

def pad_to_size(df, size):
    return df.map(lambda array: np.pad(array, ((math.floor((size-array.shape[0])/2), math.ceil((size-array.shape[0])/2)), 
        (math.floor((size-array.shape[1])/2), math.ceil((size-array.shape[1])/2)), (0,0))))



def stack_hsi(hsi_dir, filenames, img_size):
    
    # Map filenames to corresponding images with PIL and further to numpy arrays
    hsi = filenames.map(lambda filename: np.array(rescale_hsi(hsi_dir, filename, img_size))) #rescale
    #hsi = filenames.map(lambda list_of_filenames: list_of_filenames.map(lambda filename: np.array(Image.open(os.path.join(hsi_dir, filename))))) 
    
    # Map negative numbers and NaNs to zero
    hsi = hsi.map(lambda array: (np.nan_to_num(np.where(array < 0, 0, array))))
    #print(hsi.shape)
    # Stack image arrays of each ID to form a 36 x [img height] x [img width] images
    #hsi = hsi.map(np.array(arrays.squeeze())).transpose(1,2,0)
    
    
    return hsi