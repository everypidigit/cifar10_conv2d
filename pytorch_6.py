#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets.utils import download_url 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


project_name = 'cifar_dataset'


# In[3]:


dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz'
download_url(dataset_url, '.')


# In[4]:


with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    tar.extractall(path='./data')


# In[5]:


data_dir = './data/cifar10'

print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)


# In[6]:


airplane_files = os.listdir(data_dir + "/train/airplane")
print('No. of training examples for airplanes:', len(airplane_files))
print(airplane_files[:5])


# In[7]:


ship_test_files = os.listdir(data_dir + "/test/ship")
print("No. of test examples for ship:", len(ship_test_files))
print(ship_test_files[:5])


# In[8]:


dataset = ImageFolder(data_dir+'/train', transform=ToTensor())


# In[9]:


img, label = dataset[0]
print(img.shape, label)


# In[10]:


img


# In[11]:


print(dataset.classes)


# In[12]:


import matplotlib

matplotlib.rcParams['figure.facecolor'] = 'ffffff'


# In[13]:


def show_img(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
    


# In[14]:


show_img(*dataset[0])


# In[15]:


show_img(*dataset[1000])


# In[16]:


seed = 42
torch.manual_seed(seed);


# In[17]:


val_size = 5000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)
batch_size = 128


# In[18]:


train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)


# In[19]:


from torchvision.utils import make_grid


# In[20]:


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break


# In[21]:


show_batch(train_dl)


# In[22]:


# we used a fully-connected neural network in the previous try, now we will use a convolutional neural network

def apply_kernel(image, kernel):
    ri, ci = image.shape
    rk, ck = kernel.shape
    ro, co = ri-rk+1, rk-ck+1
    output = torch.zeros([ro, co])
    for i in range(ro):
        for j in range(co):
            output[i,j] = torch.sum(image[i:i+rk,j:j+ck] * kernel)
    return output


# In[23]:


sample_image = torch.tensor([
    [3, 3, 2, 1, 0], 
    [0, 0, 1, 3, 1], 
    [3, 1, 2, 2, 3], 
    [2, 0, 0, 2, 2], 
    [2, 0, 0, 0, 1]
], dtype=torch.float32)

sample_kernel = torch.tensor([
    [0, 1, 2], 
    [2, 2, 0], 
    [0, 1, 2]
], dtype=torch.float32)

apply_kernel(sample_image, sample_kernel)


# In[24]:


simpler_model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size = 3, stride=1, padding=1),
    nn.MaxPool2d(2,2)
)


# In[25]:


for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = simpler_model(images)
    print('out.shape:', out.shape)
    break


# In[26]:


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropu(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc' : acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stach(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return{'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[27]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[ ]:




