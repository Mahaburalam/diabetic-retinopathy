#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import cv2


# In[5]:


data_dir = 'data/train_images/'
csv_file = 'data/train.csv'


# In[6]:


df = pd.read_csv(csv_file)


# In[7]:


# Compute approximate means for zero-center (or calculate from dataset)
mean = [0.485, 0.456, 0.406]  # Standard ImageNet means; adjust if needed
std = [0.229, 0.224, 0.225]


# In[8]:


# Transforms for training (augmentation) and testing
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(227, scale=(0.8, 1.2)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)  # Zero-center approx
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(227),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# In[9]:


class DRDataset(Dataset):
    def __init__(self, df, transform=None, is_binary=True):
        self.df = df
        self.transform = transform
        self.is_binary = is_binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(data_dir, self.df.iloc[idx, 0] + '.png')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.iloc[idx, 1]
        if self.is_binary:
            label = 0 if label == 0 else 1  # Binary: 0=no DR, 1=DR
        else:
            if label == 0: return None  # Skip no-DR for severity
            label -= 1  # Severity: 0=mild, 1=moderate, 2=severe, 3=proliferative
        if self.transform:
            image = self.transform(image)
        return image, label


# In[10]:


# For binary
binary_df = df.copy()
train_bin_df, test_bin_df = train_test_split(binary_df, test_size=0.2, stratify=binary_df['diagnosis'], random_state=42)
train_bin_dataset = DRDataset(train_bin_df, train_transform, is_binary=True)
test_bin_dataset = DRDataset(test_bin_df, test_transform, is_binary=True)


# In[11]:


# For severity (filter DR only)
severity_df = df[df['diagnosis'] > 0]
train_sev_df, test_sev_df = train_test_split(severity_df, test_size=0.2, stratify=severity_df['diagnosis'], random_state=42)
train_sev_dataset = DRDataset(train_sev_df, train_transform, is_binary=False)
test_sev_dataset = DRDataset(test_sev_df, test_transform, is_binary=False)


# In[14]:


# DataLoaders (batch=128 as in paper)
batch_size = 128
train_bin_loader = DataLoader(train_bin_dataset, batch_size=batch_size, shuffle=True)
test_bin_loader = DataLoader(test_bin_dataset, batch_size=batch_size, shuffle=False)
train_sev_loader = DataLoader(train_sev_dataset, batch_size=batch_size, shuffle=True)
test_sev_loader = DataLoader(test_sev_dataset, batch_size=batch_size, shuffle=False)


# In[13]:


print("Data ready!")


# In[ ]:




