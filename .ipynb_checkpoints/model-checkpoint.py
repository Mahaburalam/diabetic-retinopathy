#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn


# In[2]:


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# In[3]:


class ParallelAdd(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.branch1 = ConvBlock(in_c, out_c)
        self.branch2 = ConvBlock(in_c, out_c)

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


# In[4]:


class DRModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ParallelAdd(64, 64),  # Parallel branch (add for residual effect)
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)  # Last layer to replace in transfer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# In[5]:


# Test params
model = DRModel()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total learnable parameters: {total_params}")


# In[ ]:




