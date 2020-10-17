#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
If you have any questions, please contact me with https://github.com/xufana7/AutoEncoder-with-pytorch
Author, Fan xu Aug 2020
"""
import numpy as np
import h5py
import torch
from Model_define_pytorch import AutoEncoder, DatasetFolder
import os
import torch.nn as nn
from Model_define_pytorch import NMSE
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# Parameters for training
# resume = 222
resume = -1
start_epoch = resume + 1
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
use_single_gpu = True  # select whether using single gpu or multiple gpus
device = torch.device("cuda:0")
torch.manual_seed(1)
batch_size = 64
epochs = 1000
learning_rate = 1e-3
num_warmup_steps = 0
num_workers = 4
print_freq = 100  # print frequency (default: 60)
# parameters for data
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2

# Model construction
model = AutoEncoder(feedback_bits)
if use_single_gpu:
    model = model.cuda()

else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()

# modelSave1 = './Modelsave/encoder.pth.tar'
# model.encoder.load_state_dict(torch.load(modelSave1)['state_dict'])

# modelSave2 = './Modelsave/decoder.pth.tar'
# model.decoder.load_state_dict(torch.load(modelSave2)['state_dict'])

# Data loading
data_load_address = './data'
mat = h5py.File(data_load_address + '/H_train.mat', 'r')
data = np.transpose(mat['H_train'])  # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
# split data for training(70%) and validation(30%)
np.random.shuffle(data)
start = int(data.shape[0] * 0.7)
x_train, x_test = data, data[start:]

# dataLoader for training
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# dataLoader for training
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
best_loss = 1

criterion = nn.MSELoss().cuda()
optimizer = AdamW([{'params': model.parameters(), 'initial_lr': learning_rate}], lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps= len(train_loader) * epochs, last_epoch=len(train_loader) * start_epoch
)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
training_iterator = tqdm(total=epochs * len(train_loader), desc='Epoch: ')
training_iterator.update(len(train_loader) * start_epoch)
# display_dict = {'TL':'', 'loss':'', Model_Save: False}
for epoch in range(start_epoch, epochs):
    # model training
    model.train()
    for i, input in enumerate(train_loader):
        input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, input)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        training_iterator.set_postfix({f'loss({i+1}/{len(train_loader)})':f'{loss.item():.3e}', 
                                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        training_iterator.update(1)

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # convert numpy to Tensor
            input = input.cuda()
            output = model(input)
            total_loss += criterion(output, input).item() * input.size(0)
        average_loss = total_loss / len(test_dataset)
        print(f'\nEpoch[{epoch}]:\tTest_Loss:{average_loss:.3e}')
        if average_loss < best_loss:
            # model save
            # save encoder
            modelSave1 = './Modelsave/encoder.pth.tar'
            torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            # save decoder
            modelSave2 = './Modelsave/decoder.pth.tar'
            torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            print('Model Save')
            best_loss = average_loss
