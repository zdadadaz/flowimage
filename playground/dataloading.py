# -*- coding: utf-8 -*-

import dataloader
import tqdm
import matplotlib.pyplot as plt
import numpy as np
# from guiqwt.pyplot import *

# data_loader = dataloader.Motion_DataLoader(
#                     BATCH_SIZE=2,
#                     num_workers=8,
#                     path='/home/zdadadaz/Desktop/course/medical/data/UCF101/tvl1_flow',
#                     ucf_list='../UCF_list/',
#                     ucf_split='01',
#                     in_channel=10,
#                     root_path = '../'
#                     )

data_loader = dataloader.Motion_Image_DataLoader(
                    BATCH_SIZE=2,
                    num_workers=8,
                    path='/home/zdadadaz/Desktop/course/medical/data/UCF101',
                    ucf_list='../UCF_list/',
                    ucf_split='01',
                    in_channel=32,
                    root_path = '../'
                    )


train_loader,test_loader, test_video = data_loader.run()


# for i, (data,label) in enumerate(train_loader):
#     print(data, label)
#     qq = 0
#     aa =0

for i, (data,label) in enumerate(train_loader):
    print(data.shape, label.shape)
    qq = 0
    aa =0