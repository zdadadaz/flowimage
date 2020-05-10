# -*- coding: utf-8 -*-

import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse
import os.path as o

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader.split_train_test_video import *



class motion_image_dataset(Dataset):
    def __init__(self, dic, in_channel, root_dir, mode="train"):
        self.keys= list(dic.keys())
        self.values=list(dic.values())
        self.root_dir = o.join(root_dir, 'tvl1_flow')
        self.root_dir_img = o.join(root_dir, 'UCF101_image')
        self.transform = transforms.Compose([
                transforms.Scale([112,112]),
                transforms.ToTensor(),
                ])
        self.transform_img = transforms.Compose([
            transforms.Scale([112,112]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        self.mode=mode
        self.in_channel = in_channel-1 # flow
        self.in_channel_img = in_channel # img
        self.img_rows=112
        self.img_cols=112

    
    def __len__(self):
        return len(self.keys)
    
    def stackimage(self):
        img_path = self.root_dir_img + '/v_'+self.video
        img_stack = torch.FloatTensor(3, self.in_channel_img,self.img_rows,self.img_cols)
        i = int(self.clips_idx)
        for j in range(self.in_channel_img):
            idx = i + j
            idx = str(idx)
            frame_idx = 'frame'+ idx.zfill(6)
            img_n = o.join(img_path, frame_idx +'.jpg')
            img=(Image.open(img_n))
            I = self.transform_img(img)
            img_stack[:,j,:,:] = I
            img.close()
        return img_stack
    
    def stackopf(self):
        name = 'v_'+self.video
        u = self.root_dir+ '/u/' + name
        v = self.root_dir+ '/v/'+ name
        
        flow = torch.FloatTensor(2,self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)


        for j in range(self.in_channel):
            idx = i + j
            idx = str(idx)
            frame_idx = 'frame'+ idx.zfill(6)
            h_image = u +'/' + frame_idx +'.jpg'
            v_image = v +'/' + frame_idx +'.jpg'
            
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)

            flow[0,j,:,:] = H
            flow[1,j,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow
    
    def __getitem__(self, idx):
        
        if self.mode == 'train':
            self.video, nb_clips = self.keys[idx].split('-')
            self.clips_idx = random.randint(1,int(nb_clips))
        elif self.mode == 'val':
            # self.video,self.clips_idx = self.keys[idx].split('-')
            self.video, nb_clips = self.keys[idx].split('-')
            self.clips_idx = random.randint(1,int(nb_clips))
        else:
            raise ValueError('There are only train and val mode')
        
        
        label = self.values[idx]
        label = int(label)-1 
        data = self.stackopf()
        image = self.stackimage()

        if self.mode == 'train':
            sample = (image,data)
        elif self.mode == 'val':
            sample = (image,data)
        else:
            raise ValueError('There are only train and val mode')
        return sample


class Motion_Image_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel,  path, ucf_list, ucf_split, root_path):

        self.root_path = root_path
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count={}
        self.in_channel = in_channel
        self.data_path = path
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video() # video filename list
        self.missingfile = set(['ApplyEyeMakeup_g04_c04'])
        # print(self.train_video.keys())
        
    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        with open(self.root_path+'/dataloader/dic/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame :
            videoname = line.split('_',1)[1].split('.',1)[0]
            n,g = videoname.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            self.frame_count[videoname]=dic_frame[line] 

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.get_val_dict()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video
            
    def val_sample19(self):
        self.dic_test_idx = {}
        #print len(self.test_video)
        for video in self.test_video:
            n,g = video.split('_',1)

            sampling_interval = int((self.frame_count[video]-self.in_channel+1)/19)
            for index in range(19):
                clip_idx = index*sampling_interval
                key = video + '-' + str(clip_idx+1)
                self.dic_test_idx[key] = self.test_video[video]
                
    def get_val_dict(self):
        self.dic_test_idx={}
        # print( self.test_video)
        for video in self.test_video:
            nb_clips = self.frame_count[video]-self.in_channel+1
            if nb_clips <=0 or video in self.missingfile:
                print("nb_clips smaller than frames required, so skip val",video, self.frame_count[video])
            else:
                key = video +'-' + str(nb_clips)
                self.dic_test_idx[key] = self.test_video[video] 
    
    def get_training_dic(self):
        self.dic_video_train={}
        # print(self.train_video)
        for video in self.train_video:
            nb_clips = self.frame_count[video]-self.in_channel+1
            if nb_clips <=0 or video in self.missingfile:
                print("nb_clips smaller than frames required, so skip train",video, self.frame_count[video])
            else:
                key = video +'-' + str(nb_clips)
                self.dic_video_train[key] = self.train_video[video] 
                            
    def train(self):
        training_set = motion_image_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train')
        print ('==> Training data :',len(training_set),' videos',training_set[1][0].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )

        return train_loader

    def val(self):
        validation_set = motion_image_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val')
        print ('==> Validation data :',len(validation_set),' frames',validation_set[1][1].size())
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

if __name__ == '__main__':
    data_loader =Motion_Image_DataLoader(BATCH_SIZE=1,num_workers=1,in_channel=10,
                                        path='/home/zdadadaz/Desktop/course/medical/data/UCF101/',
                                        ucf_list='../UCF_list/',
                                        ucf_split='01',
                                        root_path = '../'
                                        )
    train_loader,val_loader,test_video = data_loader.run()
    # print (train_loader,val_loader)
