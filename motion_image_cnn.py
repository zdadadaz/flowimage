# -*- coding: utf-8 -*-
import numpy as np
import pickle
from PIL import Image
import time
import tqdm
import shutil
from random import randint
import argparse
import os 
import pathlib

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from network import *
import dataloader
from unet import UNet3D

# <!-- os.environ["CUDA_VISIBLE_DEVICES"] = "1" -->

parser = argparse.ArgumentParser(description='UCF101 motion stream on resnet101')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--modelName', default='noName', type=str, metavar='N', help='model name')

def main():
    global arg
    arg = parser.parse_args()
    print (arg)

    #Prepare DataLoader
    data_loader = dataloader.Motion_Image_DataLoader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path='./../../data/UCF101',
                        ucf_list='./UCF_list/',
                        ucf_split='01',
                        in_channel=32,
                        root_path = './'
                        )
    
    train_loader,test_loader, test_video = data_loader.run()
    #Model 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Motion_image_CNN(
                        # Data Loader
                        model_name= arg.modelName,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        # Utility
                        start_epoch=arg.start_epoch,
                        resume=arg.resume,
                        evaluate=arg.evaluate,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        channel = 32,
                        test_video=test_video,
                        device = device
                        )
    #Training
    model.run()

class Motion_image_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, channel,test_video, model_name, device):
        self.model_name = model_name
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        # self.train_loader=train_loader
        # self.test_loader=test_loader
        self.dataloader={'train':train_loader, 'test':test_loader}
        self.best_prec1=0
        self.channel=channel
        self.test_video=test_video
        self.device = device
        self.output = './flow_output/'+model_name
        pathlib.Path(self.output).mkdir(parents=True, exist_ok=True)
    
        
    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = UNet3D(in_channels=3, out_channels=2)
        #print self.model
        if self.device.type == "cuda":
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        #Loss function and optimizer
        # self.criterion = nn.binary_cross_entropy_with_logits().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch=0
            prec1, val_loss = self.validate_1epoch()
            return
        
    def run(self):
        self.build_model()
        with open(os.path.join(self.output, "log.csv"), "a") as qq:
            epoch_resume = self.resume_and_evaluate()
            if self.start_epoch != 0:
                qq.write("Resuming from epoch {}\n".format(self.start_epoch))
            else:
                qq.write("Starting run from scratch\n")
                
            cudnn.benchmark = True
            for self.epoch in range(self.start_epoch, self.nb_epochs):
                print("Epoch #{}".format(self.epoch), flush=True)
                for phase in ['train', 'val']:
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_max_memory_allocated(i)
                        torch.cuda.reset_max_memory_cached(i)
                    info = self.train_1epoch(phase = phase, dataloader=self.dataloader[phase])
                    f.write("{},{},{},{},{}".format(self.epoch,
                                                        phase,
                                                        info['Loss'],
                                                        info['Epoch_loss'],
                                                        info['lr']
                                                        ))
                    is_best = info['Epoch_loss'] < self.best_prec1
                    # save model
                    if is_best:
                        self.best_prec1 = prec1
                        with open(self.output+'/motion_video_preds.pickle','wb') as f:
                            pickle.dump(self.dic_video_level_preds,f)
                        f.close() 
                    
                save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                    'optimizer' : self.optimizer.state_dict()
                },is_best,self.output+'/checkpoint.pth.tar',self.output + '/model_best.pth.tar')
    
    def train_1epoch(self, phase, dataloader ):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        total=0
        n = 0
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        with torch.set_grad_enabled(phase=="train"):
            with tqdm(total=len(dataloader)) as pbar:
                for i, (data,label) in enumerate(dataloader):
        
                    # measure data loading time
                    data_time.update(time.time() - end)
                    
                    input_var = data.to(self.device)
                    target_var = label.to(self.device)
        
                    # compute output
                    output = self.model(input_var)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(output[:,:,:-1,:,:], target_var,reduction="sum")
                   
                    # compute gradient and do SGD step
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
            
                    # calculate loss
                    total += loss.item()
                    n += label.size(0)
                    epoch_loss = total/n/112/112
                    losses.update(loss.item(), data.size(0))
                    
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    
                    pbar.set_postfix_str("loss: {:.4f}".format(epoch_loss))
                    pbar.update()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Epoch_loss':[round(epoch_loss,5)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        return info
        # record_info(info, 'flow_output/'+self.model_name+'/opf_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))

        batch_time = AverageMeter()
        losses = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys,data,label) in enumerate(progress):
            # label = label.cuda(async=True)
            # data_var = Variable(data, volatile=True).cuda(async=True)
            # label_var = Variable(label, volatile=True).cuda(async=True)

            input_var = data.to(self.device)
            target_var = label.to(self.device)

            # compute output
            output = self.model(data_var)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j].split('-',1)[0] # ApplyMakeup_g01_c01
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]
                    
        #Frame to video level accuracy
        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(video_loss,5)],
                'Prec@1':[round(video_top1,3)],
                'Prec@5':[round(video_top5,3)]
                }
        record_info(info, 'record/motion/opf_test.csv','test')
        return video_top1, video_loss

if __name__=='__main__':
    main()
