import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from skimage import io, transform
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
from tensorboardX import SummaryWriter
import time
import glob
import pandas as pd

from networks import unet
from data_augment import unet_augment

writer = SummaryWriter()

batch_size = 10 #mini-batch size
n_iters = 50000 #total iterations
learning_rate = 0.01
train_directory="train"
validation_directory="validation"
checkpoints_directory_unet="checkpoints_unet"
optimizer_checkpoints_directory_unet="optimizer_checkpoints_unet"
graphs_unet_directory="graphs_unet"
validation_batch_size=1
threshold=128

class ImageDataset(Dataset): #Defining the class to load datasets
    def __init__(self, input_dir='train',transform=None):
        self.input_dir = os.path.join("data/", input_dir)        
        self.transform = transform
        self.dirlist = os.listdir(self.input_dir)
        self.dirlist.sort()

    def __len__ (self):
        return len(os.listdir(self.input_dir))

    def __getitem__(self,idx):

        img_id= self.dirlist[idx]
        
        image=cv2.imread(os.path.join(self.input_dir,img_id, "images", img_id + ".png"))
        
        input_net_1=image
        input_net_2=image 
        image=cv2.resize(image,(256,256), interpolation = cv2.INTER_CUBIC)
        image= image.reshape((256,256,3))
        input_net_2=cv2.resize(input_net_2,(128,128), interpolation = cv2.INTER_CUBIC)



        mask_path = glob.glob(os.path.join(self.input_dir,img_id) + "/*.png")     
        no_of_masks = int(mask_path[0].split("_")[1])

        masks=cv2.imread(mask_path[0],0)
        masks=cv2.resize(masks,(256,256), interpolation = cv2.INTER_CUBIC)
        masks= masks.reshape((256,256,1))                                                                             

        sample = {'image': image, 'masks': masks}  

        if self.transform:
            sample=unet_augment(sample,vertical_prob=0.5,horizontal_prob=0.5)
                    
        #As transforms do not involve random crop, number of masks must stay the same
        sample['count'] = no_of_masks
        sample['image']= sample['image'].transpose((2, 0, 1))#The convolution function in pytorch expects data in format (N,C,H,W) N is batch size , C are channels H is height and W is width. here we convert image from (H,W,C) to (C,H,W)
        sample['masks']= sample['masks'].reshape((256,256,1)).transpose((2, 0, 1))

        sample['image'].astype(float)
        sample['image']=sample['image']/255 #image being rescaled to contain values between 0 to 1 for BCE Loss
        sample['masks'].astype(float)
        sample['masks']=sample['masks']/255

        
        return sample

train_dataset=ImageDataset(input_dir=train_directory,transform=True) #Training Dataset
validation_dataset=ImageDataset(input_dir=validation_directory,transform=True) #Validation Dataset

num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                          batch_size=validation_batch_size, 
                                          shuffle=False)


model=unet()
iteri=0
iter_new=0 

#checking if checkpoints exist to resume training and create it if not
if os.path.exists(checkpoints_directory_unet) and len(os.listdir(checkpoints_directory_unet)):
    checkpoints = os.listdir(checkpoints_directory_unet)
    checkpoints.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
    if torch.cuda.is_available():
    	model=torch.load(checkpoints_directory_unet+'/'+checkpoints[-1]) #changed to checkpoints
    else:
    	model=torch.load(checkpoints_directory_unet+'/'+checkpoints[-1],map_location='cpu')
    	
    iteri=int(re.findall(r'\d+',checkpoints[-1])[0]) # changed to checkpoints
    iter_new=iteri
    print("Resuming from iteration " + str(iteri))
elif not os.path.exists(checkpoints_directory_unet):
    os.makedirs(checkpoints_directory_unet)

if not os.path.exists(graphs_unet_directory):
    os.makedirs(graphs_unet_directory)

if torch.cuda.is_available(): #use gpu if available
    model.cuda() 

criterion=nn.BCELoss()  #Loss Class #BCE Loss has been used here to determine if the pixel belogs to class or not.(This is the case of segmentation of a single class)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) #optimizer class
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)# this will decrease the learning rate by factor of 0.1 every 10 epochs
                                                                              
                                                                              # https://discuss.pytorch.org/t/can-t-import-torch-optim-lr-scheduler/5138/6 

if  os.path.exists(optimizer_checkpoints_directory_unet) and len(os.listdir(optimizer_checkpoints_directory_unet)):
    checkpoints = os.listdir(optimizer_checkpoints_directory_unet)
    checkpoints.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
    optimizer.load_state_dict(torch.load(optimizer_checkpoints_directory_unet+'/'+checkpoints[-1])) 
    print("Resuming Optimizer from iteration " + str(iteri))
elif not os.path.exists(optimizer_checkpoints_directory_unet):
    os.makedirs(optimizer_checkpoints_directory_unet)


beg=time.time() #time at the beginning of training
print("Training Started!")
for epoch in range(num_epochs):
    print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")
    for i,datapoint in enumerate(train_loader):
        datapoint['image']=datapoint['image'].type(torch.FloatTensor) #typecasting to FloatTensor as it is compatible with CUDA
        datapoint['masks']=datapoint['masks'].type(torch.FloatTensor)

       

        if torch.cuda.is_available(): #move to gpu if available
                image = Variable(datapoint['image'].cuda()) #Converting a Torch Tensor to Autograd Variable
                masks = Variable(datapoint['masks'].cuda())
                
        else:
                image = Variable(datapoint['image'])
                masks = Variable(datapoint['masks'])
                

        optimizer.zero_grad()  #https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
        outputs = model(image)
        loss = criterion(outputs, masks)
        loss.backward() #Backprop
        optimizer.step()    #Weight update
        writer.add_scalar('Training Loss',loss.data[0], iteri)
        iteri=iteri+1
        if iteri % 10 == 0 or iteri==1:
            # Calculate Accuracy         
            validation_loss = 0
            total = 0
            # Iterate through validation dataset
            for j,datapoint_1 in enumerate(validation_loader): #for validation
                datapoint_1['image']=datapoint_1['image'].type(torch.FloatTensor)
                datapoint_1['masks']=datapoint_1['masks'].type(torch.FloatTensor)

                if torch.cuda.is_available():
                    input_image_1 = Variable(datapoint_1['image'].cuda())
                    output_image_1 = Variable(datapoint_1['masks'].cuda())

                else:
                    input_image_1 = Variable(datapoint_1['image'])
                    output_image_1 = Variable(datapoint_1['masks'])

                
                # Forward pass only to get logits/output        
                outputs_1 = model(input_image_1)
                validation_loss += criterion(outputs_1, output_image_1).data[0]
                total+=datapoint_1['masks'].size(0)
            validation_loss=validation_loss
            writer.add_scalar('Validation Loss',validation_loss, iteri) 
            # Print Loss
            time_since_beg=(time.time()-beg)/60
            print('Iteration: {}. Loss: {}. Validation Loss: {}. Time(mins) {}'.format(iteri, loss.data[0], validation_loss,time_since_beg))
        if iteri % 500 ==0:
            torch.save(model,checkpoints_directory_unet+'/model_iter_'+str(iteri)+'.pt')
            torch.save(optimizer.state_dict(),optimizer_checkpoints_directory_unet+'/model_iter_'+str(iteri)+'.pt')
            print("model and optimizer saved at iteration : "+str(iteri))
            writer.export_scalars_to_json(graphs_unet_directory+"/all_scalars_"+str(iter_new)+".json") #saving loss vs iteration data to be used by visualise.py
    scheduler.step()            
writer.close()
