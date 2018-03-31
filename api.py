import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import numpy as np
import cv2
from networks import unet
from sys import argv


checkpoints_directory_unet="checkpoints_unet"

script, img_path = argv
# python api.py /path/to/image/imagename.extension 
#will give output in the folder containing script as Output_unet.png

checkpoints_unet= os.listdir(checkpoints_directory_unet)
checkpoints_unet.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
model_unet = torch.load(checkpoints_directory_unet+'/'+checkpoints_unet[-1])

model_unet.eval()
if torch.cuda.is_available(): #use gpu if available
    model_unet.cuda() 

image = cv2.imread(img_path)
orig_width,orig_height=image.shape[0],image.shape[1]
input_unet = image

input_unet=cv2.resize(input_unet,(256,256), interpolation = cv2.INTER_CUBIC)
input_unet= input_unet.reshape((256,256,3,1))

input_unet = input_unet.transpose((3, 2, 0, 1))

input_unet.astype(float)
input_unet=input_unet/255

input_unet = torch.from_numpy(input_unet)


input_unet=input_unet.type(torch.FloatTensor)

if torch.cuda.is_available(): #use gpu if available
    input_unet = Variable(input_unet.cuda()) 
else:
	input_unet = Variable(input_unet)


out_unet = model_unet(input_unet)


out_unet =  out_unet.cpu().data.numpy()


out_unet = out_unet*255


out_unet = out_unet.transpose((2,3,0,1))
out_unet= out_unet.reshape((256,256,1))
out_unet= cv2.resize(out_unet,(orig_height,orig_width), interpolation=cv2.INTER_CUBIC)

cv2.imwrite("Output_unet.png", out_unet)
