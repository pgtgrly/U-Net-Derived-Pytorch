'''
This script does not work yet as MaxPool@D with index is not supported by ONNX
'''
import torch.nn as nn
import torch
import torch.onnx
from torch.autograd import Variable
import os

checkpoints_directory_unet="checkpoints_unet"
checkpoints_unet= os.listdir(checkpoints_directory_unet)
model_unet = torch.load(checkpoints_directory_unet+'/'+checkpoints_unet[-1],map_location={'cuda:0': 'cpu'})
print("using " + checkpoints_unet[-1])
model_unet.eval()
dummy_input = Variable(torch.randn(1, 3, 256, 256))
torch.onnx.export(model_unet, dummy_input, "ONNX/unet_derived.onnx")