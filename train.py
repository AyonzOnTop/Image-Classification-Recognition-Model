import json
import PIL
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from collections import OrderedDict
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import toolkit
import argparse
import toolkit


parser = argparse.ArgumentParser(description = 'train.py')

parser.add_argument('data_dir', action = "store", default = "./flowers/")
parser.add_argument('--gpu', dest = 'gpu', action = 'store', default = "gpu")
parser.add_argument('--learning_rate', dest = 'learning_rate', action = 'store', default = 0.008)
parser.add_argument('--save_dir', dest = 'save_dir', action = 'store', default = "./checkpoint.pth")
parser.add_argument('--epochs', dest = 'epochs', action = 'store', type = int, default = 3)
parser.add_argument('--dropout', dest = 'dropout', action = 'store', default = 0.5)
parser.add_argumemt('--arch', dest = 'arch', action = 'store', default = "vgg13", type = str)
parser.add_argument('--hidden_units', dest = 'hidden_units', action = 'store', default = 500)

pa = parser.parse_args()
root = pa.data_dir
gpu = pa.gpu
lr = pa.learning_rate
path = pa.save_dir
pa = pa.epochs
structure = pa.dropout
hidden_layer1 = pa.hidden_units

def main():
    trainloader,validatnloader,testloader = toolkit.dataloader(source)
    model, optimizer, criterion = toolkit.tom_network(hidden_size)
    toolkit.train_model(model,trainloader, validatnloader, epochs, print_every, criterion, optimizer )
    toolkit.save_checkpoint(classifier, epochs,model, optimizer )
    print('complete')
    
if __name__= "__main__":
    main()
    

main()
    
    