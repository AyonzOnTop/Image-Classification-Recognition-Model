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

parser =argparse.ArgumentParser(description = 'predict.py')

parser.add_argument(input)
parser.add_argument('--dir', action = 'store', dest = 'data_dir', default = "./flower/")
parser.add_argument('checkpoint', action = 'store', nargs = '?', type = str, default = './checkpoint.pth')
parser.add_argument('--top_k', action = 'store', dest = 'top_k', default = 5, type = int)
parser.add_argument('--category', action = 'store', dest = 'category_names', default = "cat_to_name.json")
parser.add_argument('--gpu', action = 'store', dest = "gpu", default = "gpu")

pa = parser.parse_args()
path_image = pa.input
number_of_outputs = pa.top_k
device = pa.gpu

path =pa.checkpoint



def main():
    model = toolkit.load_chkpt(path)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
    probs_top, top_classes =  toolkit.predict(path_image, model, number_of_outputs)
    labels = [cat_to_name[str(index+1)] for index in np.array(prob_tops[1][0])]
    problty = np.array(probs_top[0][0])
    
    i=0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], problty[i]))
        i += 1
    print('prediction done')

if __name__ == "__main__":
    main()

