import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob
import wandb
from model import *
import argparse
import random
import numpy as np

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser(description="A script with argparse options")

# Add an argument for an integer option
parser.add_argument("--runname", type=str, required=False)
parser.add_argument("--projectname", type=str, required=False)
parser.add_argument("--filepath", type=str, required=False)
parser.add_argument("--modelpath", type=str, required=False)
parser.add_argument("--modelname", type=str, required=True)
parser.add_argument("--batchsize", type=int, default=4)
parser.add_argument("--savingstep", type=int, default=10)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")

args = parser.parse_args()

arg_batch_size = args.batchsize
arg_epochs = args.epochs
arg_runname = args.runname
arg_projectname = args.projectname
arg_modelname = args.modelname
arg_savingstep = args.savingstep

if args.nottest:
    arg_nottest = True 
else:
    arg_nottest = False

run = wandb.init()

artifact = run.use_artifact(args.filepath, type='model')
artifact_dir = artifact.download()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if arg_modelname == 'Unet':
        model = UNet(n_channels=18, n_classes=1).to(device)  # Change n_classes based on your output
    

print(artifact_dir)
model.load_state_dict(torch.load(artifact_dir+f'/'{args.modelpath}))


print("S!")



