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
model.load_state_dict(torch.load(artifact_dir+f'/{args.modelpath}'))



# Create the dataset
in_dir = '/root/home/rgb_data_128/'
tar_dir = '/root/home/so_data_128/'

dataset = RGBStreamOrderDataset(input_dir=in_dir, target_dir=tar_dir, transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=arg_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=arg_batch_size, shuffle=True)



def evaluate_model(model, dataloader, criterion, threshold=0.5, nottest=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("------ Evaluation --------")
    model.eval()
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Apply sigmoid function to ensure outputs are in the probability space
            probs = outputs.sigmoid()
            preds = (probs > threshold).float()  # Cast to float to perform calculations

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            precision, recall, f1 = calculate_precision_recall_f1(preds, targets.float())
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            if not nottest:
                break

    avg_loss = total_loss / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches

    return avg_loss, avg_precision, avg_recall, avg_f1




