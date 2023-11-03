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
from model.py import *
import argparse
import random
import numpy as np

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)