import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob
import wandb
import random
import numpy as np
from unet_parts import *

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)


class RGBStreamOrderDataset(Dataset):
    def __init__(self, input_dir, target_dir, augment=False):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.augment = augment

        # Augmentation transforms
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.RandomCrop((224, 224)),  # Example size, adjust as needed
            transforms.ToTensor()
        ])

        # Standard transforms (without augmentation)
        self.standard_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        self.input_files = glob.glob(os.path.join(input_dir, 'tile_1_*.png'))
        self.input_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        base_filename = self.input_files[idx].rsplit('_', 1)[1]

        input_images = [Image.open(f"{self.input_dir}/tile_{channel}_{base_filename}") for channel in range(1, 7)]
        
        if self.augment:
            input_stack = [self.augment_transforms(img) for img in input_images]
        else:
            input_stack = [self.standard_transforms(img) for img in input_images]

        input_tensor = torch.cat(input_stack, dim=0)

        stream_order_filename = os.path.join(self.target_dir, f"tile_streamorder_{base_filename}")
        stream_order_image = Image.open(stream_order_filename)
        
        if self.augment:
            stream_order_image = self.augment_transforms(stream_order_image)
        else:
            stream_order_image = self.standard_transforms(stream_order_image)

        return input_tensor, stream_order_image



# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

class UNet_1(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate=0.5):
        super(UNet_1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 512)
        self.up1 = DoubleConv(1024, 256)
        self.up2 = DoubleConv(512, 128)
        self.up3 = DoubleConv(256, 64)
        self.up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        self.dropout = nn.Dropout(dropout_rate)



    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(torch.cat([x4, x5], dim=1))
        x = self.up2(torch.cat([x3, x], dim=1))
        x = self.up3(torch.cat([x2, x], dim=1))
        x = self.up4(torch.cat([x1, x], dim=1))
        logits = self.outc(x)
        return logits


class UNet_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def calculate_precision_recall_f1(preds, targets):
    # Convert boolean tensors to float for calculations
    preds = preds.float()
    targets = targets.float()

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = (preds * targets).sum()
    FP = ((1 - targets) * preds).sum()
    FN = (targets * (1 - preds)).sum()

    # Calculate Precision, Recall, and F1 score
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1



def evaluate_model(model, dataloader, criterion, threshold=0.1, nottest=True):
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
            # probs = outputs.sigmoid()
            preds = (outputs > threshold).float()  # Cast to float to perform calculations

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


def save_comparison_figures(model, dataloader, epoch, device, save_dir='comparison_figures', num_samples=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    sample_count = 0
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))  # 5 is an arbitrary height multiplier for visibility

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if sample_count >= num_samples:
                break  # Break if we have already reached the desired number of samples

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5  # Apply threshold to get binary mask

            # for i in range(inputs.shape[0]):  # Loop over each image in the batch
            #     if sample_count >= num_samples:
            #         break  # Break if we have already reached the desired number of samples

            # Access the i-th sample in the batch for both ground truth and prediction
            gt_mask = targets[sample_count].squeeze().cpu().numpy()  # Convert to NumPy array for plotting
            pred_mask = preds[sample_count].squeeze().cpu().numpy()

            axs[sample_count, 0].imshow(gt_mask, cmap='gray')
            axs[sample_count, 0].set_title(f'Sample {sample_count + 1} Ground Truth')
            axs[sample_count, 0].axis('off')

            axs[sample_count, 1].imshow(pred_mask, cmap='gray')
            axs[sample_count, 1].set_title(f'Sample {sample_count + 1} Prediction')
            axs[sample_count, 1].axis('off')

            sample_count += 1  # Increment the sample counter

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.suptitle(f'Comparison for Epoch {epoch}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Adjust the top spacing to accommodate the suptitle

    figure_path = os.path.join(save_dir, f'epoch_{epoch}_comparison.png')
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
    plt.savefig(figure_path)
    wandb.log({f'Images/epoch_{epoch}': wandb.Image(f'{save_dir}/epoch_{epoch}_comparison.png')})

    plt.close()


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

        # Original encoder: Use ResNet as the encoder with pretrained weights
        original_encoder = models.resnet18(pretrained=True)

        # Modify the first convolution layer to accept 18-channel input
        self.first_conv = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Copy weights from the original first layer (for the first 3 channels)
        self.first_conv.weight.data[:, :3] = original_encoder.conv1.weight.data

        # Use the rest of the layers from the original encoder
        self.encoder_layers = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *list(original_encoder.children())[4:-2]  # Exclude the original first conv layer and the fully connected layers
        )
        
        self.encoder = nn.Sequential(
            self.first_conv,
            self.encoder_layers
        )

        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder: A simple decoder with transpose convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)  # Output depth map
        )

    def forward(self, x):
        # Forward pass through the encoder
        x = self.encoder(x)

        # Forward pass through the decoder
        x = self.decoder(x)
        
        return x
