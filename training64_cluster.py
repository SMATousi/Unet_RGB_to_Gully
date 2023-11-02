import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import Precision, Recall, F1Score
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob
import wandb

class RGBStreamOrderDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        # List all files in the input directory
        self.input_files = glob.glob(os.path.join(input_dir, 'tile_1_*.png'))
        # Sort the files to ensure alignment
        self.input_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Base filename without the channel and extension
        base_filename = self.input_files[idx].rsplit('_', 1)[0]

        # Stack the RGB images to form an input with 18 channels
        input_images = [Image.open(f"{base_filename}_{channel}.png") for channel in range(1, 7)]
        input_stack = [self.transform(img) for img in input_images]
        input_tensor = torch.cat(input_stack, dim=0)  # Concatenate along the channel dimension

        # Corresponding stream order file
        stream_order_filename = os.path.join(self.target_dir, f"tile_streamorder_{base_filename.split('_')[-1]}.png")
        stream_order_image = Image.open(stream_order_filename)
        
        if self.transform is not None:
            stream_order_image = self.transform(stream_order_image)
        
        return input_tensor, stream_order_image



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
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

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(torch.cat([x4, x5], dim=1))
        x = self.up2(torch.cat([x3, x], dim=1))
        x = self.up3(torch.cat([x2, x], dim=1))
        x = self.up4(torch.cat([x1, x], dim=1))
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



def evaluate_model(model, dataloader, criterion, threshold=0.5):
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

            break

    avg_loss = total_loss / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches

    return avg_loss, avg_precision, avg_recall, avg_f1



def save_comparison_figures(model, dataloader, epoch, device, save_dir='comparison_figures', num_samples=5):
    model.eval()
    sample_count = 0
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2))  # Adjust the figure size as needed

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = (outputs.sigmoid() > 0.5).float()  # Get the predictions as 0 or 1

            for i in range(inputs.size(0)):
                if sample_count >= num_samples:
                    break

                gt_mask = targets[i].squeeze().cpu()  # Assuming the unwanted dimension is already removed
                pred_mask = preds[i].squeeze().cpu()  # Assuming the unwanted dimension is already removed

                # Ground truth mask subplot
                axs[sample_count, 0].imshow(gt_mask, cmap='gray')
                axs[sample_count, 0].set_title(f'Epoch {epoch} - GT Sample {sample_count + 1}')
                axs[sample_count, 0].axis('off')

                # Predicted mask subplot
                axs[sample_count, 1].imshow(pred_mask, cmap='gray')
                axs[sample_count, 1].set_title(f'Epoch {epoch} - Pred Sample {sample_count + 1}')
                axs[sample_count, 1].axis('off')

                sample_count += 1

            if sample_count >= num_samples:
                break

    plt.tight_layout()
    fig.suptitle(f'Comparison for Epoch {epoch}', fontsize=16)
    plt.subplots_adjust(top=0.9)

    # Save the figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/epoch_{epoch}_comparison.png')
    plt.close()

# Initializing the WANDB

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Gully-detection-64-Unet", name="batch-4-TestRun"
    
#     # track hyperparameters and run metadata
# #     config={
# #     "learning_rate": 0.02,
# #     "architecture": "CNN",
# #     "dataset": "CIFAR-100",
# #     "epochs": 20,
# #     }
# )

if not os.path.exists('comparison_figures'):
    os.makedirs('comparison_figures')


# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # This also scales the image to [0, 1]
])

# Create the dataset
in_dir = '/root/home/rgb_data_64/rgb_data_64/'
tar_dir = '/root/home/rgb_data_64/so_data_64/'

dataset = RGBStreamOrderDataset(input_dir=in_dir, target_dir=tar_dir, transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# Create a DataLoader
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Data is loaded ...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print(device)

# Instantiate the model
model = UNet(n_channels=18, n_classes=1).to(device)  # Change n_classes based on your output
criterion = nn.MSELoss()  # Change loss function based on your task
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Model is created ...")

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    print("--------- Training ------------")
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        break

    train_loss, train_precision, train_recall, train_f1 = evaluate_model(model, train_loader, criterion)
    test_loss, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion)

    # Print both training and test loss
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss:}, '
          f'Train Precision: {train_precision:}, Train Recall: {train_recall:}, Train F1: {train_f1:}, '
          f'Test Loss: {test_loss:}, '
          f'Test Precision: {test_precision:}, Test Recall: {test_recall:}, Test F1: {test_f1:}')

    # Save model every 10 epochs
    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), f'./model_epoch_{epoch+1}.pth')
        save_comparison_figures(model, test_loader, epoch + 1, device)
        print(f'Model saved and comparison figures generated for Epoch {epoch + 1}.')

print('Finished Training')
