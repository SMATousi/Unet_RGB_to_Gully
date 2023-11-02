import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import Precision, Recall, F1Score
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
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


def evaluate_model(model, dataloader, criterion, threshold=0.5):
    print("-------- Evaluation -----------")
    model.eval()
    total_loss = 0

    # Define metrics
    precision_metric = Precision(threshold=threshold).to(device)
    recall_metric = Recall(threshold=threshold).to(device)
    f1_metric = F1Score(threshold=threshold).to(device)

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Convert outputs to binary mask
            preds = outputs.sigmoid() > threshold  # Use sigmoid if the output is not a probability yet

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            # Update metrics
            precision_metric.update(preds.int(), targets.int())
            recall_metric.update(preds.int(), targets.int())
            f1_metric.update(preds.int(), targets.int())

    # Compute the metrics
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1_score = f1_metric.compute()

    return total_loss / len(dataloader.dataset), precision.item(), recall.item(), f1_score.item()


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

        # break

    train_loss, train_precision, train_recall, train_f1 = evaluate_model(model, train_loader, criterion)
    test_loss, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion)

    # Print both training and test loss
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss:}, '
          f'Train Precision: {train_precision:}, Train Recall: {train_recall:}, Train F1: {train_f1:}, '
          f'Test Loss: {test_loss:}, '
          f'Test Precision: {test_precision:}, Test Recall: {test_recall:}, Test F1: {test_f1:}')

    # Save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'./model_epoch_{epoch+1}.pth')

print('Finished Training')
