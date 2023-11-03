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

def main():

    parser = argparse.ArgumentParser(description="A script with argparse options")

    # Add an argument for an integer option
    parser.add_argument("--runname", type=str, required=True)
    parser.add_argument("--projectname", type=str, required=True)
    parser.add_argument("--modelname", type=str, required=True)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")

    args = parser.parse_args()

    arg_batch_size = args.batchsize
    arg_epochs = args.epochs
    arg_runname = args.runname
    arg_projectname = args.projectname
    arg_modelname = args.modelname

    args = parser.parse_args()

    
    wandb.init(
        # set the wandb project where this run will be logged
        project=arg_projectname, name=arg_runname
        
        # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": 0.02,
    #     "architecture": "CNN",
    #     "dataset": "CIFAR-100",
    #     "epochs": 20,
    #     }
    )


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

    train_loader = DataLoader(train_dataset, batch_size=arg_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=arg_batch_size, shuffle=False)


    # Create a DataLoader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("Data is loaded ...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    print(device)

    # Instantiate the model
    if arg_modelname == 'Unet':
        model = UNet(n_channels=18, n_classes=1).to(device)  # Change n_classes based on your output
    
    # criterion = nn.MSELoss()  # Change loss function based on your task
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce learning rate every 5 epochs by a factor of 0.1

    print("Model is created ...")

    # Training loop
    num_epochs = arg_epochs
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

            if not args.nottest:
                break

        train_loss, train_precision, train_recall, train_f1 = evaluate_model(model, train_loader, criterion, nottest=args.nottest)
        test_loss, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, nottest=args.nottest)

        # Print both training and test loss
        print(f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {train_loss:}, '
            f'Train Precision: {train_precision:}, Train Recall: {train_recall:}, Train F1: {train_f1:}, '
            f'Test Loss: {test_loss:}, '
            f'Test Precision: {test_precision:}, Test Recall: {test_recall:}, Test F1: {test_f1:}')
        wandb.log({"Train/train_loss": train_loss, "Train/train_precision": train_precision, "Train/train_recall": train_recall, "Train/train_f1": train_f1})
        wandb.log({"Test/test_loss": test_loss, "Test/test_precision": test_precision, "Test/test_recall": test_recall, "Test/test_f1": test_f1})
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'./model_epoch_{epoch+1}.pth')
            artifact = wandb.Artifact(f'./model_epoch_{epoch+1}', type='model')
            artifact.add_file(f'./model_epoch_{epoch+1}.pth')
            wandb.log(artifact)
            save_comparison_figures(model, test_loader, epoch + 1, device)
            print(f'Model saved and comparison figures generated for Epoch {epoch + 1}.')

    print('Finished Training')





