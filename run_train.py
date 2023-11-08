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

def main():

    parser = argparse.ArgumentParser(description="A script with argparse options")

    # Add an argument for an integer option
    parser.add_argument("--runname", type=str, required=True)
    parser.add_argument("--projectname", type=str, required=True)
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


    print(arg_nottest)

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

    
    # Define the paths to the first and second directories
    first_directory = tar_dir
    second_directory = in_dir
    
    # List all files in the first directory
    for root, dirs, files in os.walk(first_directory):
        for file in files:
            if file.startswith("tile_streamorder_") and file.endswith(".png"):
                file_path = os.path.join(root, file)
                img = Image.open(file_path)
                if img.size != (64, 64):
                    # Remove the image from the first directory
                    os.remove(file_path)
    
                    # Get the image number from the filename
                    image_number = file.split('_')[-1].split('.')[0]
    
                    # Remove corresponding images in the second directory
                    for second_root, second_dirs, second_files in os.walk(second_directory):
                        for second_file in second_files:
                            if second_file.endswith(f"{image_number}.png"):
                                second_file_path = os.path.join(second_root, second_file)
                                os.remove(second_file_path)
    print("Done!")

    dataset = RGBStreamOrderDataset(input_dir=in_dir, target_dir=tar_dir, transform=transform)


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=arg_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=arg_batch_size, shuffle=True)


    # Create a DataLoader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("Data is loaded ...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    print(device)

    # Instantiate the model
    if arg_modelname == 'Unet_1':
        model = UNet_1(n_channels=18, n_classes=1).to(device)  # Change n_classes based on your output
    if arg_modelname == 'Unet_2':
        model = UNet_2(n_channels=18, n_classes=1).to(device)  # Change n_classes based on your output
    
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

            if arg_nottest:
                continue
            else:
                break

        train_loss, train_precision, train_recall, train_f1 = evaluate_model(model, train_loader, criterion, nottest=arg_nottest)
        test_loss, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, nottest=arg_nottest)

        # Print both training and test loss
        print(f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {train_loss:}, '
            f'Train Precision: {train_precision:}, Train Recall: {train_recall:}, Train F1: {train_f1:}, '
            f'Test Loss: {test_loss:}, '
            f'Test Precision: {test_precision:}, Test Recall: {test_recall:}, Test F1: {test_f1:}')
        wandb.log({"Train/train_loss": train_loss, "Train/train_precision": train_precision, "Train/train_recall": train_recall, "Train/train_f1": train_f1})
        wandb.log({"Test/test_loss": test_loss, "Test/test_precision": test_precision, "Test/test_recall": test_recall, "Test/test_f1": test_f1})
        # Save model every 10 epochs
        if (epoch + 1) % arg_savingstep == 0:
            torch.save(model.state_dict(), f'./model_epoch_{epoch+1}.pth')
            artifact = wandb.Artifact(f'model_epoch_{epoch+1}', type='model')
            artifact.add_file(f'./model_epoch_{epoch+1}.pth')
            wandb.log_artifact(artifact)
            save_comparison_figures(model, test_loader, epoch + 1, device)
            print(f'Model saved and comparison figures generated for Epoch {epoch + 1}.')

    print('Finished Training')


if __name__ == "__main__":
    main()


