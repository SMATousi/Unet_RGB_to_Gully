from model import *
import argparse
from tqdm import tqdm


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
    parser.add_argument("--dilationpixel", type=int, default=1)
    parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")

    args = parser.parse_args()

    arg_batch_size = args.batchsize
    arg_epochs = args.epochs
    arg_runname = args.runname
    arg_projectname = args.projectname
    arg_modelname = args.modelname
    arg_savingstep = args.savingstep
    arg_dilationpixel = args.dilationpixel

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
            # config={
            # "learning_rate": 0.02,
            # "architecture": "CNN",
            # "dataset": "CIFAR-100",
            # "epochs": 20,
            # }
    )

    
    
    in_dir = '/root/home/128x128_raw/rgb_images/'
    tar_dir = '/root/home/128x128_raw/ground_truth/'

    dataset = RGBGroundTruthDataset(rgb_dir=in_dir, 
                                    gt_dir=tar_dir, 
                                    years=['0','1','2','3','4','5'], 
                                    dilation_pixels=arg_dilationpixel)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=arg_batch_size,num_workers=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=arg_batch_size,num_workers=8, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SingleResNetLSTMUNet(num_classes=1, hidden_dim=512, lstm_layers=10)
    model = model.to(device)  # Assuming you have a device (like 'cuda' or 'cpu')

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = arg_epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (rgb_images, gt_image) in enumerate(train_loader):
            # Move data to the device
            rgb_images = [img.to(device) for img in rgb_images]
            gt_image = gt_image.to(device)
            gt_image = gt_image.unsqueeze(1)
            gt_image = gt_image.float()

            # Forward pass
            outputs = model(rgb_images)
            # targets = gt_image > 0
            # targets = targets.float()
            loss = criterion(outputs, gt_image)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
#             print("done")
            
            if arg_nottest:
                continue
            else:
                break

        if (epoch + 1) % arg_savingstep == 0:

            train_loss, train_precision, train_recall, train_f1 = evaluate_model_det(model, 
                                                                                 train_loader, 
                                                                                 criterion, 
                                                                                 nottest=arg_nottest)

            test_loss, test_precision, test_recall, test_f1 = evaluate_model_det(model, 
                                                                             test_loader, 
                                                                             criterion, 
                                                                             nottest=arg_nottest)

            # Print both training and test loss
            print(f'Epoch {epoch+1}/{num_epochs}, '
                f'Train Loss: {train_loss:}, '
                f'Train Precision: {train_precision:}, Train Recall: {train_recall:}, Train F1: {train_f1:}, '
                f'Test Loss: {test_loss:}, '
                f'Test Precision: {test_precision:}, Test Recall: {test_recall:}, Test F1: {test_f1:}')
            wandb.log({"Train/train_loss": train_loss, "Train/train_precision": train_precision, "Train/train_recall": train_recall, "Train/train_f1": train_f1})
            wandb.log({"Test/test_loss": test_loss, "Test/test_precision": test_precision, "Test/test_recall": test_recall, "Test/test_f1": test_f1})

    print('Finished Training')
    

if __name__ == "__main__":
    main()
