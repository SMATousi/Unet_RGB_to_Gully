import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

import os
import torch
import matplotlib.pyplot as plt
# import pytorch_lightning as pl
# import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from model import *
import wandb
import argparse

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)


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

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


root = "."
SimpleOxfordPetDataset.download(root)



# init train, val, test sets
train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")
test_dataset = SimpleOxfordPetDataset(root, "test")

# It is a good practice to check datasets don`t intersects with each other
assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=arg_batch_size, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=arg_batch_size, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=arg_batch_size, shuffle=False, num_workers=4)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet_1(n_channels=3, n_classes=1, dropout_rate=0.5).to(device)



criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = arg_epochs

for epoch in range(num_epochs):
    print("--------- Training ------------")
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_dataloader):
#         inputs, targets = inputs.to(device), targets.to(device)
        inputs = batch["image"].to(device)
        targets = batch["mask"].to(device)
        
        inputs = inputs.float()
        targets = targets.float()

        # Forward pass
        outputs = model(inputs)
#         targets = targets > 0
#         targets = targets.float()
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

#         if arg_nottest:
#             continue
#         else:
#             break

    if (epoch + 1) % arg_savingstep == 0:

        train_loss, train_precision, train_recall, train_f1 = evaluate_model_pet(model, train_dataloader, criterion, nottest=arg_nottest)
        test_loss, test_precision, test_recall, test_f1 = evaluate_model_pet(model, test_dataloader, criterion, nottest=arg_nottest)

        # Print both training and test loss
        print(f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {train_loss:}, '
            f'Train Precision: {train_precision:}, Train Recall: {train_recall:}, Train F1: {train_f1:}, '
            f'Test Loss: {test_loss:}, '
            f'Test Precision: {test_precision:}, Test Recall: {test_recall:}, Test F1: {test_f1:}')
        wandb.log({"Train/train_loss": train_loss, "Train/train_precision": train_precision, "Train/train_recall": train_recall, "Train/train_f1": train_f1})
        wandb.log({"Test/test_loss": test_loss, "Test/test_precision": test_precision, "Test/test_recall": test_recall, "Test/test_f1": test_f1})


