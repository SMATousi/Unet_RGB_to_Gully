from model import *

in_dir = '/root/home/128x128_raw/rgb_images/'
tar_dir = '/root/home/128x128_raw/ground_truth/'

dataset = RGBGroundTruthDataset(rgb_dir=in_dir, gt_dir=tar_dir, years=['0','1','2','3','4','5'], dilation_pixels=1)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2,num_workers=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2,num_workers=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SingleResNetLSTMUNet(num_classes=1, hidden_dim=6, lstm_layers=10)
model = model.to(device)  # Assuming you have a device (like 'cuda' or 'cpu')

# Loss Function
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (rgb_images, gt_image) in enumerate(train_loader):
        # Move data to the device
        rgb_images = [img.to(device) for img in rgb_images]
        gt_image = gt_image.to(device)
        gt_image = gt_image.unsqueeze(1)

        # Forward pass
        outputs = model(rgb_images)
        # targets = gt_image > 0
        # targets = targets.float()
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}')
            running_loss = 0.0

print('Finished Training')
