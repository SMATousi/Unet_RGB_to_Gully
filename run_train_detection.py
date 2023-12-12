from model import *

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
        targets = gt_image > 0
        targets = targets.float()
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
