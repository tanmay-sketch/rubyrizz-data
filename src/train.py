import sys
import os
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import RubiksCubeDataset, transform
from model import ImprovedCNN
from loss import yolo_loss
from wandb_setup import initialize_wandb, log_metrics, finish_wandb

def collate_fn(batch):
    images, boxes, labels = zip(*batch)

    # Stack images
    images = torch.stack(images, 0)

    # Concatenate all bounding boxes and labels
    max_num_boxes = max(box.size(0) for box in boxes)

    padded_boxes = torch.zeros(len(boxes), max_num_boxes, 4)
    padded_labels = torch.zeros(len(labels), max_num_boxes).long()

    for i in range(len(boxes)):
        num_boxes = boxes[i].size(0)
        if num_boxes > 0:
            padded_boxes[i, :num_boxes] = boxes[i]
            padded_labels[i, :num_boxes] = labels[i]
        else:
            # Handle images without any bounding boxes
            padded_boxes[i, :1] = -1  # Use -1 to indicate no bounding boxes
            padded_labels[i, :1] = -1

    return images, padded_boxes, padded_labels

def main():
    # Load data.yaml
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_config_path = os.path.join(base_path, 'data', 'data.yaml')
    with open(data_config_path, 'r') as file:
        data_config = yaml.safe_load(file)

    # Use absolute paths for the dataset directories
    train_img_dir = os.path.join(base_path, data_config['train'])
    val_img_dir = os.path.join(base_path, data_config['val'])
    test_img_dir = os.path.join(base_path, data_config['test'])
    num_classes = data_config['nc']

    # Assuming your label directories are named similarly to the image directories
    train_label_dir = train_img_dir.replace('images', 'labels')
    val_label_dir = val_img_dir.replace('images', 'labels')
    test_label_dir = test_img_dir.replace('images', 'labels')

    # Debug: Print dataset paths
    print(f"Train images: {train_img_dir}, Train labels: {train_label_dir}")
    print(f"Validation images: {val_img_dir}, Validation labels: {val_label_dir}")
    print(f"Test images: {test_img_dir}, Test labels: {test_label_dir}")

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])

    # Create datasets and data loaders
    train_dataset = RubiksCubeDataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=train_transform)
    val_dataset = RubiksCubeDataset(img_dir=val_img_dir, label_dir=val_label_dir, transform=val_test_transform)
    test_dataset = RubiksCubeDataset(img_dir=test_img_dir, label_dir=test_label_dir, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Initialize model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ImprovedCNN(num_classes=num_classes).to(device)

    # Training setup
    num_epochs = 25
    initial_lr = 0.001

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Initialize W&B
    config = {
        "learning_rate": initial_lr,
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "num_classes": num_classes,
    }
    initialize_wandb("yolo_rubiks_cube", config)

    def prepare_targets(outputs, boxes, labels, num_classes):
        batch_size, channels, height, width = outputs.size()
        targets = torch.zeros_like(outputs).to(device)
        for i in range(batch_size):
            for j in range(len(labels[i])):
                if labels[i][j] != -1:  # Ensure there are valid boxes
                    x, y, w, h = boxes[i][j]
                    class_id = labels[i][j]
                    targets[i, class_id, :, :] = 1  # class score
                    targets[i, num_classes, :, :] = x
                    targets[i, num_classes + 1, :, :] = y
                    targets[i, num_classes + 2, :, :] = w
                    targets[i, num_classes + 3, :, :] = h
        return targets

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, boxes, labels) in enumerate(train_loader):
            images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)

            outputs = model(images)

            targets = prepare_targets(outputs, boxes, labels, num_classes)

            loss = yolo_loss(outputs, targets, num_classes=num_classes)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")
        log_metrics({"train_loss": avg_train_loss}, epoch)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, boxes, labels in val_loader:
                images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)

                outputs = model(images)

                targets = prepare_targets(outputs, boxes, labels, num_classes)

                loss = yolo_loss(outputs, targets, num_classes=num_classes)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss}")
        log_metrics({"val_loss": avg_val_loss}, epoch)

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Save the model if validation loss is the best we've seen so far.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_yolo_rubiks_cube.pth')

    # Save the final model
    torch.save(model.state_dict(), 'final_yolo_rubiks_cube.pth')

    # Evaluate on the test dataset
    test_loss = 0.0
    with torch.no_grad():
        for images, boxes, labels in test_loader:
            images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)

            outputs = model(images)

            targets = prepare_targets(outputs, boxes, labels, num_classes)

            loss = yolo_loss(outputs, targets, num_classes=num_classes)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")
    log_metrics({"test_loss": avg_test_loss}, num_epochs)

    # Finish W&B run
    finish_wandb()

if __name__ == '__main__':
    main()
