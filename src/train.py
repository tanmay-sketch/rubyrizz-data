import sys
import os

# Add the project directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src import RubiksCubeDataset, YOLOv5Simple, yolo_loss, transform

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load data.yaml
with open('data/data.yaml', 'r') as file:
    data_config = yaml.safe_load(file)

train_img_dir = data_config['train']
val_img_dir = data_config['val']
num_classes = data_config['nc']

# Assuming your label directories are named similarly to the image directories
train_label_dir = train_img_dir.replace('images', 'labels')
val_label_dir = val_img_dir.replace('images', 'labels')

# Create datasets and data loaders
train_dataset = RubiksCubeDataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=transform)
val_dataset = RubiksCubeDataset(img_dir=val_img_dir, label_dir=val_label_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize model
model = YOLOv5Simple(num_classes=num_classes).to(device)

# Training setup
num_epochs = 25
learning_rate = 0.001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, boxes, labels in train_loader:
        images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Prepare targets for the loss function
        targets = torch.zeros_like(outputs).to(device)
        for i in range(len(labels)):
            targets[i, labels[i], :4] = boxes[i]

        loss = yolo_loss(outputs, targets, num_classes=num_classes)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, boxes, labels in val_loader:
            images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)
            
            outputs = model(images)
            
            targets = torch.zeros_like(outputs).to(device)
            for i in range(len(labels)):
                targets[i, labels[i], :4] = boxes[i]
            
            loss = yolo_loss(outputs, targets, num_classes=num_classes)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader)}")

torch.save(model.state_dict(), 'yolo_rubiks_cube.pth')