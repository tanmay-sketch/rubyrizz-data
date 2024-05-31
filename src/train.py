import sys
import os
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# Add the project directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import RubiksCubeDataset, YOLOv5Moderate, yolo_loss, transform

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

# Create datasets and data loaders
train_dataset = RubiksCubeDataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=transform)
val_dataset = RubiksCubeDataset(img_dir=val_img_dir, label_dir=val_label_dir, transform=transform)
test_dataset = RubiksCubeDataset(img_dir=test_img_dir, label_dir=test_label_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Initialize model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = YOLOv5Moderate(num_classes=num_classes).to(device)

# Training setup
num_epochs = 25
initial_lr = 0.001

optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

def prepare_targets(outputs, boxes, labels, num_classes):
    targets = torch.zeros_like(outputs).to(device)
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -1:  # Ensure there are valid boxes
                targets[i, labels[i][j], :4] = boxes[i][j]
    return targets

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, boxes, labels in train_loader:
        images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        targets = prepare_targets(outputs, boxes, labels, num_classes)
        
        loss = yolo_loss(outputs, targets, num_classes=num_classes)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, boxes, labels in val_loader:
            images, boxes, labels = images.to(device), boxes.to(device), labels.to(device)
            
            outputs = model(images)
            
            targets = prepare_targets(outputs, boxes, labels, num_classes)
            
            loss = yolo_loss(outputs, targets, num_classes=num_classes)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss}")

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

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss}")
