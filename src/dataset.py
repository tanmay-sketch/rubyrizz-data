import os
import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class RubiksCubeDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

        # Debug: Print the number of images and labels found
        print(f"Found {len(self.img_files)} images and {len(self.label_files)} labels in {img_dir} and {label_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        image = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    label, x_center, y_center, width, height = map(float, line.split())
                    labels.append(int(label))
                    boxes.append([x_center, y_center, width, height])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
