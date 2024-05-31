import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def visualize_annotations(image_path, label_path):
    image = Image.open(image_path)
    boxes = []
    labels = []

    with open(label_path, "r") as file:
        for line in file:
            label, x_center, y_center, width, height = map(float, line.split())
            labels.append(int(label))
            boxes.append([x_center, y_center, width, height])

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i, box in enumerate(boxes):
        x_center, y_center, width, height = box
        x_center *= image.width
        y_center *= image.height
        width *= image.width
        height *= image.height

        x = x_center - width / 2
        y = y_center - height / 2

        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, str(labels[i]), color='white', backgroundcolor='red', fontsize=12)

    plt.show()

# Example usage
image_path = '/Users/tanmay/Documents/Coding/Repositories/rubyrizz-data/data/train/images/3.rf.6f349dfe5c59bb9331718ed27e9814cb.jpg'
label_path = '/Users/tanmay/Documents/Coding/Repositories/rubyrizz-data/data/train/labels/3.rf.6f349dfe5c59bb9331718ed27e9814cb.txt'
visualize_annotations(image_path, label_path)
