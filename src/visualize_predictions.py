import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import ImprovedCNN

# Load the model
num_classes = 6
model = ImprovedCNN(num_classes=num_classes)
model.load_state_dict(torch.load('best_yolo_rubiks_cube.pth'))
model.eval()

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Non-Maximum Suppression (NMS) function
def nms(boxes, scores, iou_threshold):
    indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
    return indices

# Post-process and visualize the output
def postprocess_and_visualize(image_path, outputs, threshold=0.5, iou_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    outputs = outputs.squeeze(0)  # Remove batch dimension
    outputs = outputs.permute(1, 2, 0)  # Change to (H, W, C) format
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    image_width, image_height = image.size

    # Print raw outputs for debugging
    print("Raw outputs:", outputs)

    boxes = []
    scores = []
    classes = []

    for i in range(num_classes):
        class_scores = torch.sigmoid(outputs[:, :, i])
        x_center = outputs[:, :, num_classes + 0]
        y_center = outputs[:, :, num_classes + 1]
        width = outputs[:, :, num_classes + 2]
        height = outputs[:, :, num_classes + 3]

        mask = class_scores > threshold
        for j in range(mask.shape[0]):
            for k in range(mask.shape[1]):
                if mask[j, k]:
                    x_center_scaled = x_center[j, k] * image_width
                    y_center_scaled = y_center[j, k] * image_height
                    width_scaled = width[j, k] * image_width
                    height_scaled = height[j, k] * image_height

                    x = x_center_scaled - width_scaled / 2
                    y = y_center_scaled - height_scaled / 2

                    boxes.append([x, y, x + width_scaled, y + height_scaled])
                    scores.append(class_scores[j, k].item())
                    classes.append(i)

    if len(boxes) > 0:
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        keep = nms(boxes, scores, iou_threshold)

        for idx in keep:
            box = boxes[idx]
            score = scores[idx]
            class_id = classes[idx]

            x, y, x2, y2 = box
            width = x2 - x
            height = y2 - y

            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x, y, f"Class {class_id} ({score:.2f})", color='white', backgroundcolor='red', fontsize=12)

            # Debug: Print the coordinates and class score
            print(f"Class {class_id}, Score: {score:.2f}, x: {x}, y: {y}, width: {width}, height: {height}")

    plt.show()

image_path = 'image.jpg'  

# Preprocess the image
image = preprocess_image(image_path)

# Run the model on the preprocessed image
with torch.no_grad():
    outputs = model(image)

# Visualize the results
postprocess_and_visualize(image_path, outputs)
