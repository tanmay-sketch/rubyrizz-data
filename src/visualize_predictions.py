import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import YOLOv5Moderate

# Load the model
num_classes = 6
model = YOLOv5Moderate(num_classes=num_classes)
model.load_state_dict(torch.load('final_yolo_rubiks_cube.pth'))
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

# Post-process and visualize the output
def postprocess_and_visualize(image_path, outputs, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    outputs = outputs.squeeze(0)  # Remove batch dimension
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    image_width, image_height = image.size

    for i in range(outputs.size(0)):
        class_score, x_center, y_center, width, height = outputs[i]
        class_score = torch.sigmoid(class_score)
        if class_score > threshold:
            x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height

            # Ensure width and height are positive
            width = max(width, 0)
            height = max(height, 0)

            x = x_center - width / 2
            y = y_center - height / 2

            # Ensure x, y, width, height are within image boundaries
            x = max(x, 0)
            y = max(y, 0)
            width = min(width, image_width - x)
            height = min(height, image_height - y)

            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x, y, f"Class {i} ({class_score:.2f})", color='white', backgroundcolor='red', fontsize=12)

            # Debug: Print the coordinates and class score
            print(f"Class {i}, Score: {class_score:.2f}, x: {x}, y: {y}, width: {width}, height: {height}")

    plt.show()

# Path to the image you want to test
image_path = 'image.png'  # Replace with your image path

# Preprocess the image
image = preprocess_image(image_path)

# Run the model on the preprocessed image
with torch.no_grad():
    outputs = model(image)

# Visualize the results
postprocess_and_visualize(image_path, outputs)
