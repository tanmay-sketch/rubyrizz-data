import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import ImprovedCNN

# Load the model
num_classes = 6
model = ImprovedCNN(num_classes=num_classes)
model.load_state_dict(torch.load('final_yolo_rubiks_cube.pth'))
model.eval()

# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
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
    outputs = outputs.permute(1, 2, 0)  # Change to (H, W, C) format
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    image_width, image_height = image.size

    num_preds = outputs.size(2) // (num_classes + 4)
    for i in range(num_preds):
        class_scores = torch.sigmoid(outputs[:, :, i]).numpy()
        x_center = outputs[:, :, num_classes + i].numpy()
        y_center = outputs[:, :, num_classes + i + 1].numpy()
        width = outputs[:, :, num_classes + i + 2].numpy()
        height = outputs[:, :, num_classes + i + 3].numpy()

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

                    # Ensure x, y, width, height are within image boundaries
                    x = max(x, 0)
                    y = max(y, 0)
                    width_scaled = min(width_scaled, image_width - x)
                    height_scaled = min(height_scaled, image_height - y)

                    rect = patches.Rectangle((x, y), width_scaled, height_scaled, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    plt.text(x, y, f"Class {i} ({class_scores[j, k]:.2f})", color='white', backgroundcolor='red', fontsize=12)

                    # Debug: Print the coordinates and class score
                    print(f"Class {i}, Score: {class_scores[j, k]:.2f}, x: {x}, y: {y}, width: {width_scaled}, height: {height_scaled}")

    plt.show()

# Path to the image you want to test
image_path = 'image.jpg'  # Replace with your image path

# Preprocess the image
image = preprocess_image(image_path)

# Run the model on the preprocessed image
with torch.no_grad():
    outputs = model(image)

# Visualize the results
postprocess_and_visualize(image_path, outputs)
