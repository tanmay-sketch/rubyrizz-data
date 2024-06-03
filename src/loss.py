import torch
import torch.nn.functional as F

def yolo_loss(preds, targets, num_classes, lambda_coord=5, lambda_noobj=0.5):
    # Extract the different components of the predictions and targets
    batch_size, _, height, width = preds.size()
    preds = preds.permute(0, 2, 3, 1).contiguous()
    targets = targets.permute(0, 2, 3, 1).contiguous()

    class_preds = preds[:, :, :, :num_classes]  # Predictions for class scores
    box_preds = preds[:, :, :, num_classes:]     # Predictions for bounding boxes

    class_targets = targets[:, :, :, :num_classes]  # Targets for class scores
    box_targets = targets[:, :, :, num_classes:]     # Targets for bounding boxes

    # Calculate the class loss
    class_loss = F.binary_cross_entropy_with_logits(class_preds, class_targets)

    # Calculate the coordinate loss
    coord_loss = lambda_coord * F.mse_loss(box_preds, box_targets, reduction='sum')

    # Total loss
    total_loss = class_loss + coord_loss

    return total_loss
