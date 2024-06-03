import torch
import torch.nn as nn
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

    # Calculate the class loss using Binary Cross-Entropy Loss
    class_loss = F.binary_cross_entropy_with_logits(class_preds, class_targets, reduction='sum')

    # Calculate the coordinate loss using Smooth L1 Loss
    box_loss = lambda_coord * F.smooth_l1_loss(box_preds, box_targets, reduction='sum')

    # Total loss
    total_loss = (class_loss + box_loss) / batch_size

    return total_loss
