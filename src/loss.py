import torch.nn.functional as F

def yolo_loss(predictions, targets, num_classes):
    # Split the predictions into class scores, bounding box coordinates
    class_preds = predictions[..., 0]
    box_preds = predictions[..., 1:]

    class_targets = targets[..., 0]
    box_targets = targets[..., 1:]

    class_loss = F.cross_entropy(class_preds, class_targets)

    box_loss = F.mse_loss(box_preds, box_targets)

    return class_loss + box_loss
