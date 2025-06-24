import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchmetrics.functional as tmf


def safe_tensor_to_numpy(tensor):
    """Convert tensor to numpy safely"""
    if tensor is None:
        return None
    if isinstance(tensor, (int, float)):
        return tensor
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def compute_loss(outputs, batch, task_type, device):
    """Compute loss for the current task"""
    if task_type == "classification":
        logits = outputs["logits"]
        labels = batch["label"].to(device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        # Return loss and components for consistency
        loss_components = {"loss": loss}
        return loss, loss_components

    elif task_type == "segmentation":
        seg_output = outputs["segmentation"]
        mask = batch["mask"].to(device)
        # Ensure mask has a channel dimension [B,1,H,W] if provided as [B,H,W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Ensure mask dimensions match segmentation output channels
        if seg_output.shape[1] > 1 and mask.shape[1] == 1:
            # Convert binary mask to one-hot encoding
            mask_onehot = torch.zeros_like(seg_output)
            mask_onehot[:, 0:1] = 1.0 - mask  # Background
            mask_onehot[:, 1:2] = mask  # Foreground
            mask = mask_onehot

        loss = nn.BCEWithLogitsLoss()(seg_output, mask)

        # Update compute_loss to return both loss and loss_components
        loss_components = {"loss": loss}
        return loss, loss_components

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def compute_metrics(outputs, batch, task_type, device):
    """Compute metrics for evaluation"""
    metrics = {}

    if task_type == "classification":
        logits = outputs["logits"]
        labels = batch["label"].to(device)
        preds = torch.argmax(logits, dim=1)
        num_classes = logits.shape[1]
        # Dynamically choose binary vs multiclass
        if num_classes == 2:
            # Binary classification metrics
            metrics["accuracy"] = tmf.accuracy(preds, labels, task="binary")
            metrics["precision"] = tmf.precision(preds, labels, task="binary")
            metrics["recall"] = tmf.recall(preds, labels, task="binary")
            metrics["f1"] = tmf.f1_score(preds, labels, task="binary")
        else:
            # Multiclass classification metrics
            metrics["accuracy"] = tmf.accuracy(preds, labels, task="multiclass", num_classes=num_classes)
            metrics["precision_macro"] = tmf.precision(
                preds, labels, average="macro", task="multiclass", num_classes=num_classes
            )
            metrics["recall_macro"] = tmf.recall(
                preds, labels, average="macro", task="multiclass", num_classes=num_classes
            )
            metrics["f1_macro"] = tmf.f1_score(
                preds, labels, average="macro", task="multiclass", num_classes=num_classes
            )

    elif task_type == "segmentation":
        seg_output = outputs["segmentation"]
        mask_tensor = batch["mask"].to(device)
        # Get prediction map: argmax for multi-channel, threshold for single-channel
        if seg_output.shape[1] > 1:
            preds = torch.argmax(seg_output, dim=1)  # [B,H,W]
        else:
            preds = (torch.sigmoid(seg_output) > 0.5).int().squeeze(1)  # [B,H,W]
        # Prepare mask
        # Binarize mask tensor
        mask = (mask_tensor.squeeze(1) > 0.5).int()  # [B,H,W]
        # Use torchmetrics for metrics
        metrics["iou"] = tmf.jaccard_index(preds, mask, task="binary")
        metrics["dice"] = tmf.f1_score(preds, mask, task="binary")
        metrics["precision"] = tmf.precision(preds, mask, task="binary")
        metrics["recall"] = tmf.recall(preds, mask, task="binary")
        # Specificity: TN / (TN + FP)
        metrics["specificity"] = tmf.specificity(preds, mask, task="binary")

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return metrics


def train_epoch(
    model,
    dataloader,
    task_type,
    device,
    optimizer,
    compute_loss_fn,
    compute_metrics_fn,
    gradient_clip_norm=1.0,
):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    batch_metrics = []

    # Create progress bar
    pbar = tqdm(dataloader, desc=f"Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        batch["image"] = batch["image"].to(device)
        if "label" in batch:
            batch["label"] = batch["label"].to(device)
        if "mask" in batch:
            batch["mask"] = batch["mask"].to(device)

        # Add task info to batch if it doesn't exist
        if "task" not in batch:
            batch["task"] = [task_type] * batch["image"].size(0)

        # Ensure proper format for the model
        if isinstance(batch["task"], list) and not isinstance(batch["task"][0], str):
            batch["task"] = [task_type] * batch["image"].size(0)

        # Forward pass
        outputs = model(batch)

        # Compute loss
        loss, loss_components = compute_loss_fn(outputs, batch, task_type, device)

        # Ensure loss_components is always a dictionary
        if not isinstance(loss_components, dict):
            loss_components = {"loss": loss}

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        # Calculate metrics
        metrics = compute_metrics_fn(outputs, batch, task_type, device)

        # Add loss components to metrics for tracking
        for k, v in loss_components.items():
            metrics[k] = v.item() if isinstance(v, torch.Tensor) else v

        batch_metrics.append(metrics)

        # For segmentation, track dice score for monitoring
        if task_type == "segmentation" and "dice" in metrics:
            dice = metrics["dice"]

        # Update progress bar with most relevant metrics
        display_metrics = {}
        if task_type == "segmentation":
            display_metrics = {
                "loss": f"{loss.item():.4f}",
                "dice": f'{metrics.get("dice", 0):.4f}',
            }
        else:
            display_metrics = {
                "loss": f"{loss.item():.4f}",
                "acc": f'{metrics.get("accuracy", 0):.4f}',
            }

        pbar.set_postfix(display_metrics)

        # Accumulate loss
        total_loss += loss.item()

    # Calculate average metrics
    avg_metrics = {}
    if batch_metrics:
        for key in batch_metrics[0].keys():
            values = [safe_tensor_to_numpy(m[key]) for m in batch_metrics if key in m]
            if values:
                avg_metrics[key] = np.mean(values)

    avg_loss = total_loss / len(dataloader)

    return avg_loss, avg_metrics


def validate(model, dataloader, task_type, device, compute_loss_fn, compute_metrics_fn):
    """Validate model on the validation set"""
    model.eval()

    total_loss = 0
    batch_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            # Move data to device
            batch["image"] = batch["image"].to(device)
            if "label" in batch:
                batch["label"] = batch["label"].to(device)
            if "mask" in batch:
                batch["mask"] = batch["mask"].to(device)

            # Add task info to batch if it doesn't exist
            if "task" not in batch:
                batch["task"] = [task_type] * batch["image"].size(0)

            # Ensure proper format for the model
            if isinstance(batch["task"], list) and not isinstance(
                batch["task"][0], str
            ):
                batch["task"] = [task_type] * batch["image"].size(0)

            # Forward pass
            outputs = model(batch)

            # Compute loss
            loss, _ = compute_loss_fn(outputs, batch, task_type, device)
            total_loss += loss.item()

            # Calculate metrics
            metrics = compute_metrics_fn(outputs, batch, task_type, device)
            batch_metrics.append(metrics)

    # Calculate average metrics
    avg_metrics = {}
    if batch_metrics:
        for key in batch_metrics[0].keys():
            values = [safe_tensor_to_numpy(m[key]) for m in batch_metrics if key in m]
            if values:
                avg_metrics[key] = np.mean(values)

    avg_loss = total_loss / len(dataloader)

    return avg_loss, avg_metrics
