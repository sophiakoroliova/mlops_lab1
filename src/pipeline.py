import os
import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List

logger = logging.getLogger(__name__)

def train_epoch(model, loader, optimizer, criterion, device):
    """Performs a single training epoch over the dataset."""
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Standard PyTorch training step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, device, criterion=None):
    """Evaluates the model on validation or test data and calculates metrics."""
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Optional: calculate loss for validation monitoring
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate classification metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_score": precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)[2]
    }
    if criterion:
        metrics["loss"] = running_loss / len(loader)
    return metrics


def run_training(model, train_loader, val_loader, optimizer, criterion, device, epochs, output_dir):
    """Manages the full training cycle, including checkpointing and plotting."""
    history = {'train_loss': [], 'val_loss': []}
    best_f1 = 0.0

    # Path for saving the best model
    model_save_path = os.path.join(output_dir, 'best_model.pth')

    for epoch in range(epochs):
        # Training and validation steps
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, device, criterion=criterion)

        # Store loss history for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_results['loss']:.4f} | F1: {val_results['f1_score']:.4f}")

        # Model Checkpointing: Save the state if validation F1 improves
        if val_results['f1_score'] > best_f1:
            best_f1 = val_results['f1_score']
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved new best model (F1: {best_f1:.4f})")

    # Generate and save the learning curves plot inside the output directory
    plot_path = os.path.join(output_dir, 'loss_plot.png')

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()
    logger.info(f"Loss plot saved to {plot_path}")