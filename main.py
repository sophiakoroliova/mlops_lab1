import os
import yaml
import torch
import logging
from src.dataset import get_dataloaders
from src.model import ImprovedCNN
from src.pipeline import run_training, evaluate

def main():
    # Load project configurations from YAML
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Define and create the output directory for artifacts
    output_dir = config['training'].get('save_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Define paths for logs and artifacts
    log_path = os.path.join(output_dir, "training.log")
    model_path = os.path.join(output_dir, config['training']['save_path'])

    # Setup logging to both file and console
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    # Set device to GPU (cuda) if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data loaders for training, validation, and testing
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Initialize the model, optimizer, and loss function
    model = ImprovedCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    # Start the training process for N epochs
    # Pass output_dir to handle artifact saving inside the pipeline
    run_training(model, train_loader, val_loader, optimizer, criterion, device,
                 config['training']['epochs'], output_dir)

    # IMPORTANT: Load the best model weights saved during training for final evaluation
    print("\nLoading the best model for final testing")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Perform final evaluation on the unseen test set
    final_metrics = evaluate(model, test_loader, device)
    print(f"Final results on test data: {final_metrics}")


if __name__ == "__main__":
    main()