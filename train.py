# --- Imports ---
# This section imports all necessary libraries for training, evaluation, and analysis.
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt
import os
import copy
import inspect
import json
from datetime import datetime
import csv
from utils import (
    plot_metrics,
    plot_misclassified_images,
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    save_classification_report,
    save_model_analysis,
    save_model_analysis_csv,
)
import sys
import argparse
import importlib
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Optional: Import Intel Extension for PyTorch for performance boost on Intel hardware.
try:
    import intel_extension_for_pytorch as ipex
    _HAS_IPEX = True
except ImportError:
    _HAS_IPEX = False

# --- Logger Class ---
# Concept: Redirects all console output (e.g., from `print` statements) to both the
# terminal and a log file simultaneously. This is crucial for keeping a persistent
# record of training progress, hyperparameters, and results.
class Logger(object):
    def __init__(self, filename="training.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8') # Overwrite log on each new run.

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- Training Function ---
# Defines the core logic for one epoch of training.
def train(model, device, train_loader, optimizer, epoch):
    """
    Args:
        model (nn.Module): The neural network model to train.
        device (torch.device): The device (CPU, CUDA, XPU) to train on.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        epoch (int): The current epoch number.
    """
    model.train()  # Set the model to training mode (enables Dropout, etc.).
    pbar = tqdm(train_loader)  # Wrap the loader with tqdm for a progress bar.
    correct = 0
    processed = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data and targets to the selected device.
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        # 1. Clear gradients from the previous iteration.
        optimizer.zero_grad()
        # 2. Forward pass: compute predicted outputs by passing inputs to the model.
        output = model(data)
        # 3. Calculate the loss.
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        # 4. Backward pass: compute gradient of the loss with respect to model parameters.
        loss.backward()
        # 5. Perform a single optimization step (parameter update).
        optimizer.step()

        # --- Batch-level accuracy calculation ---
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:.2f}'
        )
    
    # Return average loss and accuracy for the entire epoch.
    train_loss /= len(train_loader)
    train_acc = 100. * correct / len(train_loader.dataset)
    return train_loss, train_acc

# --- Testing Function ---
# Defines the core logic for evaluating the model on the test dataset.
def test(model, device, test_loader):
    """
    Args:
        model (nn.Module): The neural network model to evaluate.
        device (torch.device): The device (CPU, CUDA, XPU) to evaluate on.
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    model.eval()  # Set the model to evaluation mode (disables Dropout, etc.).
    test_loss = 0
    correct = 0
    # `torch.no_grad()` disables gradient computation, saving memory and speeding up inference.
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

# --- Helper Functions for File Naming and Directory Management ---
def _derive_suffix_from_model() -> str:
    """
    Derives a filename suffix from the model's module name (e.g., 'model_v1.py' -> '_v1').
    This allows all output files (logs, plots) to be uniquely named for each model version.
    """
    module_file = inspect.getsourcefile(Net) or ''
    base = os.path.basename(module_file)
    name, _ = os.path.splitext(base)
    if name == 'model':
        return ''
    if name.startswith('model_'):
        return f"_{name[len('model_'):]}"
    return f"_{name.replace('model', '')}"

def _ensure_analytics_dirs(base_dir: str = 'analytics'):
    """Creates the necessary subdirectories for saving analysis files."""
    os.makedirs(os.path.join(base_dir, 'pre_training'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'post_training'), exist_ok=True)

def _save_hparams(hparams: dict, base_dir: str = 'analytics', suffix: str = '') -> str:
    """Saves the hyperparameter dictionary to a JSON file."""
    path = os.path.join(base_dir, 'pre_training', f'hparams{suffix}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(hparams, f, indent=2, ensure_ascii=False, default=str)
    return path

def _print_hparams(hparams: dict):
    """Prints the hyperparameter configuration in a readable format."""
    print("\n--- Hyperparameters & Run Config ---")
    for k, v in hparams.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  - {kk}: {vv}")
        else:
            print(f"{k}: {v}")
    print("------------------------------------\n")

# --- Main Execution Block ---
def main():
    """Main function to orchestrate the entire training and evaluation pipeline."""
    # --- 1. Setup: Directories, Logging, and File Suffix ---
    suffix = _derive_suffix_from_model()
    _ensure_analytics_dirs('analytics')
    log_path = os.path.join('analytics', f"training{suffix}.log")
    sys.stdout = Logger(filename=log_path)

    # --- 2. Data Statistics Calculation ---
    # This block calculates the mean and standard deviation of the training dataset.
    # While we use the standard values (0.1307, 0.3081) for normalization, this serves as a
    # validation step to ensure the dataset is behaving as expected.
    print("Calculating dataset statistics (mean and std)...")
    temp_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=1000, shuffle=False, num_workers=0)
    mean = 0.
    std = 0.
    for images, _ in temp_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(temp_loader.dataset)
    std /= len(temp_loader.dataset)
    print(f"\n--- Calculated Dataset Statistics ---")
    print(f"Calculated Mean: {mean.item():.4f}")
    print(f"Calculated Std: {std.item():.4f}")
    print(f"Used for Normalization: Mean=0.1307, Std=0.3081")
    print("-------------------------------------\n")

    # --- 3. Data Preparation and Augmentation ---
    # Define the transformations to be applied to the training and test datasets.
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # --- 4. Dataset and DataLoader Creation ---
    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    # --- 5. Device Selection and Seeding ---
    # Set a seed for reproducibility.
    SEED = 1
    torch.manual_seed(SEED)
    # Prioritize hardware: Intel XPU > NVIDIA CUDA > CPU.
    xpu_available = bool(getattr(torch, 'xpu', None)) and torch.xpu.is_available()
    cuda_available = torch.cuda.is_available()
    print("Intel XPU Available?", xpu_available)
    print("CUDA Available?", cuda_available)
    if xpu_available:
        device = torch.device("xpu")
        try:
            torch.xpu.manual_seed(SEED)
        except AttributeError:
            pass
    elif cuda_available:
        device = torch.device("cuda")
        torch.cuda.manual_seed(SEED)
    else:
        device = torch.device("cpu")

    # DataLoader arguments are optimized for the selected device.
    dataloader_args = dict(shuffle=True, batch_size=128)
    if device.type in ("cuda", "xpu"):
        dataloader_args.update(num_workers=2, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    # For test loader, shuffling is unnecessary; reuse args but enforce shuffle=False
    test_loader = torch.utils.data.DataLoader(test_data, **{**dataloader_args, 'shuffle': False})

    # --- 6. Data Visualization and Pre-Training Analysis ---
    # Save a sample of the initial training data batch for visual inspection.
    images, _ = next(iter(train_loader))
    figure = plt.figure(figsize=(12, 12))
    for index in range(1, 61):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    data_sample_path = os.path.join('analytics', 'pre_training', f'data_sample{suffix}.png')
    plt.savefig(data_sample_path)
    # Close figure to free resources in long runs or repeated invocations
    try:
        plt.close(figure)
    except Exception:
        pass
    print(f"Saved a sample of the training data to {data_sample_path}\n")

    # --- 7. Model Initialization and Analysis ---
    print("Using device:", device)
    model = Net().to(device)
    # Display a summary of the model architecture and parameter count.
    try:
        # Use a CPU copy for summary if on XPU, as torchsummary may not support it directly.
        summary_device = 'cpu' if device.type == 'xpu' else device.type
        summary(copy.deepcopy(model).to(summary_device), input_size=(1, 28, 28), device=summary_device)
    except Exception as e:
        print(f"Warning: Skipping model summary due to: {e}")
    # Save detailed model analysis (params, RF) to JSON and CSV files.
    analysis_info = save_model_analysis(model, output_dir='analytics', suffix=suffix)
    print(f"Model analysis saved: {analysis_info['json']}")
    csv_info = save_model_analysis_csv(model, output_dir='analytics', suffix=suffix)
    print(f"Model analysis CSVs saved: {csv_info}")

    # --- 8. Optimizer and Scheduler Configuration ---
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    # ReduceLROnPlateau dynamically reduces the learning rate when a metric has stopped improving.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',      # Reduce LR when the monitored quantity (test_loss) stops decreasing.
        factor=0.5,      # New LR = LR * factor.
        patience=1,      # Number of epochs with no improvement after which LR will be reduced.
        threshold=1e-3,  # Threshold for measuring the new optimum.
        min_lr=1e-5,     # A lower bound on the learning rate.
    )
    EPOCHS = 20

    # --- 9. Hyperparameter Logging ---
    # Collect and save all relevant hyperparameters and configuration details for this run.
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams = {
        'run_timestamp': datetime.now().isoformat(),
        'model_source_file': inspect.getsourcefile(Net),
        'total_trainable_params': total_params,
        'optimizer': {
            'type': type(optimizer).__name__,
            'lr': optimizer.param_groups[0].get('lr'),
            'momentum': optimizer.param_groups[0].get('momentum'),
            'weight_decay': optimizer.param_groups[0].get('weight_decay'),
        },
        'scheduler': {
            'type': type(scheduler).__name__,
            'mode': scheduler.mode,
            'factor': scheduler.factor,
            'patience': scheduler.patience,
            'threshold': scheduler.threshold,
            'min_lr': scheduler.min_lrs[0],
        },
        'epochs': EPOCHS,
        'device': str(device),
        'batch_size': train_loader.batch_size,
        'train_dataset_size': len(train_data),
        'test_dataset_size': len(test_data),
        'train_transforms': str(train_transforms),
    }
    hparams_path = _save_hparams(hparams, base_dir='analytics', suffix=suffix)
    _print_hparams(hparams)
    print(f"Saved hyperparameters to {hparams_path}")

    # --- 10. Main Training and Evaluation Loop ---
    metrics = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    metrics_path = os.path.join('analytics', 'post_training', f'metrics{suffix}.csv')
    with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"EPOCH: {epoch} (Learning Rate: {current_lr:.6f})")
        # Perform one epoch of training and testing.
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        # Update the learning rate scheduler based on the validation loss.
        scheduler.step(test_loss)
        # Store metrics for plotting.
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        # Append epoch metrics to the CSV log.
        with open(metrics_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{current_lr:.6f}", f"{train_loss:.6f}", f"{train_acc:.2f}", f"{test_loss:.6f}", f"{test_acc:.2f}"])

    # --- 11. Post-Training Analysis and Visualization ---
    print("\n--- Training Complete ---")
    print("Generating performance plots and reports...")
    plot_metrics(metrics['train_loss'], metrics['train_acc'], metrics['test_loss'], metrics['test_acc'], output_dir='analytics', suffix=suffix)
    plot_misclassified_images(model, device, test_loader, output_dir='analytics', suffix=suffix)
    cm = compute_confusion_matrix(model, device, test_loader, num_classes=10)
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)], normalize='true', output_dir='analytics', suffix=suffix)
    plot_per_class_accuracy(cm, class_names=[str(i) for i in range(10)], output_dir='analytics', suffix=suffix)
    save_classification_report(cm, output_dir='analytics', suffix=suffix)

# --- Entry Point ---
# This allows the script to be run from the command line.
if __name__ == '__main__':
    # The argument parser allows specifying which model file to use.
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default='model', help='Model file name to use (e.g., model, model_v1, model_v2)')
    args = parser.parse_args()

    # Dynamically import the `Net` class from the specified model file.
    # This makes the training script modular and reusable for different architectures.
    print(f"Attempting to load 'Net' class from '{args.model}.py'")
    model_module = importlib.import_module(args.model)
    Net = model_module.Net
    print("Model loaded successfully. Starting main training process...")
    main()