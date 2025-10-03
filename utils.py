# Import necessary libraries for plotting and visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
import torch
import torch.nn as nn
import os
import json

# --- Function to Plot Training Metrics ---
# Concept: Visualizing the training process is crucial for understanding model performance.
# Technical: This function uses Matplotlib to create subplots for loss and accuracy curves.
def plot_metrics(train_losses, train_acc, test_losses, test_acc, output_dir: str = 'analytics', suffix: str = ''):
    """
    This function plots the training and testing loss and accuracy curves
    and saves the plot as a PNG image.
    
    Args:
        train_losses (list[float]): Per-epoch training loss.
        train_acc (list[float]): Per-epoch training accuracy.
        test_losses (list[float]): Per-epoch test loss.
        test_acc (list[float]): Per-epoch test accuracy.
        output_dir (str): Base analytics directory to save outputs.
        suffix (str): Filename suffix like '', '_1', '_2' based on model file.
    """
    # Ensure directories exist
    post_dir = os.path.join(output_dir, 'post_training')
    os.makedirs(post_dir, exist_ok=True)
    # Epoch index starting at 1 for readability
    epochs = list(range(1, max(len(train_losses), len(test_losses)) + 1))

    # Helper to compute tight y-lims with padding
    def _ylims(series, is_acc=False):
        if not series:
            return None
        smin = min(series)
        smax = max(series)
        rng = max(smax - smin, 1e-9)
        pad = max(0.1 * rng, 0.2 if is_acc else 0.005)  # ensure small but visible padding
        lo = smin - pad
        hi = smax + pad
        if is_acc:
            lo = max(lo, 0.0)
            hi = min(hi, 100.0)
        else:
            lo = max(lo, 0.0)
        # If range is extremely small, widen slightly
        if hi - lo < (2.0 if is_acc else 0.02):
            center = 0.5 * (hi + lo)
            half = (1.0 if is_acc else 0.01)
            lo, hi = center - half, center + half
            if is_acc:
                lo = max(lo, 0.0)
                hi = min(hi, 100.0)
            else:
                lo = max(lo, 0.0)
        return lo, hi

    # Create main figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    # Plot Losses
    axs[0, 0].plot(epochs[:len(train_losses)], train_losses, marker='o', markersize=3, linewidth=1.2, label='Train')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    yl = _ylims(train_losses, is_acc=False)
    if yl:
        axs[0, 0].set_ylim(*yl)
    axs[0, 0].grid(True, which='both', linestyle='--', alpha=0.3)
    axs[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
    axs[0, 0].yaxis.set_minor_locator(AutoMinorLocator(2))

    axs[0, 1].plot(epochs[:len(test_losses)], test_losses, marker='o', markersize=3, linewidth=1.2, color='tab:orange', label='Test')
    axs[0, 1].set_title('Test Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    yl = _ylims(test_losses, is_acc=False)
    if yl:
        axs[0, 1].set_ylim(*yl)
    axs[0, 1].grid(True, which='both', linestyle='--', alpha=0.3)
    axs[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
    axs[0, 1].yaxis.set_minor_locator(AutoMinorLocator(2))

    # Plot Accuracies (percent scale)
    axs[1, 0].plot(epochs[:len(train_acc)], train_acc, marker='o', markersize=3, linewidth=1.2, label='Train')
    axs[1, 0].set_title('Training Accuracy (%)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy (%)')
    yl = _ylims(train_acc, is_acc=True)
    if yl:
        axs[1, 0].set_ylim(*yl)
        # finer ticks for accuracy (in %)
        span = yl[1] - yl[0]
        if span <= 2:
            major = 0.2
        elif span <= 5:
            major = 0.5
        elif span <= 10:
            major = 1.0
        elif span <= 20:
            major = 2.0
        else:
            major = 5.0
        axs[1, 0].yaxis.set_major_locator(MultipleLocator(major))
        axs[1, 0].yaxis.set_minor_locator(MultipleLocator(major / 2))
    axs[1, 0].grid(True, which='both', linestyle='--', alpha=0.3)

    axs[1, 1].plot(epochs[:len(test_acc)], test_acc, marker='o', markersize=3, linewidth=1.2, color='tab:green', label='Test')
    axs[1, 1].set_title('Test Accuracy (%)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy (%)')
    yl = _ylims(test_acc, is_acc=True)
    if yl:
        axs[1, 1].set_ylim(*yl)
        # finer ticks for accuracy (in %)
        span = yl[1] - yl[0]
        if span <= 2:
            major = 0.2
        elif span <= 5:
            major = 0.5
        elif span <= 10:
            major = 1.0
        elif span <= 20:
            major = 2.0
        else:
            major = 5.0
        axs[1, 1].yaxis.set_major_locator(MultipleLocator(major))
        axs[1, 1].yaxis.set_minor_locator(MultipleLocator(major / 2))
    axs[1, 1].grid(True, which='both', linestyle='--', alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(post_dir, f'training_performance{suffix}.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved training performance graph to {out_path}")

    # Additional zoomed view for the last few epochs to capture minor fluctuations
    zoom_n = 5
    if len(epochs) > zoom_n:
        z_epochs = list(range(len(epochs) - zoom_n + 1, len(epochs) + 1))
        fig2, axs2 = plt.subplots(2, 2, figsize=(14, 9))
        # Slice last N
        tl_z = train_losses[-zoom_n:]
        tsl_z = test_losses[-zoom_n:]
        ta_z = train_acc[-zoom_n:]
        tsa_z = test_acc[-zoom_n:]

        axs2[0, 0].plot(z_epochs[:len(tl_z)], tl_z, marker='o', markersize=3, linewidth=1.2)
        axs2[0, 0].set_title(f'Training Loss (last {zoom_n} epochs)')
        axs2[0, 0].set_xlabel('Epoch')
        axs2[0, 0].set_ylabel('Loss')
        yl = _ylims(tl_z, is_acc=False)
        if yl:
            axs2[0, 0].set_ylim(*yl)
        axs2[0, 0].grid(True, which='both', linestyle='--', alpha=0.3)
        axs2[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
        axs2[0, 0].yaxis.set_minor_locator(AutoMinorLocator(2))

        axs2[0, 1].plot(z_epochs[:len(tsl_z)], tsl_z, marker='o', markersize=3, linewidth=1.2, color='tab:orange')
        axs2[0, 1].set_title(f'Test Loss (last {zoom_n} epochs)')
        axs2[0, 1].set_xlabel('Epoch')
        axs2[0, 1].set_ylabel('Loss')
        yl = _ylims(tsl_z, is_acc=False)
        if yl:
            axs2[0, 1].set_ylim(*yl)
        axs2[0, 1].grid(True, which='both', linestyle='--', alpha=0.3)
        axs2[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
        axs2[0, 1].yaxis.set_minor_locator(AutoMinorLocator(2))

        axs2[1, 0].plot(z_epochs[:len(ta_z)], ta_z, marker='o', markersize=3, linewidth=1.2)
        axs2[1, 0].set_title(f'Training Accuracy (%) (last {zoom_n} epochs)')
        axs2[1, 0].set_xlabel('Epoch')
        axs2[1, 0].set_ylabel('Accuracy (%)')
        yl = _ylims(ta_z, is_acc=True)
        if yl:
            axs2[1, 0].set_ylim(*yl)
            span = yl[1] - yl[0]
            if span <= 2:
                major = 0.2
            elif span <= 5:
                major = 0.5
            elif span <= 10:
                major = 1.0
            elif span <= 20:
                major = 2.0
            else:
                major = 5.0
            axs2[1, 0].yaxis.set_major_locator(MultipleLocator(major))
            axs2[1, 0].yaxis.set_minor_locator(MultipleLocator(major / 2))
        axs2[1, 0].grid(True, which='both', linestyle='--', alpha=0.3)

        axs2[1, 1].plot(z_epochs[:len(tsa_z)], tsa_z, marker='o', markersize=3, linewidth=1.2, color='tab:green')
        axs2[1, 1].set_title(f'Test Accuracy (%) (last {zoom_n} epochs)')
        axs2[1, 1].set_xlabel('Epoch')
        axs2[1, 1].set_ylabel('Accuracy (%)')
        yl = _ylims(tsa_z, is_acc=True)
        if yl:
            axs2[1, 1].set_ylim(*yl)
            span = yl[1] - yl[0]
            if span <= 2:
                major = 0.2
            elif span <= 5:
                major = 0.5
            elif span <= 10:
                major = 1.0
            elif span <= 20:
                major = 2.0
            else:
                major = 5.0
            axs2[1, 1].yaxis.set_major_locator(MultipleLocator(major))
            axs2[1, 1].yaxis.set_minor_locator(MultipleLocator(major / 2))
        axs2[1, 1].grid(True, which='both', linestyle='--', alpha=0.3)

        fig2.tight_layout()
        out_path2 = os.path.join(post_dir, f'training_performance_zoom{suffix}.png')
        fig2.savefig(out_path2, dpi=160, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved zoomed training performance graph to {out_path2}")

# --- Function to Plot Misclassified Images ---
# Concept: Error analysis is key to improving a model. Visualizing where the model fails helps identify patterns.
# Technical: This function iterates through the test set, finds incorrect predictions, and plots them.
def plot_misclassified_images(model, device, test_loader, num_images: int = 10, output_dir: str = 'analytics', suffix: str = ''):
    """
    This function finds, plots, and saves a specified number of misclassified
    images from the test set.
    """
    misclassified_images = []
    misclassified_labels = []
    correct_labels = []
    
    # Set the model to evaluation mode and disable gradients.
    model.eval()
    with torch.no_grad():
        # Loop through the test data.
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Get model predictions.
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            # Identify the indices where the prediction does not match the target.
            misclassified_inds = (pred.eq(target.view_as(pred)) == False).nonzero(as_tuple=True)[0]
            
            # Store the misclassified examples until we have enough.
            for ind in misclassified_inds:
                if len(misclassified_images) < num_images:
                    misclassified_images.append(data[ind])
                    misclassified_labels.append(pred[ind].item())
                    correct_labels.append(target[ind].item())

    if not misclassified_images:
        print("No misclassified images found to plot.")
        return

    # Ensure directories exist
    post_dir = os.path.join(output_dir, 'post_training')
    os.makedirs(post_dir, exist_ok=True)

    # Create a figure to plot the images.
    fig = plt.figure(figsize=(10, 5))
    for i in range(len(misclassified_images)):
        # Add a subplot for each image.
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        # `imshow` displays the image tensor. `.cpu().squeeze()` moves the tensor to the CPU and removes extra dimensions.
        ax.imshow(misclassified_images[i].cpu().squeeze(), cmap='gray_r')
        # Set the title to show the model's prediction and the true label.
        ax.set_title(f"Pred: {misclassified_labels[i]}\nTrue: {correct_labels[i]}")

    # Save the figure with misclassified images.
    out_path = os.path.join(post_dir, f'misclassified_images{suffix}.png')
    plt.savefig(out_path)
    print(f"Saved {len(misclassified_images)} misclassified images to {out_path}")


# --- Confusion Matrix Utilities ---
def compute_confusion_matrix(model, device, data_loader, num_classes: int = 10) -> torch.Tensor:
    """Compute confusion matrix C where C[i, j] = count of class i predicted as j."""
    # Create CM on the same device as the model/data to avoid device mismatch (CPU vs XPU/CUDA)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            preds = logits.argmax(dim=1)
            # vectorized bincount approach
            idx = target * num_classes + preds
            # bincount may produce a tensor on the same device as idx; ensure it is on 'device'.
            binc = torch.bincount(idx, minlength=num_classes*num_classes).to(device)
            cm += binc.view(num_classes, num_classes)
    return cm


def plot_confusion_matrix(cm: torch.Tensor, class_names=None, normalize: str | None = None, output_dir: str = 'analytics', suffix: str = ''):
    """
    Plot and save a confusion matrix.
    normalize: None | 'true' (row-wise) | 'pred' (col-wise) | 'all'
    """
    post_dir = os.path.join(output_dir, 'post_training')
    os.makedirs(post_dir, exist_ok=True)

    cm_np = cm.cpu().numpy().astype(float)
    title = 'Confusion Matrix'
    if normalize is not None:
        if normalize == 'true':
            denom = cm_np.sum(axis=1, keepdims=True)
            denom[denom == 0] = 1
            cm_np = cm_np / denom
            title += ' (normalized by true)'
        elif normalize == 'pred':
            denom = cm_np.sum(axis=0, keepdims=True)
            denom[denom == 0] = 1
            cm_np = cm_np / denom
            title += ' (normalized by pred)'
        elif normalize == 'all':
            total = cm_np.sum()
            total = total if total > 0 else 1
            cm_np = cm_np / total
            title += ' (normalized overall)'

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_np, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    classes = class_names if class_names is not None else [str(i) for i in range(cm.shape[0])]
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # annotate
    thresh = cm_np.max() / 2 if cm_np.size > 0 else 0.5
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            val = cm_np[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha='center', va='center', color='white' if val > thresh else 'black', fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(post_dir, f'confusion_matrix{suffix}.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved confusion matrix to {out_path}")


def plot_per_class_accuracy(cm: torch.Tensor, class_names=None, output_dir: str = 'analytics', suffix: str = ''):
    """Plot per-class accuracy = diag(cm) / row_sum(cm)."""
    post_dir = os.path.join(output_dir, 'post_training')
    os.makedirs(post_dir, exist_ok=True)
    cm = cm.to(torch.float32)
    denom = cm.sum(dim=1).clamp_min(1.0)
    acc = (cm.diag() / denom).cpu().numpy()
    classes = class_names if class_names is not None else [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(classes)), acc, color='steelblue')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-class Accuracy')
    for i, v in enumerate(acc):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(post_dir, f'per_class_accuracy{suffix}.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved per-class accuracy chart to {out_path}")


def save_classification_report(cm: torch.Tensor, output_dir: str = 'analytics', suffix: str = ''):
    """Save basic metrics (overall acc, per-class precision/recall/F1) to JSON."""
    post_dir = os.path.join(output_dir, 'post_training')
    os.makedirs(post_dir, exist_ok=True)
    cm = cm.to(torch.float32)
    total = cm.sum().item() if cm.numel() > 0 else 0.0
    overall_acc = (cm.diag().sum() / cm.sum().clamp_min(1.0)).item() if total > 0 else 0.0
    tp = cm.diag()
    per_class_recall = (tp / cm.sum(dim=1).clamp_min(1.0)).tolist()
    per_class_precision = (tp / cm.sum(dim=0).clamp_min(1.0)).tolist()
    f1 = []
    for p, r in zip(per_class_precision, per_class_recall):
        denom = (p + r) if (p + r) > 0 else 1.0
        f1.append(2 * p * r / denom)
    report = {
        'overall_accuracy': overall_acc,
        'per_class': {
            'precision': per_class_precision,
            'recall': per_class_recall,
            'f1': f1,
        }
    }
    out_path = os.path.join(post_dir, f'classification_report{suffix}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Saved classification report to {out_path}")


# --- Model Analysis (Params + Receptive Field) ---
def _count_module_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_param_breakdown(model: nn.Module) -> dict:
    """
    Return total trainable params and a per-module breakdown for key layers.
    Filters to Conv2d, BatchNorm2d, Linear, and Pooling layers for readability.
    """
    include_types = (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d)
    layers = []
    total = 0
    for name, module in model.named_modules():
        if name == '':
            # skip root itself in the list, but include its params in total via sum below
            continue
        if isinstance(module, include_types):
            params = _count_module_params(module)
            meta = {
                'name': name,
                'type': type(module).__name__,
                'params': params,
            }
            # Attach common attributes when available
            if isinstance(module, nn.Conv2d):
                meta.update({
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'dilation': module.dilation,
                    'groups': module.groups,
                    'bias': module.bias is not None,
                })
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                meta.update({
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'dilation': getattr(module, 'dilation', 1),
                })
            elif isinstance(module, nn.Linear):
                meta.update({
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None,
                })
            layers.append(meta)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total, 'layers': layers}


def compute_receptive_field(model: nn.Module) -> dict:
    """
    Compute receptive field progression for a model by iterating Conv/Pool layers in registration order.
    Uses standard formula with dilation. Returns final RF and per-layer progression.
    Note: This assumes a straightforward feedforward path (no complex branching).
    """
    rf = 1  # receptive field size
    j = 1   # jump (effective stride)
    progression = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
            s = m.stride[0] if isinstance(m.stride, tuple) else m.stride
            d = m.dilation[0] if isinstance(m.dilation, tuple) else m.dilation
            eff_k = d * (k - 1) + 1
            rf = rf + (eff_k - 1) * j
            progression.append({'name': name, 'type': 'Conv2d', 'kernel': k, 'stride': s, 'dilation': d, 'rf': rf, 'jump': j * s})
            j = j * s
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else (m.stride[0] if m.stride is not None else k)
            d = getattr(m, 'dilation', 1)
            eff_k = (d if isinstance(d, int) else d) * (k - 1) + 1
            rf = rf + (eff_k - 1) * j
            progression.append({'name': name, 'type': type(m).__name__, 'kernel': k, 'stride': s, 'dilation': d, 'rf': rf, 'jump': j * s})
            j = j * s

    return {'final_rf': rf, 'progression': progression}


def save_model_analysis(model: nn.Module, output_dir: str = 'analytics', suffix: str = '') -> dict:
    """
    Save model parameter breakdown and receptive field progression as JSON under analytics/pre_training.
    Returns dict with saved file path.
    """
    pre_dir = os.path.join(output_dir, 'pre_training')
    os.makedirs(pre_dir, exist_ok=True)
    params_info = compute_param_breakdown(model)
    rf_info = compute_receptive_field(model)
    report = {
        'params': params_info,
        'receptive_field': rf_info,
    }
    out_path = os.path.join(pre_dir, f'model_analysis{suffix}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved model analysis to {out_path}")
    return {'json': out_path}


def _as_pair(v):
    if isinstance(v, tuple):
        return v
    return (v, v)


def save_model_analysis_csv(model: nn.Module, output_dir: str = 'analytics', suffix: str = '') -> dict:
    """Save two CSVs: model_params{suffix}.csv and model_rf_progression{suffix}.csv under pre_training."""
    import csv
    pre_dir = os.path.join(output_dir, 'pre_training')
    os.makedirs(pre_dir, exist_ok=True)

    # Params breakdown CSV
    params_info = compute_param_breakdown(model)
    params_csv = os.path.join(pre_dir, f'model_params{suffix}.csv')
    with open(params_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'name', 'type', 'params',
            'in_channels', 'out_channels',
            'kernel_h', 'kernel_w', 'stride_h', 'stride_w', 'padding_h', 'padding_w', 'dilation_h', 'dilation_w', 'groups', 'bias',
            'in_features', 'out_features'
        ])
        for layer in params_info['layers']:
            k_h = k_w = s_h = s_w = p_h = p_w = d_h = d_w = groups = bias = in_ch = out_ch = in_f = out_f = ''
            if 'in_channels' in layer:
                in_ch = layer['in_channels']
            if 'out_channels' in layer:
                out_ch = layer['out_channels']
            if 'kernel_size' in layer:
                k_h, k_w = _as_pair(layer['kernel_size'])
            if 'stride' in layer:
                s_h, s_w = _as_pair(layer['stride'])
            if 'padding' in layer:
                p_h, p_w = _as_pair(layer['padding'])
            if 'dilation' in layer:
                d_h, d_w = _as_pair(layer['dilation'])
            if 'groups' in layer:
                groups = layer['groups']
            if 'bias' in layer:
                bias = layer['bias']
            if 'in_features' in layer:
                in_f = layer['in_features']
            if 'out_features' in layer:
                out_f = layer['out_features']
            writer.writerow([
                layer.get('name', ''), layer.get('type', ''), layer.get('params', ''),
                in_ch, out_ch,
                k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w, groups, bias,
                in_f, out_f
            ])

    # RF progression CSV
    rf_info = compute_receptive_field(model)
    rf_csv = os.path.join(pre_dir, f'model_rf_progression{suffix}.csv')
    with open(rf_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'name', 'type', 'kernel', 'stride', 'dilation', 'jump', 'rf'])
        for idx, step in enumerate(rf_info['progression'], start=1):
            writer.writerow([
                idx,
                step.get('name', ''),
                step.get('type', ''),
                step.get('kernel', ''),
                step.get('stride', ''),
                step.get('dilation', ''),
                step.get('jump', ''),
                step.get('rf', ''),
            ])
    print(f"Saved model params CSV to {params_csv}")
    print(f"Saved model RF progression CSV to {rf_csv}")
    return {'params_csv': params_csv, 'rf_csv': rf_csv}
