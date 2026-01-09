import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import splitfolders
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------

def split_data(input_folder, output_folder, ratio=(0.8, 0.2)):
    """
    Splits data from input_folder into train/val sets in output_folder.
    """
    if not os.path.exists(output_folder):
        print(f"Splitting data from {input_folder} to {output_folder} with ratio {ratio}...")
        splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=ratio, group_prefix=None, move=False)
    else:
        print(f"Data split directory {output_folder} already exists. Skipping split.")

def get_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=4, pin_memory=True):
    """
    Creates and returns DataLoaders for train, val (and test if available) sets.
    Return: train_loader, val_loader, test_loader, class_names
    """
    
    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            # scale=(0.4, 1.0) prevents extreme close-ups (zoom).
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # Resize slightly larger then crop
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {}
    for x in ['train', 'val']:
        path = os.path.join(data_dir, x)
        if os.path.exists(path):
            image_datasets[x] = datasets.ImageFolder(root=path, transform=data_transforms[x])
        else:
            raise FileNotFoundError(f"Directory {path} not found.")

    test_path = os.path.join(data_dir, 'test')
    test_exists = os.path.exists(test_path)
    if test_exists:
        image_datasets['test'] = datasets.ImageFolder(root=test_path, transform=data_transforms['test'])

    # DataLoader arguments
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    
    if num_workers > 0:
        loader_args['persistent_workers'] = True
        loader_args['prefetch_factor'] = 2

    loaders = {
        x: DataLoader(image_datasets[x], shuffle=(x == 'train'), **loader_args)
        for x in image_datasets
    }

    class_names = image_datasets['train'].classes
    
    # Fallback for test loader
    test_loader = loaders['test'] if test_exists else loaders['val']

    return loaders['train'], loaders['val'], test_loader, class_names

# -----------------------------------------------------------------------------
# File & History Management
# -----------------------------------------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history, f, indent=4)

def load_history(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_checkpoint(state, filename):
    torch.save(state, filename)

# -----------------------------------------------------------------------------
# Plotting & Visualization
# -----------------------------------------------------------------------------

def plot_training_history(history, save_path=None):
    """
    Plots training and validation loss and accuracy from the history dictionary.
    """
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    train_acc = history.get('train_acc', [])
    val_acc = history.get('val_acc', [])
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Train Loss')
    plt.plot(epochs, val_loss, 'b-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r-', label='Train Acc')
    plt.plot(epochs, val_acc, 'b-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def unnormalize_tensor(tensor):
    """
    Un-normalizes a tensor (3, H, W) approx back to [0,1] range for visualization.
    Assumes standard ImageNet mean/std.
    """
    # Clone and detach
    img = tensor.cpu().clone().detach()
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    
    # Clip
    img = torch.clamp(img, 0, 1)
    return img

def collect_examples(model, dataloader, device, num_correct=5, num_incorrect=5):
    """
    Collects a few correct and incorrect predictions from the validation set.
    Returns lists of (image_np, pred_label, true_label, tensor_shape, np_shape).
    """
    model.eval()
    correct_all = []
    incorrect_all = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(len(targets)):
                img_t = unnormalize_tensor(inputs[i])
                img_np = img_t.permute(1, 2, 0).numpy()  # (H, W, C)
                pred = preds[i].item()
                true = targets[i].item()

                item = (img_np, pred, true, tuple(img_t.shape), img_np.shape)

                if pred == true:
                    correct_all.append(item)
                else:
                    incorrect_all.append(item)

            if len(correct_all) > 50 and len(incorrect_all) > 50:
                break

    # Randomly sample
    n_corr = min(num_correct, len(correct_all))
    n_inc = min(num_incorrect, len(incorrect_all))
    correct = random.sample(correct_all, n_corr) if n_corr > 0 else []
    incorrect = random.sample(incorrect_all, n_inc) if n_inc > 0 else []
    
    return correct, incorrect

def plot_examples(correct, incorrect, classes, out_path=None):
    """
    Plots a grid of correct and incorrect examples.
    """
    ncol = max(len(correct), len(incorrect), 1)
    # If we have very few examples, adjust figsize
    fig, axes = plt.subplots(2, ncol, figsize=(4 * ncol, 8))
    
    # If ncol=1, subplots returns a 1D array or single axes depending on squeeze.
    # We ensure axes is 2D array: (2, ncol)
    if ncol == 1:
        axes = np.array(axes).reshape(2, 1)
    elif len(axes.shape) == 1: # Should not happen with 2 rows but just in case
        axes = np.array(axes).reshape(2, ncol)

    # Plot correct
    for idx in range(ncol):
        ax = axes[0, idx]
        if idx < len(correct):
            img, p, t, _, _ = correct[idx]
            ax.imshow(img)
            ax.set_title(f"Pred: {classes[p]}\nTrue: {classes[t]}", color='green')
        else:
            ax.text(0.5, 0.5, "No Example", ha='center', va='center')
        ax.axis('off')

    # Plot incorrect
    for idx in range(ncol):
        ax = axes[1, idx]
        if idx < len(incorrect):
            img, p, t, _, _ = incorrect[idx]
            ax.imshow(img)
            ax.set_title(f"Pred: {classes[p]}\nTrue: {classes[t]}", color='red')
        else:
            ax.text(0.5, 0.5, "No Example", ha='center', va='center')
        ax.axis('off')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.close()
