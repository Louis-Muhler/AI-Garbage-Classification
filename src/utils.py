import os
import torch
import splitfolders
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import glob
import re
import matplotlib.pyplot as plt

def split_garbage_data(input_dir, output_dir, seed=42):
    """
    Splits the Kaggle dataset into 80% training and 20% validation.
    Ensures reproducibility for the final paper by using a fixed seed.
    """
    if not os.path.exists(output_dir):
        # splitfolders automatically creates 'train' and 'val' subfolders
        splitfolders.ratio(input_dir, output=output_dir, 
                           seed=seed, ratio=(.8, .2), 
                           group_prefix=None, move=False)
        print(f"Data successfully split into: {output_dir}")
    else:
        print("Split folder already exists. Skipping split process.")

def get_data_loaders(data_dir, batch_size=32, image_size=224):
    """
    Creates Training and Validation DataLoaders with enhanced Data Augmentation.
    Augmentation helps the model generalize better and reduces overfitting.
    """
    
    # Check if GPU is available to avoid UserWarnings
    use_cuda = torch.cuda.is_available()

    data_transforms = {
        'train': transforms.Compose([
            # scale=(0.4, 1.0) prevents extreme close-ups (zoom).
            # Previously (default 0.08) the model often only saw background -> Underfitting.
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5), # Standard 0.5 instead of 0.1
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15), # Slightly more rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    loaders = {
        x: DataLoader(
            image_datasets[x], 
            batch_size=batch_size, 
            shuffle=True if x == 'train' else False, 
            num_workers=4,
            pin_memory=use_cuda, # Only True if GPU is actually used
            persistent_workers=True 
        )
        for x in ['train', 'val']
    }
    
    class_names = image_datasets['train'].classes
    return loaders, class_names

def save_model(model, path):
    """
    Saves the model state dictionary and ensures the directory exists
    to prevent RuntimeError.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
        
    torch.save(model.state_dict(), path)
    print(f"Model successfully saved to: {path}")

def load_model(model, path, device):
    """Loads a saved model state dictionary or full model object."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Checkpoint is the full model (object) - Legacy Fix
    if isinstance(checkpoint, torch.nn.Module):
        print("Info: Loaded full model object. Extracting state_dict...")
        model.load_state_dict(checkpoint.state_dict())
    
    # Checkpoint is only weights (Standard)
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
        
    return model

def plot_training_history(history, save_dir='plots'):
    """
    Plots training/validation loss and accuracy.
    Saves the result as plot[x].png with an incrementing index x.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    existing_plots = glob.glob(os.path.join(save_dir, 'plot[*].png'))
    indices = []
    for f in existing_plots:
        match = re.search(r'plot\[(\d+)\]', f)
        if match:
            indices.append(int(match.group(1)))
    
    next_idx = max(indices) + 1 if indices else 1
    save_path = os.path.join(save_dir, f'plot[{next_idx}].png')

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss (Left)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy (Right)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Performance plot saved as: {save_path}")

if __name__ == "__main__":
    split_garbage_data("data", "data_split")