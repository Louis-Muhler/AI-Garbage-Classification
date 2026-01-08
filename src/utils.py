import os
import torch
import splitfolders
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
        print(f"✅ Data successfully split into: {output_dir}")
    else:
        print("ℹ️ Split folder already exists. Skipping split process.")

def get_data_loaders(data_dir, batch_size=32, image_size=224):
    """
    Creates Training and Validation DataLoaders with efficient multiprocessing.
    Uses 4 workers as requested to optimize data loading pipeline.
    """
    
    # Define transformations for training and validation
    # Training includes data augmentation to improve model depth and generalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets using ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # Create DataLoaders with 4 workers for parallel CPU processing
    # pin_memory=True speeds up the data transfer from CPU to GPU
    loaders = {
        x: DataLoader(
            image_datasets[x], 
            batch_size=batch_size, 
            shuffle=True if x == 'train' else False, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        for x in ['train', 'val']
    }
    
    # Extract class names for evaluation and documentation in the paper
    class_names = image_datasets['train'].classes
    
    return loaders, class_names

def save_model(model, path):
    """Saves the model state dictionary for final submission."""
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Loads a saved model state dictionary."""
    model.load_state_dict(torch.load(path, map_location=device))
    return model