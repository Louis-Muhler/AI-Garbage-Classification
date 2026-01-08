from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=4, pin_memory=False):
    """
    Sets up the data loaders for training, validation, and testing.
    """
    # 1. Image transformations (Systematic Preprocessing)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),  # Simple augmentation for better depth
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Loading the datasets using ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x])
        for x in ['train', 'val']
    }

    # Handle optional test split
    if os.path.exists(f"{data_dir}/test"):
        image_datasets['test'] = datasets.ImageFolder(root=f"{data_dir}/test", transform=data_transforms['test'])
        test_exists = True
    else:
        test_exists = False

    # 3. Creating the DataLoaders
    # Optimize data loading with persistent workers and pinning memory if feasible
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    
    # On Windows, persistent_workers can sometimes cause issues or be slow to start, 
    # but generally good for speed. Only enable if num_workers > 0.
    if num_workers > 0:
        loader_args['persistent_workers'] = True
        loader_args['prefetch_factor'] = 2

    loaders = {
        x: DataLoader(image_datasets[x], shuffle=(x == 'train'), **loader_args)
        for x in image_datasets
    }

    class_names = image_datasets['train'].classes
    
    # If no test set, use validation set as test set placeholder to avoid breakages
    test_loader = loaders['test'] if test_exists else loaders['val']
    
    return loaders['train'], loaders['val'], test_loader, class_names

