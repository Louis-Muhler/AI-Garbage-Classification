from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, img_size=224):
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
        for x in ['train', 'val', 'test']
    }

    # 3. Creating the DataLoaders
    loaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'))
        for x in ['train', 'val', 'test']
    }

    class_names = image_datasets['train'].classes
    return loaders['train'], loaders['val'], loaders['test'], class_names

