from pathlib import Path
from typing import Tuple, Dict
import torch
from torchvision import transforms, datasets


def get_dataloaders(data_dir: str = "data_split",
                    img_size: int = 128,
                    batch_size: int = 16,
                    num_workers: int = 4,
                    normalize: str = "zero_one") -> Tuple[Dict[str, torch.utils.data.DataLoader], dict]:
    """Create train/val/test DataLoaders.

    Expects folder structure: data_split/train, data_split/val, data_split/test
    with class subfolders.
    Returns a dict of loaders and a class_to_idx mapping.
    """
    data_dir = Path(data_dir)
    # Standard ImageNet mean/std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Use ToTensor to scale pixel values to [0,1]. Optionally rescale to [-1,1].
    base_train = [transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    base_val = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]

    # Note: If we use pre-trained models later, we might need the specific mean/std normalization.
    # For now, keeping it simple or per user spec. 
    # If normalize is 'minus1_1', we adjust. 
    
    # Adding Normalization matching default requirement if needed, 
    # but the attachment didn't specifically enforce ImageNet norm in the loop unless requested.
    # However the existing workspace had normalization. I will replicate the existing workspace's normalization 
    # or follow the attachment? The attachment code has logic for "minus1_1" or "zero_one".
    # I will stick to the attachment logic for now to ensure I'm using the "new" code.
    
    if normalize == "minus1_1":
        # move from [0,1] -> [-1,1]
        base_train.append(transforms.Lambda(lambda t: t * 2.0 - 1.0))
        base_val.append(transforms.Lambda(lambda t: t * 2.0 - 1.0))
    else:
        # keep [0,1]
        pass
        
    train_transforms = transforms.Compose(base_train)
    val_test_transforms = transforms.Compose(base_val)

    datasets_map = {}
    loaders = {}

    for split, tf in [("train", train_transforms), ("val", val_test_transforms), ("test", val_test_transforms)]:
        path = data_dir / split
        if not path.exists():
            datasets_map[split] = None
            loaders[split] = None
            continue
        ds = datasets.ImageFolder(root=str(path), transform=tf)
        shuffle = True if split == "train" else False
        
        # Adjust num_workers if on Windows/issues arise, but assuming 4 is fine
        import os
        workers = num_workers if os.name != 'nt' else 0 # Safer on Windows
        
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
        datasets_map[split] = ds
        loaders[split] = loader

    class_to_idx = None
    if datasets_map.get("train") is not None:
        class_to_idx = datasets_map["train"].class_to_idx
    else:
        # fallback to val or test
        for s in ("val", "test"):
            if datasets_map.get(s) is not None:
                class_to_idx = datasets_map[s].class_to_idx
                break

    return loaders, class_to_idx
