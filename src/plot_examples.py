import torch
import matplotlib.pyplot as plt
import random
from data_loader import get_data_loaders
from model import get_model


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unnormalize_tensor(img_tensor):
    """Un-normalize a tensor image (C,H,W) to [0,1] range numpy HWC."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.clone()
    img = img * std + mean
    return img.permute(1, 2, 0).cpu().numpy()


def collect_examples(model, loader, num_correct=5, num_incorrect=5, device=None, max_candidates=1000):
    """Collect random examples: up to `max_candidates` candidates are scanned,
    then `num_correct` / `num_incorrect` are sampled randomly from the found
    correct/incorrect pools. This avoids always taking the first images.
    """
    correct_all = []
    incorrect_all = []
    model.eval()
    device = device or get_device()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu()
            labels_cpu = labels.cpu()
            for i in range(images.size(0)):
                p = preds[i].item()
                t = labels_cpu[i].item()
                img_tensor = images[i].cpu()
                img_np = unnormalize_tensor(img_tensor)
                tensor_shape = tuple(img_tensor.shape)
                np_shape = img_np.shape
                if p == t:
                    correct_all.append((img_np, p, t, tensor_shape, np_shape))
                else:
                    incorrect_all.append((img_np, p, t, tensor_shape, np_shape))

            # small safety to avoid scanning entire dataset when it's large
            if len(correct_all) + len(incorrect_all) >= max_candidates:
                break

    # sample randomly from the pools
    n_corr = min(num_correct, len(correct_all))
    n_inc = min(num_incorrect, len(incorrect_all))
    correct = random.sample(correct_all, n_corr) if n_corr > 0 else []
    incorrect = random.sample(incorrect_all, n_inc) if n_inc > 0 else []
    return correct, incorrect


def plot_examples(correct, incorrect, classes):
    ncol = max(len(correct), len(incorrect), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(4 * ncol, 8))
    if ncol == 1:
        axes = axes.reshape(2, 1)

    for idx in range(ncol):
        ax = axes[0, idx]
        if idx < len(correct):
            img, p, t, tensor_shape, np_shape = correct[idx]
            ax.imshow(img)
            ax.set_title(f"Pred: {classes[p]}\nTrue: {classes[t]}\nT:{tensor_shape} N:{np_shape}")
            print(f"Correct[{idx}] tensor shape: {tensor_shape}, numpy shape: {np_shape}")
        ax.axis('off')

        ax = axes[1, idx]
        if idx < len(incorrect):
            img, p, t, tensor_shape, np_shape = incorrect[idx]
            ax.imshow(img)
            ax.set_title(f"Pred: {classes[p]}\nTrue: {classes[t]}\nT:{tensor_shape} N:{np_shape}")
            print(f"Incorrect[{idx}] tensor shape: {tensor_shape}, numpy shape: {np_shape}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()



