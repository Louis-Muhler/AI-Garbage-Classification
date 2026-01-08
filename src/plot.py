"""Plotting helpers: history plots and example prediction images."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_history(history_path: str, out_dir: str = "models/plots"):
    """Load `history.json` and create loss/accuracy plots saved to `out_dir`."""
    p = Path(history_path)
    if not p.exists():
        print(f"History file not found: {history_path}")
        return
        
    with open(p, "r") as fh:
        history = json.load(fh)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if we have data
    if "train_loss" not in history or not history["train_loss"]:
        print("Empty history data.")
        return

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss Plot
    plt.figure()
    plt.plot(epochs, history.get("train_loss", []), label="train_loss")
    plt.plot(epochs, history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png")
    plt.close()

    # Accuracy Plot
    plt.figure()
    plt.plot(epochs, history.get("train_acc", []), label="train_acc")
    plt.plot(epochs, history.get("val_acc", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png")
    plt.close()
    
    print(f"Plots saved to {out_dir}")


def imshow_tensor(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # Standard un-normalization for visualization if ImageNet stats were used
    # If the model used [0,1] or [-1,1], this might need adjustment, but for display 
    # it's usually safer to just clip if we aren't sure of normalization.
    # The attachment implementation:
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    # img = img * np.array(std) + np.array(mean) # Only if normalized with these stats
    # Since loader.py supports simpler normalization, we might just clip.
    # But let's assume raw tensors for now or close to [0,1].
    
    # If values are [-1, 1], map to [0, 1]
    if img.min() < 0:
        img = (img + 1) / 2.0
        
    img = np.clip(img, 0, 1)
    return img


def example_predictions(model: torch.nn.Module, dataset: torch.utils.data.Dataset, class_names, device: torch.device, out_path: str = "models/plots/example_predictions.png"):
    """Find one correctly and one incorrectly predicted sample and save a small plot."""
    model.eval()
    correct_img = None
    correct_label = None
    correct_pred = None
    wrong_img = None
    wrong_label = None
    wrong_pred = None

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True) # shuffle True to get different ones
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu()
            for i in range(x.size(0)):
                img = x[i].cpu()
                t = int(y[i].item())
                p = int(preds[i].item())
                
                if t == p and correct_img is None:
                    correct_img = img
                    correct_label = t
                    correct_pred = p
                if t != p and wrong_img is None:
                    wrong_img = img
                    wrong_label = t
                    wrong_pred = p
            
            if correct_img is not None and wrong_img is not None:
                break

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    # Correct
    if correct_img is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(imshow_tensor(correct_img))
        label_text = class_names[correct_label] if correct_label < len(class_names) else str(correct_label)
        plt.title(f"Correct: {label_text}")
        plt.axis("off")
    else:
        plt.subplot(1, 2, 1)
        plt.text(0.5, 0.5, "No correct sample found", ha="center")
        plt.axis("off")

    # Wrong
    if wrong_img is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(imshow_tensor(wrong_img))
        true_text = class_names[wrong_label] if wrong_label < len(class_names) else str(wrong_label)
        pred_text = class_names[wrong_pred] if wrong_pred < len(class_names) else str(wrong_pred)
        plt.title(f"Wrong: True={true_text}\nPred={pred_text}")
        plt.axis("off")
    else:
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, "No wrong sample found", ha="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
