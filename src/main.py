import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from train import train_model
from data_loader import get_data_loaders
from model import get_model
from plot_examples import collect_examples, plot_examples
from prepare_data import split_data


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_training_history(history, out_path=None):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Trainings- und Validierungs-Loss')
    plt.xlabel('Epochen')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    plt.title('Trainings- und Validierungs-Accuracy')
    plt.xlabel('Epochen')
    plt.ylabel('Genauigkeit')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()


def save_checkpoint(model, path):
    ensure_dir(os.path.dirname(path) or '.')
    torch.save(model.state_dict(), path)


def save_history(history, path):
    ensure_dir(os.path.dirname(path) or '.')
    torch.save(history, path)


def load_history(path):
    return torch.load(path)


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate/plot pipeline controller')
    parser.add_argument('--mode', choices=['train', 'plot_history', 'plot_examples', 'eval', 'split'], default='train')
    parser.add_argument('--model', choices=['logistic', 'linear'], default='logistic')
    parser.add_argument('--data_dir', default='data_split')
    parser.add_argument('--input', default='data', help='Input data folder used by split mode')
    parser.add_argument('--checkpoint', default='models/checkpoint.pth')
    parser.add_argument('--history', default='models/history.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_correct', type=int, default=5)
    parser.add_argument('--num_incorrect', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_dim = 3 * args.img_size * args.img_size

    if args.mode == 'train':
        # For building we need number of classes from data
        train_loader, val_loader, test_loader, classes = get_data_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
        num_classes = len(classes)
        model = get_model(args.model, input_dim, num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        dataloaders = {'train': train_loader, 'val': val_loader}
        best_model, history = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=args.epochs)

        # persist
        save_checkpoint(best_model, args.checkpoint)
        save_history(history, args.history)
        print(f"Saved checkpoint to {args.checkpoint} and history to {args.history}")

        # Save training results (plot) in the same folder as history/checkpoint
        out_dir = os.path.dirname(args.history) or os.path.dirname(args.checkpoint) or 'models'
        ensure_dir(out_dir)
        plot_path = os.path.join(out_dir, 'training_results.png')
        plot_training_history(history, out_path=plot_path)

    elif args.mode == 'plot_history':
        if not os.path.isfile(args.history):
            raise FileNotFoundError(f"History file not found: {args.history}")
        history = load_history(args.history)
        plot_training_history(history)

    elif args.mode == 'plot_examples':
        # load model checkpoint and plot examples
        train_loader, val_loader, test_loader, classes = get_data_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
        num_classes = len(classes)
        model = get_model(args.model, input_dim, num_classes).to(device)
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        correct, incorrect = collect_examples(model, test_loader, num_correct=args.num_correct, num_incorrect=args.num_incorrect, device=device)
        print(f"Collected {len(correct)} correct and {len(incorrect)} incorrect examples.")
        plot_examples(correct, incorrect, classes)

    elif args.mode == 'eval':
        # simple evaluation on test set
        train_loader, val_loader, test_loader, classes = get_data_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
        num_classes = len(classes)
        model = get_model(args.model, input_dim, num_classes).to(device)
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        # run simple accuracy
        model.eval()
        correct_count = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(1).cpu()
                correct_count += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Test accuracy: {correct_count/total:.4f} ({correct_count}/{total})")

    elif args.mode == 'split':
        # split raw data into train/val/test using prepare_data.split_data
        print(f"Splitting data from '{args.input}' into '{args.data_dir}'")
        split_data(input_folder=args.input, output_folder=args.data_dir)


if __name__ == '__main__':
    main()