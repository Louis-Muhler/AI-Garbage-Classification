import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from datetime import datetime

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
    
    # Save as JSON for readability
    json_path = str(path).replace('.pt', '.json')
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(json_path, 'w') as f:
        json.dump(history_serializable, f, indent=4)


def load_history(path):
    return torch.load(path)


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate/plot pipeline controller')
    parser.add_argument('--mode', choices=['train', 'plot_history', 'plot_examples', 'eval', 'split'], default='train')
    parser.add_argument('--model', choices=['logistic', 'linear', 'cnn', 'slim_cnn'], default='slim_cnn')
    parser.add_argument('--data_dir', default='data_split')
    parser.add_argument('--input', default='data', help='Input data folder used by split mode')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_correct', type=int, default=5)
    parser.add_argument('--num_incorrect', type=int, default=5)
    parser.add_argument('--checkpoint', default='checkpoint.pth', help='Path to model checkpoint')
    parser.add_argument('--history', default='history.pt', help='Path to training history')
    args = parser.parse_args()

    # Create model specific directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join('models', args.model, timestamp)
    ensure_dir(model_dir)

    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')
    history_path = os.path.join(model_dir, 'history.pt')
    plot_path = os.path.join(model_dir, 'training_results.png')
    report_path = os.path.join(model_dir, 'report.txt')

    # Save run arguments
    with open(os.path.join(model_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Input dim for linear models (flat size)
    input_dim = 3 * args.img_size * args.img_size

    if args.mode == 'train':
        print(f"Starting training for model: {args.model}")
        print(f"Output directory: {model_dir}")
        
        train_loader, val_loader, test_loader, classes = get_data_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
        num_classes = len(classes)
        
        model = get_model(args.model, input_dim, num_classes, img_size=args.img_size).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        dataloaders = {'train': train_loader, 'val': val_loader}
        best_model, history = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=args.epochs)

        # persist
        save_checkpoint(best_model, checkpoint_path)
        save_history(history, history_path)
        print(f"Saved checkpoint to {checkpoint_path} and history to {history_path}")

        # Save training results (plot)
        plot_training_history(history, out_path=plot_path)
        print(f"Saved plot to {plot_path}")
        
        # Save simple report
        with open(report_path, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Val Loss: {history['val_loss'][-1]:.4f}\n")
            f.write(f"Final Train Acc: {history['train_acc'][-1]:.4f}\n")
            f.write(f"Final Val Acc: {history['val_acc'][-1]:.4f}\n")

        # Automatically collect and plot examples
        print("Collecting example predictions...")
        correct, incorrect = collect_examples(best_model, val_loader, 
                                            num_correct=args.num_correct, 
                                            num_incorrect=args.num_incorrect, 
                                            device=device)
        example_plot_path = os.path.join(model_dir, 'example_predictions.png')
        plot_examples(correct, incorrect, classes, out_path=example_plot_path)
        print(f"Saved example predictions to {example_plot_path}")

    elif args.mode == 'plot_history':
        if os.path.exists(args.history):
            history = load_history(args.history)
            plot_training_history(history)
        else:
            print(f"No history found at {args.history}")

    elif args.mode == 'plot_examples':
        train_loader, val_loader, test_loader, classes = get_data_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
        num_classes = len(classes)
        model = get_model(args.model, input_dim, num_classes, img_size=args.img_size).to(device)
        
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded checkpoint from {args.checkpoint}")
            
            correct, incorrect = collect_examples(model, val_loader, 
                                                num_correct=args.num_correct, 
                                                num_incorrect=args.num_incorrect, 
                                                device=device)
            plot_examples(correct, incorrect, classes) 
        else:
            print(f"Checkpoint not found at {args.checkpoint}")

    elif args.mode == 'eval':
        train_loader, val_loader, test_loader, classes = get_data_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
        num_classes = len(classes)
        model = get_model(args.model, input_dim, num_classes, img_size=args.img_size).to(device)
        
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            model.eval()
            
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            print(f'Accuracy of the network on test images: {100 * correct / total:.2f}%')
        else:
            print(f"Checkpoint not found at {args.checkpoint}")

    elif args.mode == 'split':
        split_data(args.input, args.data_dir)


if __name__ == '__main__':
    main()