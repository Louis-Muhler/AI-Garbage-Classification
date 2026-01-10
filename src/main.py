import argparse
import os
import sys

# Ensure local modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import from the unified, updated modules
from train import train_model
from model import get_model
from utils import (
    get_data_loaders, 
    split_data, 
    collect_examples, 
    plot_examples, 
    ensure_dir, 
    plot_training_history, 
    save_history, 
    load_history, 
    save_checkpoint
)

def main():
    # ---------------------------------------------------------
    # CLI Arguments Setup
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description='Unified AI Garbage Classification Pipeline')
    
    # Modes
    parser.add_argument('--mode', choices=['train', 'plot_history', 'plot_examples', 'eval', 'split'], default='train', 
                        help='Execution mode: train model, plot existing history, visualize examples, evaluate, or split data.')
    
    # Model Architecture
    parser.add_argument('--model', choices=['logistic', 'simple_cnn', 'custom_resnet', 'resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_large'], 
                        default='custom_resnet', 
                        help='Model architecture to use. "custom_resnet" is the optimized ResNet implementation.')
    
    # Data Paths
    parser.add_argument('--data_dir', default='data_split', help='Directory containing the split dataset (train/val/test)')
    parser.add_argument('--input', default='data', help='Raw input data folder used by split mode')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--img_size', type=int, default=128, help='Input image resolution (e.g. 128x128)')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    
    # Advanced Training Options (New Functionalities)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for faster data transfer to CUDA')
    
    # Transfer Learning Options
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for transfer learning models')
    parser.add_argument('--freeze', action='store_true', help='Freeze backbone layers (transfer learning without fine-tuning). Default is to Unfreeze (fine-tune).')
    parser.add_argument('--label_smoothing', type=float, default=0.15, help='Label smoothing for CrossEntropyLoss')

    # Output / Visualization
    parser.add_argument('--checkpoint', default='checkpoint.pth', help='Filename for model checkpoint')
    parser.add_argument('--history', default='history.json', help='Filename for training history')
    parser.add_argument('--num_correct', type=int, default=5, help='Number of correct examples to plot')
    parser.add_argument('--num_incorrect', type=int, default=5, help='Number of incorrect examples to plot')

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Setup Paths and Device
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join('models', args.model, timestamp)
    
    # Define output paths for training artifacts
    if args.mode == 'train':
        ensure_dir(model_dir)
        checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')
        # Changed default to .json in args, but keeping flexible
        history_path = os.path.join(model_dir, 'history.json')
        plot_path = os.path.join(model_dir, 'training_results.png')
        report_path = os.path.join(model_dir, 'report.txt')
        
        # Save run arguments for reproducibility
        with open(os.path.join(model_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Input dim for linear models (flat size)
    input_dim = 3 * args.img_size * args.img_size

    # ---------------------------------------------------------
    # Execution Modes
    # ---------------------------------------------------------
    if args.mode == 'train':
        print(f"Starting training for model: {args.model}")
        print(f"Output directory: {model_dir}")

        # Ensure data exists
        if not os.path.exists(args.data_dir):
            if os.path.exists(args.input):
                print(f"Data directory '{args.data_dir}' not found. Generating split from '{args.input}'...")
                split_data(args.input, args.data_dir)
            else:
                 raise FileNotFoundError(f"Input data directory '{args.input}' not found. Please provide valid input data.")

        # Data Loading
        train_loader, val_loader, test_loader, classes = get_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        num_classes = len(classes)

        # Model Initialization
        freeze_backbone = args.freeze
        model = get_model(
            args.model,
            input_dim,
            num_classes,
            img_size=args.img_size,
            dropout=args.dropout,
            freeze_backbone=freeze_backbone
        ).to(device)

        # Optimization Setup
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )

        # Training Loop
        dataloaders = {'train': train_loader, 'val': val_loader}
        best_model, history = train_model(
            model, 
            dataloaders, 
            criterion, 
            optimizer, 
            num_epochs=args.epochs, 
            device=device,
            scheduler=scheduler,
            patience=args.patience,
            checkpoint_path=checkpoint_path
        )

        # Persist Final Results
        # Save the state dictionary of the best model found during training
        save_checkpoint(best_model.state_dict(), checkpoint_path)
        
        # Ensure history values are JSON serializable (convert tensors to floats)
        history_serializable = {k: [float((v.cpu().item() if torch.is_tensor(v) else v)) for v in vals] for k, vals in history.items()}
        save_history(history_serializable, history_path)
        print(f"Saved final best checkpoint to {checkpoint_path}")

        # Plot and Save Results
        plot_training_history(history_serializable, save_path=plot_path)
        print(f"Saved training plot to {plot_path}")

        # Generate Report
        with open(report_path, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Best Val Loss: {min(history_serializable['val_loss']):.4f}\n")
            f.write(f"Best Val Acc: {max(history_serializable['val_acc']):.4f}\n")

        # Example Predictions (Visualization)
        print("Collecting example predictions...")
        correct, incorrect = collect_examples(
            best_model, val_loader,
            num_correct=args.num_correct,
            num_incorrect=args.num_incorrect,
            device=device
        )
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
        data_loader_args = dict(batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers) if hasattr(args, 'num_workers') else dict(batch_size=args.batch_size, img_size=args.img_size)
        
        train_loader, val_loader, test_loader, classes = get_data_loaders(args.data_dir, **data_loader_args)
        num_classes = len(classes)

        input_dim = 3 * args.img_size * args.img_size
        freeze_backbone = not args.unfreeze
        model = get_model(
            args.model,
            input_dim,
            num_classes,
            img_size=args.img_size,
            dropout=args.dropout,
            freeze_backbone=freeze_backbone
        ).to(device)

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
        # Same here
        train_loader, val_loader, test_loader, classes = get_data_loaders(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)
        num_classes = len(classes)

        model = get_model(
            args.model,
            input_dim,
            num_classes,
            img_size=args.img_size
        ).to(device)

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

            print(f'Accuracy on test set: {100 * correct / total:.2f}%')
        else:
            print(f"Checkpoint not found at {args.checkpoint}")

    elif args.mode == 'split':
        split_data(args.input, args.data_dir)


if __name__ == '__main__':
    main()
