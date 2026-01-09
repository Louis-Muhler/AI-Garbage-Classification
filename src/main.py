import torch
import torch.nn as nn
from utils import get_data_loaders, save_model, load_model
from model import SimpleGarbageCNN
from train import train_model
from utils import plot_training_history

# --- CONFIGURATION (Adjust everything here) ---
DATA_DIR = 'data_split'
BATCH_SIZE = 32
LEARNING_RATE = 0.001 #
EPOCHS = 100
IMAGE_SIZE = 128
MODEL_SAVE_PATH = 'models/best_model.pth'
RESUME_TRAINING = False
# ----------------------------------------------

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Define Loss and Optimizer
    loaders, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")

    # 3. Initialize Model
    model = SimpleGarbageCNN(num_classes=num_classes)
    if RESUME_TRAINING:
        print(f"Loading weights from {MODEL_SAVE_PATH} to continue training...")
        try:
            model = load_model(model, MODEL_SAVE_PATH, device)
            print("Successfully loaded model weights.")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting from scratch...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Reduce LR on Plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )

    # 5. Start Training
    print(f"\n--- STARTING TRAINING: {IMAGE_SIZE}x{IMAGE_SIZE} for {EPOCHS} Epochs ---")
    
    model, history = train_model(
        model, loaders, criterion, optimizer, 
        EPOCHS, device, 
        scheduler=scheduler, patience=10,
        checkpoint_path=MODEL_SAVE_PATH
    )

    # 6. Visualization
    plot_training_history(history, save_dir='plots')

    # 7. Save Final Model (Best model is already saved during training via checkpoint_path)
    # But we can save the final state as well if we want, or just rely on 'best'.
    # train_model returns the best state_dict loaded model, so we can save it again to be sure.
    save_model(model, MODEL_SAVE_PATH)
    print(f"Best Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()