import torch
import torch.nn as nn
from src.utils import get_data_loaders, save_model
from src.model import SimpleGarbageCNN
from src.train import train_model

# --- CONFIGURATION (Adjust everything here) ---
DATA_DIR = 'data/split_data'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
IMAGE_SIZE = 224
MODEL_SAVE_PATH = 'models/baseline_model.pth'
# ----------------------------------------------

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    loaders, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")

    # 3. Initialize "Plain" CNN
    model = SimpleGarbageCNN(num_classes=num_classes)

    # 4. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Start Training
    trained_model, history = train_model(
        model, loaders, criterion, optimizer, EPOCHS, device
    )

    # 6. Save final result
    save_model(trained_model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()