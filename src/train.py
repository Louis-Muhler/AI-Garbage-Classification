import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_model(model, loaders, criterion, optimizer, num_epochs, device):
    """
    Main training function with time tracking and batch progress logging.
    All metrics are tracked for the final paper documentation.
    """
    start_time_total = time.time()
    model.to(device)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # For progress calculation
            num_batches = len(loaders[phase])
            dataset_size = len(loaders[phase].dataset)

            for i, (inputs, labels) in enumerate(loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Log every 10 batches
                if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                    percent = (i + 1) / num_batches * 100
                    elapsed = time.time() - epoch_start_time
                    print(f"{phase.capitalize()} Batch {i+1}/{num_batches} ({percent:.1f}%) - Elapsed: {elapsed:.1f}s")

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f">>> {phase.capitalize()} Result - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

    # Calculate and print total training time
    total_time = time.time() - start_time_total
    print(f"\n{'='*20}")
    print(f"TRAINING COMPLETE")
    print(f"Total time: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"{'='*20}")

    return model, history