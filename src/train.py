import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os

def train_model(model, loaders, criterion, optimizer, num_epochs, device, scheduler=None, patience=10, checkpoint_path=None):
    """
    Training loop with:
    - ReduceLROnPlateau (via scheduler)
    - Early Stopping
    - Best Model Saving (returns the best model)
    """
    start_time_total = time.time()
    
    # Best Model Tracking
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0 # Counter for Early Stopping
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    model.to(device)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                # LR Scheduler Step (if not ReduceLROnPlateau, it comes after val)
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # Number of batches for progress display
            dataset_size = len(loaders[phase].dataset)
            num_batches = len(loaders[phase])

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

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # History speichern
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # --- Scheduler Step (ReduceLROnPlateau) ---
                if scheduler is not None:
                     # Check if scheduler is ReduceLROnPlateau (needs metric)
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                        curr_lr = optimizer.param_groups[0]['lr']
                        print(f"Current LR: {curr_lr}")
                    else:
                        scheduler.step()

                # --- Best Model Saving & Early Stopping ---
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"New Best Val Loss: {best_loss:.4f}")
                    # Optional: Save checkpoint
                    if checkpoint_path:
                        torch.save(model.state_dict(), checkpoint_path) # Saves weights only
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve}/{patience} epochs.")

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch Time: {epoch_time:.0f}s")

        # Early Stopping Check
        if epochs_no_improve >= patience:
            print(f"\nEarly Stopping triggered after {epoch+1} epochs!")
            break

    # Load best weights
    print(f"\nTraining complete. Best Val Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    
    # Calculate and print total training time
    total_time = time.time() - start_time_total
    print(f"Total training time: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"{'='*20}")

    return model, history