import torch
import time
import copy

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=10):
    since = time.time()
    
    # Save best model weights for later
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Lists to record metrics for documentation/plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Optional: use a scheduler if optimizer is passed, but maybe adding one here from snippet
    # The snippet used ExponentialLR. Let's add it if not present, but usually passed in.
    # To minimize logic changes, I'll stick to basic training but add the scheduler from the snippet `train.py`.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and a validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluation mode

            running_loss = 0.0
            # Initialize as tensor to ensure .double() works later
            running_corrects = torch.tensor(0, device=device)

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # compute statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # Fix potential type error if running_corrects is not a tensor initially (though it is here)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # save history for later plotting
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # save model weights if this is the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Step scheduler
        scheduler.step()
        
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights and return
    model.load_state_dict(best_model_wts)
    return model, history