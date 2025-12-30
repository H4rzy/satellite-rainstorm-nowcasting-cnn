from lib import *
from imagetransform import *
from dataset import *

def update_params(net):
    params_to_update_1 = []  # phần features đầu (layer1, layer2)
    params_to_update_2 = []  # phần features cuối (layer3, layer4)
    params_to_update_3 = []  # phần output (fc cuối)

    for name, param in net.named_parameters():
        if "conv1" in name or "bn1" in name or "layer1" in name or "layer2" in name:
            param.requires_grad = True
            params_to_update_1.append(param)
        elif "layer3" in name or "layer4" in name:
            param.requires_grad = True
            params_to_update_2.append(param)
        elif "fc" in name:
            param.requires_grad = True
            params_to_update_3.append(param)
    return params_to_update_1, params_to_update_2, params_to_update_3



def train_model(model, dataloader_dict, criterion, optimizer, num_epoch, scheduler):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_loss = float('inf') # theo dõi loss tốt nhất
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epoch):
        print(f'\nEpoch {epoch + 1}/{num_epoch}')
        print('-' * 30)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_correct = 0

            for inputs, labels in tqdm(dataloader_dict[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_correct += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_correct.double() / len(dataloader_dict[phase].dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'best_model_wiser.pth')
                    print(f"New best model saved with val loss: {best_loss:.4f}")
                    
                # if epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(model.state_dict())
                #     torch.save(model.state_dict(), 'best_model_dumb.pth')
                #     print(f"New best model saved with val acc: {best_acc:.4f}")
                    
                scheduler.step(epoch_loss)
                
                
                for param_group in optimizer.param_groups:
                    print(f"--> Current LR: {param_group['lr']:.6f}")

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.4f}')

    print("\nTraining Completed!")
    print("-" * 40)
    print(f"Final Train Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final Val Loss:   {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.4f}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies
