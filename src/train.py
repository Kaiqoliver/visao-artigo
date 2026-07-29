import time
import copy
import torch
from torch.optim import lr_scheduler

def train_model(model, dataloaders, criterion, optimizer, device, working_mode, num_epochs=30):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Learning rate policy baseada no artigo: divide por 10 (gamma=0.1) a cada 10 épocas (30/3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_preds = []
        epoch_true = []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    # O GoogLeNet retorna uma Tupla durante o treinamento por conta das perdas auxiliares
                    if phase == 'train' and isinstance(outputs, tuple):
                        loss1 = criterion(outputs[0], labels)
                        loss2 = criterion(outputs[1], labels)
                        loss3 = criterion(outputs[2], labels)
                        loss = loss1 + 0.3 * loss2 + 0.3 * loss3 # Pesos padrão para loss auxiliares
                        _, preds = torch.max(outputs[0], 1)
                    else:
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'val':  
                    epoch_preds.extend(preds.tolist())
                    epoch_true.extend(labels.data.tolist())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} ---> {} Loss: {:.4f} Acc: {:.4f}'.format(working_mode, phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_preds, best_true = epoch_preds, epoch_true

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        # Atualiza a política do learning rate
        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('{} ---> Training complete in {:.0f}m {:.0f}s'.format(working_mode, time_elapsed // 60, time_elapsed % 60))
    print('{} ---> Best val Acc: {:4f}'.format(working_mode, best_acc))

    model.load_state_dict(best_model_wts)
    return {
                'model_ft': model, 
                'best_preds': best_preds, 
                'best_true': best_true, 
                'val_acc_history': val_acc_history, 
                'val_loss_history': val_loss_history, 
                'train_acc_history': train_acc_history, 
                'train_loss_history': train_loss_history
            }