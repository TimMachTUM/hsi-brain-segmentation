import torch
import segmentation_models_pytorch as smp
import wandb
import numpy as np
import cv2
import matplotlib.pyplot as plt

def train_and_validate(model, trainloader, validationloader, criterion, optimizer, epochs=10, model_name=None, device='cuda', batch_print=10):
    """
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
        :param model_name: Model file name (default=None)
    Returns
        train_losses, val_losses: List of losses per epoch
    """
    train_losses, val_losses = [], []
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device).float()
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss += loss.item()

            if (i + 1) % batch_print == 0:  # Adjust the condition based on your preference
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / batch_print:.4f}')
                running_loss = 0.0  # Reset running loss after printing
                
        # Calculate and print the average loss per epoch
        train_loss = train_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
        wandb.log({"epoch":epoch+1, "train/loss": train_loss}, step=epoch+1)

        if model_name:
            torch.save(model.state_dict(), f'./models/{model_name}_epoch{epoch+1}.pth')
            model_artifact = wandb.Artifact(f"{model_name}_epoch{epoch+1}", type="model")
            model_artifact.add_file(f'./models/{model_name}_epoch{epoch+1}.pth')
            wandb.log_artifact(model_artifact)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validationloader):
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        
        val_loss = val_running_loss / len(validationloader)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')
        wandb.log({"epoch":epoch+1, "validation/loss": val_loss}, step=epoch+1)
    
    return train_losses, val_losses

def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate_model(model, dataloader, device, with_wandb=True):
    model.eval()
    recall = 0
    precision = 0
    f1_score = 0
    accuracy = 0
    dice_score = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device).int()
            outputs = model(inputs)
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, labels, mode='binary', threshold=0.5)
            recall += smp.metrics.recall(tp, fp, fn, tn, reduction='micro')
            precision += smp.metrics.precision(tp, fp, fn, tn, reduction='micro')
            f1_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')
            accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro')
            pred = outputs > 0.5
            dice_score += dice_coefficient(pred, labels)
            
    precision /= len(dataloader)
    recall /= len(dataloader)
    f1_score /= len(dataloader)
    accuracy /= len(dataloader)
    dice_score /= len(dataloader)
    if with_wandb:
        wandb.log({"test/precision": precision, "test/recall": recall, "test/f1_score": f1_score, "test/accuracy": accuracy, "test/dice_score": dice_score})
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Dice Score: {dice_score:.4f}, Accuracy: {accuracy:.4f}')


def model_pipeline(model, trainloader, validationloader, testloader, criterion, optimizer, config, project, epochs=10, model_name=None, device='cuda', batch_print=10):
    with wandb.init(project=project, config=config):
        config = wandb.config
        model = model.to(device)
        train_loss, val_loss = train_and_validate(model, trainloader, validationloader, criterion, optimizer, epochs, model_name, device, batch_print)
        evaluate_model(model, testloader, device)
        return model, train_loss, val_loss

def predict(model, data, device):
    model.to(device)
    model.eval()
    data = data.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(data)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
    return prediction.squeeze(1)

def show_overlay(model, data, device):
    prediction = predict(model, data[0], device)
    image = data[2]
    image = image[:,:,[2,1,0]]
    overlay = np.zeros_like(image)
    overlay[prediction.cpu().numpy().squeeze(0) == 1] = [0, 255, 0]
    combined = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(combined)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def show_training_step(output, image):
    prediction = torch.sigmoid(output)
    prediction = (prediction > 0.5).float()
    img = image.squeeze().cpu().numpy()
    img = img[:,:,[2,1,0]]
    prediction = prediction.squeeze()
    overlay = np.zeros_like(img)
    overlay[prediction.cpu().numpy() == 1] = [0, 255, 0]
    combined = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(combined)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()