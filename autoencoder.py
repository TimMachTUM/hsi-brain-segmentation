import torch
import torch.nn as nn
import wandb
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(826, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResidualBlock(512),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResidualBlock(512),
            nn.Conv2d(512, 826, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def model_pipeline_autoencoder(model, trainloader, validationloader, criterion, optimizer, config, project, epochs=10, model_name=None, device='cuda', batch_print=10):
    with wandb.init(project=project, config=config):
        config = wandb.config
        model = model.to(device)
        train_loss, val_loss = train_and_validate_autoencoder(model, trainloader, validationloader, criterion, optimizer, epochs, model_name, device, batch_print)
        return model, train_loss, val_loss
    

def train_and_validate_autoencoder(model, trainloader, validationloader, criterion, optimizer, epochs=10, model_name=None, device='cuda', batch_print=10):
    """
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=10)
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
            inputs = data[0].to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, inputs)
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
                inputs = data[0].to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, inputs)
                val_running_loss += loss.item()

                if i == 0:
                    encoded_output = model.encoder(inputs)[0, 0].cpu().numpy()  # Get the first image and the first channel
                    encoded_output = (encoded_output - encoded_output.min()) / (encoded_output.max() - encoded_output.min()) * 255
                    encoded_output = encoded_output.astype(np.uint8)
                    wandb.log({"Validation Image": wandb.Image(encoded_output, caption=f"Epoch {epoch+1}")}, step=epoch+1)
        
        val_loss = val_running_loss / len(validationloader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')
        wandb.log({"epoch":epoch+1, "validation/loss": val_loss}, step=epoch+1)
    
    return train_losses, val_losses