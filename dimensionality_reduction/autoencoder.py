import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np

from dimensionality_reduction.gaussian import GaussianChannelReduction
from dimensionality_reduction.conv_reducer import ConvReducer


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
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
    def __init__(self, output_channels=1):
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
            nn.Conv2d(128, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(output_channels, 128, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(512, 826, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DeeperAutoencoder(nn.Module):
    def __init__(self, output_channels=1):
        super(DeeperAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(826, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(output_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512),
            nn.Conv2d(512, 826, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
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
        )
        # Two outputs for the mean and log variance
        self.fc_mu = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.fc_logvar = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

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
            nn.Conv2d(512, 826, kernel_size=3, stride=1, padding=1),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class GaussianAutoEncoder(nn.Module):
    def __init__(self, num_input_channels=826, num_reduced_channels=3, priors=None):
        super(GaussianAutoEncoder, self).__init__()
        self.encoder = GaussianChannelReduction(
            num_input_channels, num_reduced_channels, priors
        )
        self.decoder = nn.Sequential(
            # Optional: Additional convolutional layers
            nn.Conv2d(num_reduced_channels, num_input_channels, kernel_size=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


class ConvAutoEncoder(nn.Module):
    def __init__(
        self,
        num_input_channels=826,
        num_reduced_channels=3,
        mean_list=None,
        log_sigma_list=None,
    ):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = ConvReducer(
            num_input_channels, num_reduced_channels, mean_list, log_sigma_list
        )
        self.decoder = nn.Sequential(
            # Optional: Additional convolutional layers
            nn.Conv2d(num_reduced_channels, num_input_channels, kernel_size=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return recon_loss + kl_div


def model_pipeline_autoencoder(
    model,
    trainloader,
    validationloader,
    criterion,
    optimizer,
    config,
    project,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
):
    with wandb.init(project=project, config=config, name=config["model"]):
        config = wandb.config
        model = model.to(device)
        train_loss, val_loss = train_and_validate_autoencoder(
            model,
            trainloader,
            validationloader,
            criterion,
            optimizer,
            epochs,
            model_name,
            device,
            batch_print,
        )
        return model, train_loss, val_loss


def train_and_validate_autoencoder(
    model,
    trainloader,
    validationloader,
    criterion,
    optimizer,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
):
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
    min_val_loss = np.inf
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

            if (i + 1) % batch_print == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / batch_print:.4f}"
                )
                running_loss = 0.0

        # Calculate and print the average loss per epoch
        train_loss = train_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss}, step=epoch + 1)

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
                    encoded_output = model.encoder(inputs)[0].cpu().numpy().squeeze()
                    # Normalize the output image to be in the range [0, 255]
                    if encoded_output.shape[0] == 1:
                        image = np.stack(
                            [encoded_output, encoded_output, encoded_output], axis=-1
                        )
                    elif encoded_output.shape[0] == 3:
                        image = encoded_output.transpose(1, 2, 0)
                    image_min = image.min()
                    image_max = image.max()

                    image = 255 * (image - image_min) / (image_max - image_min)
                    image = image.astype(np.uint8)

                    # Log the RGB image to WandB
                    wandb.log(
                        {
                            "Validation Image": wandb.Image(
                                image, caption=f"Epoch {epoch+1}"
                            )
                        },
                        step=epoch + 1,
                    )

        val_loss = val_running_loss / len(validationloader)
        val_losses.append(val_loss)

        if model_name:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), f"./models/{model_name}.pth")
                model_artifact = wandb.Artifact(f"{model_name}", type="model")
                model_artifact.add_file(f"./models/{model_name}.pth")
                wandb.log_artifact(model_artifact)

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "validation/loss": val_loss}, step=epoch + 1)

    return train_losses, val_losses


def train_and_validate_variational_autoencoder(
    model,
    trainloader,
    validationloader,
    optimizer,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
):
    """
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param epochs: Number of epochs (default=10)
        :param model_name: Model file name (default=None)
    Returns
        train_losses, val_losses: List of losses per epoch
    """
    train_losses, val_losses = [], []
    min_val_loss = np.inf

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs = data[0].to(device)
            outputs, mu, logvar = model(inputs)
            optimizer.zero_grad()
            loss = vae_loss(outputs, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss += loss.item()

            if (
                i + 1
            ) % batch_print == 0:  # Adjust the condition based on your preference
                print(
                    f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / batch_print:.4f}"
                )
                running_loss = 0.0  # Reset running loss after printing

        # Calculate and print the average loss per epoch
        train_loss = train_loss / len(trainloader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss}, step=epoch + 1)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validationloader):
                inputs = data[0].to(device)
                outputs, mu, logvar = model(inputs)

                loss = vae_loss(outputs, inputs, mu, logvar)
                val_running_loss += loss.item()

                if i == 0:
                    encoded_output = (
                        outputs.mean(dim=1, keepdim=True)[0, 0].cpu().numpy()
                    )  # Get the first image and the first channel
                    wandb.log(
                        {
                            "Validation Image": wandb.Image(
                                encoded_output, caption=f"Epoch {epoch+1}"
                            )
                        },
                        step=epoch + 1,
                    )

        if model_name:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), f"./models/{model_name}.pth")
                model_artifact = wandb.Artifact(f"{model_name}", type="model")
                model_artifact.add_file(f"./models/{model_name}.pth")
                wandb.log_artifact(model_artifact)

        val_loss = val_running_loss / len(validationloader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "validation/loss": val_loss}, step=epoch + 1)

    return train_losses, val_losses


def model_pipeline_variational_autoencoder(
    model,
    trainloader,
    validationloader,
    optimizer,
    config,
    project,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
):
    with wandb.init(project=project, config=config):
        config = wandb.config
        model = model.to(device)
        train_loss, val_loss = train_and_validate_variational_autoencoder(
            model,
            trainloader,
            validationloader,
            optimizer,
            epochs,
            model_name,
            device,
            batch_print,
        )
        return model, train_loss, val_loss


def build_gaussian_channel_reducer(
    num_input_channels=826, num_reduced_channels=3, load_from_path=None, device="cuda"
):
    model = GaussianAutoEncoder(num_input_channels, num_reduced_channels).to(device)
    if load_from_path:
        model.load_state_dict(torch.load(load_from_path))
    return model.encoder

def build_conv_channel_reducer(
    num_input_channels=826, num_reduced_channels=3, load_from_path=None, device="cuda"
):
    model = ConvAutoEncoder(num_input_channels, num_reduced_channels).to(device)
    if load_from_path:
        model.load_state_dict(torch.load(load_from_path))
    return model.encoder
