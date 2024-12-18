import torch
import segmentation_models_pytorch as smp
import wandb
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset import build_FIVES_dataloaders
from sklearn.metrics import precision_score
import cv2
from typing import Literal
from ipywidgets import interact, FloatSlider, fixed


def train_and_validate(
    model,
    trainloader,
    validationloader,
    criterion,
    optimizer,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
    with_overlays=False,
):
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
    highest_dice = 0
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
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                if with_overlays and i == 0:
                    log_segmentation_example(model, data, device, epoch)
        val_loss = val_running_loss / len(validationloader)
        val_losses.append(val_loss)

        precision, _, _, _, dice_score = evaluate_model(
            model, validationloader, device, with_wandb=False
        )
        if model_name:
            if dice_score > highest_dice:
                highest_dice = dice_score
                torch.save(model.state_dict(), f"./models/{model_name}.pth")
                model_artifact = wandb.Artifact(f"{model_name}", type="model")
                model_artifact.add_file(f"./models/{model_name}.pth")
                wandb.log_artifact(model_artifact)

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "validation/loss": val_loss}, step=epoch + 1)
        wandb.log({"epoch": epoch + 1, "precision": precision}, step=epoch + 1)
        wandb.log({"epoch": epoch + 1, "dice_score": dice_score}, step=epoch + 1)

    return train_losses, val_losses


def log_segmentation_example(model, data, device, epoch, title="Validation Overlay", channel_reducer=None):
    inputs, labels = data[0][0], data[1][0].to(device).float()
    prediction = predict(model, inputs, device).squeeze(0).cpu().numpy()

    # Convert the labels and prediction to integer masks
    labels = labels.cpu().numpy().squeeze()

    # Assuming inputs are in a suitable format (e.g., normalized between 0 and 1 or uint8)
    if channel_reducer:
        inputs = channel_reducer(inputs.unsqueeze(0).to(device)).squeeze(0)
        
    input_image = inputs.cpu().numpy().squeeze()  # Convert to HWC format
    if input_image.shape[0] == 1:  # Grayscale (single channel)
        input_image = np.stack(
            [input_image, input_image, input_image], axis=-1
        )  # Convert to 3-channel grayscale RGB
    elif input_image.shape[0] == 3:  # RGB image
        input_image = input_image.transpose(1, 2, 0)  # Convert to HWC format

    image_min = input_image.min()
    image_max = input_image.max()

    # Apply min-max normalization to scale values to [0, 255]
    input_image = 255 * (input_image - image_min) / (image_max - image_min)
    input_image = input_image.astype(np.uint8)  # Convert to uint8 after scaling

    # Prepare the masks for wandb logging
    mask_data = {
        "predictions": {
            "mask_data": prediction,  # The predicted mask
            "class_labels": {
                0: "Background",
                1: "Blood Vessel",
            },  # Change according to your classes
        },
        "ground_truth": {
            "mask_data": labels,  # The ground truth mask
            "class_labels": {
                0: "Background",
                1: "Blood Vessel",
            },  # Change according to your classes
        },
    }

    # Log the image and masks using wandb
    wandb.log(
        {title: wandb.Image(input_image, masks=mask_data, caption=f"Epoch {epoch+1}")},
        step=epoch + 1,
    )


def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def evaluate_model(model, dataloader, device, with_wandb=True, threshold=0.5):
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
            tp, fp, fn, tn = smp.metrics.get_stats(
                outputs, labels, mode="binary", threshold=threshold
            )
            recall += smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
            precision += smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            f1_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
            pred = outputs > 0.5
            dice_score += dice_coefficient(pred, labels)

    precision /= len(dataloader)
    recall /= len(dataloader)
    f1_score /= len(dataloader)
    accuracy /= len(dataloader)
    dice_score /= len(dataloader)
    if with_wandb:
        wandb.log(
            {
                "test/precision": precision,
                "test/recall": recall,
                "test/f1_score": f1_score,
                "test/accuracy": accuracy,
                "test/dice_score": dice_score,
            }
        )
    print(
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Dice Score: {dice_score:.4f}, Accuracy: {accuracy:.4f}"
    )
    return precision, recall, f1_score, accuracy, dice_score


def model_pipeline(
    model,
    trainloader,
    validationloader,
    testloader,
    criterion,
    optimizer,
    config,
    project,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
    evaluate=True,
    with_overlays=False,
):
    with wandb.init(project=project, config=config, name=model_name):
        config = wandb.config
        train_loss, val_loss = train_and_validate(
            model,
            trainloader,
            validationloader,
            criterion,
            optimizer,
            epochs,
            model_name,
            device,
            batch_print,
            with_overlays,
        )
        if evaluate:
            if model_name:
                model.load_state_dict(torch.load(f"./models/{model_name}.pth"))
            evaluate_model(model, testloader, device)
        return model, train_loss, val_loss


def predict(model, data, device, with_sigmoid=True, threshold=0.5):
    model.to(device)
    model.eval()
    data = data.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(data)
        if with_sigmoid:
            prediction = torch.sigmoid(prediction)
        prediction = (prediction > threshold).float()
    return prediction.squeeze(1)


def show_overlay(
    model,
    data,
    device,
    with_sigmoid=True,
    title=None,
    with_precision=True,
    with_postprocessing=True,
    threshold=0.5,
):
    prediction = predict(
        model, data[0], device, with_sigmoid=with_sigmoid, threshold=threshold
    )
    image = data[2]
    labels = data[1]

    prediction_np = prediction.cpu().numpy().squeeze(0)

    _plot_overlay(
        title, with_precision, with_postprocessing, image, labels, prediction_np
    )


def show_interactive_overlay(model, data, device, title):
    image = data[2]
    labels = data[1]
    model.to(device)
    model.eval()
    data = data[0].unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(data)

    def apply_slider(
        threshold,
        prediction,
        title,
        with_sigmoid=True,
        with_postprocessing=True,
        with_precision=True,
    ):
        with torch.no_grad():
            if with_sigmoid:
                prediction = torch.sigmoid(prediction)
            prediction = (prediction > threshold).float()
        prediction = prediction.squeeze(1)

        prediction_np = prediction.cpu().numpy().squeeze(0)
        _plot_overlay(
            title, with_precision, with_postprocessing, image, labels, prediction_np
        )

    threshold_slider = FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Threshold:",
        readout_format=".2f",
    )
    interact(
        apply_slider,
        threshold=threshold_slider,
        prediction=fixed(prediction),
        title=title,
    )


def _plot_overlay(
    title, with_precision, with_postprocessing, image, labels, prediction_np
):
    if with_postprocessing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        prediction_np = cv2.morphologyEx(prediction_np, cv2.MORPH_OPEN, kernel)

    labels_np = labels.cpu().numpy().squeeze(0)

    overlay = np.zeros_like(image)
    # Green for prediction
    overlay[prediction_np == 1] = [0, 255, 0]
    # Blue for ground truth
    overlay[labels_np == 1] = [0, 0, 255]
    # Cyan for intersection (both prediction and ground truth are 1)
    overlay[(prediction_np == 1) & (labels_np == 1)] = [0, 255, 255]
    combined = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    if with_precision:
        precision = precision_score(labels_np.flatten(), prediction_np.flatten())
        if title:
            title += f" - Precision: {precision:.2f}"
        else:
            title = f"Precision: {precision:.2f}"

    plt.figure(figsize=(10, 10))
    plt.imshow(combined)
    plt.axis("off")  # Turn off axis numbers and ticks
    if title:
        plt.suptitle(title)
    plt.show()


def show_training_step(output, image):
    prediction = torch.sigmoid(output)
    prediction = (prediction > 0.5).float()
    img = image.squeeze().cpu().numpy()
    img = img[:, :, [2, 1, 0]]
    prediction = prediction.squeeze()
    overlay = np.zeros_like(img)
    overlay[prediction.cpu().numpy() == 1] = [0, 255, 0]
    combined = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(combined)
    plt.axis("off")  # Turn off axis numbers and ticks
    plt.show()


def build_segmentation_model(
    encoder,
    architecture: Literal[
        "Unet",
        "UnetPlusPlus",
        "MAnet",
        "Linknet",
        "FPN",
        "PSPNet",
        "DeepLabV3Plus",
        "PAN",
    ] = "Unet",
    device="cuda",
    in_channels=1,
):
    if architecture == "Unet":
        model = smp.Unet(
            encoder, encoder_weights=None, in_channels=in_channels, classes=1
        )
    elif architecture == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder, encoder_weights=None, in_channels=in_channels, classes=1
        )
    elif architecture == "MAnet":
        model = smp.FPN(
            encoder, encoder_weights=None, in_channels=in_channels, classes=1
        )
    elif architecture == "Linknet":
        model = smp.Linknet(
            encoder, encoder_weights=None, in_channels=in_channels, classes=1
        )
    elif architecture == "FPN":
        model = smp.FPN(
            encoder, encoder_weights=None, in_channels=in_channels, classes=1
        )
    elif architecture == "PSPNet":
        model = smp.PSPNet(
            encoder, encoder_weights=None, in_channels=in_channels, classes=1
        )
    elif architecture == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder, encoder_weights=None, in_channels=in_channels, classes=1
        )
    elif architecture == "PAN":
        model = smp.PAN(
            encoder, encoder_weights=None, in_channels=in_channels, classes=1
        )
    return model.to(device)


def build_criterion(loss: Literal["Dice", "BCE", "Focal"] = "Dice", gamma=2):
    if loss == "Dice":
        return smp.losses.DiceLoss(mode="binary")
    elif loss == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    elif loss == "Focal":
        return smp.losses.FocalLoss(mode="binary", gamma=gamma)


def build_optimizer(
    model, learning_rate=0.001, optimizer: Literal["Adam", "SGD"] = "Adam"
):
    if optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    return torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        encoder = config["encoder"]
        device = config["device"]
        architecture = config["architecture"]
        loss = config["loss"]
        learning_rate = config["learning_rate"]
        optimizer = config["optimizer"]
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        channels = config["channels"]
        proportion_augmented_data = config["proportion_augmented_data"]
        if "gamma" in config:
            gamma = config["gamma"]

        model = build_segmentation_model(
            encoder, architecture=architecture, device=device, in_channels=channels
        )
        criterion = build_criterion(loss, gamma=gamma if "gamma" in config else 2)
        optimizer = build_optimizer(
            model, learning_rate=learning_rate, optimizer=optimizer
        )
        trainloader, validationloader, testloader = build_FIVES_dataloaders(
            batch_size=batch_size,
            proportion_augmented_data=proportion_augmented_data,
            num_channels=channels,
            width=config["width"],
            height=config["height"],
        )

        train_losses, val_losses = [], []

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

                if (i + 1) % 10 == 0:  # Adjust the condition based on your preference
                    print(
                        f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}"
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
                    inputs, labels = data[0].to(device), data[1].to(device).float()
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    if i == 0:
                        log_segmentation_example(model, data, device, epoch)

            val_loss = val_running_loss / len(validationloader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "validation/loss": val_loss}, step=epoch + 1)
            precision, _, _, _, dice_score = evaluate_model(
                model, testloader, device, with_wandb=False
            )
            wandb.log({"epoch": epoch + 1, "precision": precision}, step=epoch + 1)
            wandb.log({"epoch": epoch + 1, "dice_score": dice_score}, step=epoch + 1)

            if dice_score > 0.8:
                torch.save(model.state_dict(), f"./models/{run.name}.pth")

        del model
        torch.cuda.empty_cache()
