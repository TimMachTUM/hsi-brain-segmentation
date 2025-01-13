import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from FADA.segmentation_model import (
    SegmentationModelFADA,
    SegmentationWithChannelReducerFADA,
)
from FADA.discriminator import PixelDiscriminator
from FADA.classifier import Classifier
from FADA.feature_extractor import FeatureExtractor
from dataset import (
    HSIDataset,
    build_FIVES_random_crops_dataloaders,
    build_hsi_dataloader,
    build_hsi_testloader,
)
from segmentation_util import load_model, log_segmentation_example, evaluate_model
from segmentation_util import build_segmentation_model, build_criterion, build_optimizer
from segmentation_models_pytorch.encoders import get_encoder
from dimensionality_reduction.autoencoder import (
    build_gaussian_channel_reducer,
    build_conv_channel_reducer,
)
import os


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float() * F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights * torch.sum(loss, dim=1))


def model_pipeline(
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target,
    testloader_target,
    config,
    project,
    device="cuda",
    batch_print=10,
    evaluate=True,
    with_overlays=False,
    save_wandb=True,
):
    with wandb.init(project=project, config=config, name=config["model"]):
        config = wandb.config
        return init_model_and_train(
            trainloader_source,
            validationloader_source,
            testloader_source,
            trainloader_target,
            testloader_target,
            config,
            device,
            batch_print,
            evaluate,
            with_overlays,
            save_wandb=save_wandb,
        )


def init_model_and_train(
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target,
    testloader_target,
    config,
    device,
    batch_print,
    evaluate,
    with_overlays,
    save_wandb=True,
):
    segmentation_model = build_segmentation_model(
        config.encoder, config.architecture, device, in_channels=config.in_channels
    )
    if "pretrained" in config:
        print(f"Loading pretrained model from {config.pretrained}")
        segmentation_model = load_model(segmentation_model, config.pretrained, device)

    reducer = None
    if "gaussian" in config:
        print("Using Gaussian Channel Reduction")
        reducer = build_gaussian_channel_reducer(
            num_input_channels=826,
            num_reduced_channels=config.in_channels,
            load_from_path=config.gaussian,
            device=device,
        )
    elif "conv_reducer" in config:
        print("Using Convolutional Channel Reduction")
        reducer = build_conv_channel_reducer(
            num_input_channels=826,
            num_reduced_channels=config.in_channels,
            load_from_path=config.conv_reducer,
            device=device,
        )

    feature_extractor = FeatureExtractor(segmentation_model).to(device)
    classifier = Classifier(segmentation_model).to(device)
    discriminator = PixelDiscriminator(
        input_nc=get_encoder(config.encoder, config.in_channels).out_channels[-1],
        ndf=config.ndf,
        num_classes=1,
    ).to(device)

    criterion_segmentation = build_criterion(config.seg_loss)
    optimizer_fea = build_optimizer(
        feature_extractor,
        learning_rate=config.learning_rate_fea,
        optimizer=config.optimizer,
    )
    optimizer_cls = build_optimizer(
        classifier,
        learning_rate=config.learning_rate_cls,
        optimizer=config.optimizer,
    )
    optimizer_dis = build_optimizer(
        discriminator,
        learning_rate=config.learning_rate_dis,
        optimizer=config.optimizer,
    )
    with_contrastive_loss = "contrastive_loss" in config and config.contrastive_loss
    penalize_rings_weight = (
        config.penalize_rings_weight if "penalize_rings_weight" in config else 0.25
    )

    train_loss, domain_loss, val_loss_source, val_loss_target = train_and_validate(
        feature_extractor,
        classifier,
        discriminator,
        trainloader_source,
        validationloader_source,
        testloader_source,
        trainloader_target,
        testloader_target,
        criterion_segmentation,
        optimizer_fea,
        optimizer_cls,
        optimizer_dis,
        epochs=config.epochs,
        model_name=config.model,
        device=device,
        batch_print=batch_print,
        with_overlays=with_overlays,
        hsi_reducer=reducer,
        save_wandb=save_wandb,
        with_contrastive_loss=with_contrastive_loss,
        penalize_rings_weight=penalize_rings_weight,
    )
    if evaluate:
        if config.model:
            model = (
                SegmentationModelFADA(feature_extractor, classifier)
                if not reducer
                else SegmentationWithChannelReducerFADA(
                    reducer, feature_extractor, classifier
                )
            )
            model = load_model(model, f"./models/{config.model}.pth", device)

        evaluate_model(model, testloader_target, device)
    return model, train_loss, domain_loss, val_loss_source, val_loss_target


def train_and_validate(
    feature_extractor,
    classifier,
    discriminator,
    trainloader_source,
    validationloader_source,
    testloader_source,
    trainloader_target,
    testloader_target,
    criterion_segmentation,
    optimizer_fea,
    optimizer_cls,
    optimizer_D,
    epochs=10,
    model_name=None,
    device="cuda",
    batch_print=10,
    with_overlays=False,
    hsi_reducer=None,
    save_wandb=True,
    with_contrastive_loss=False,
    penalize_rings_weight=0.25,
):
    train_losses, domain_losses, val_losses_source, val_losses_target = [], [], [], []
    highest_dice = 0

    # Create infinite iterators
    source_domain_iter = itertools.cycle(trainloader_source)
    target_domain_iter = itertools.cycle(trainloader_target)

    # Calculate total number of batches per epoch
    num_batches = max(
        len(trainloader_source),
        len(trainloader_target),
    )
    # Define the contrastive loss function
    contrastive_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        feature_extractor.train()
        classifier.train()
        discriminator.train()

        running_loss = 0.0
        train_loss = 0.0
        domain_loss = 0.0

        for batch_idx in range(num_batches):
            optimizer_fea.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_D.zero_grad()
            # Segmentation Training Step
            seg_data = next(source_domain_iter)
            src_input, src_label = (
                seg_data[0].to(device),
                seg_data[1].to(device).float(),
            )
            # Get target domain data for the current window
            target_domain_data = next(target_domain_iter)
            tgt_input = target_domain_data[0].to(device)

            src_size = src_input.shape[-2:]
            tgt_size = tgt_input.shape[-2:]

            # Forward pass through feature_extractor and classifier
            src_features = feature_extractor(src_input)
            src_pred = classifier(src_features)
            temperature = 1.8
            src_pred = src_pred.div(temperature)
            loss_seg = criterion_segmentation(src_pred, src_label)
            loss_seg.backward()

            # generate soft labels
            src_soft_label = F.softmax(src_pred, dim=1).detach().to(device)
            src_soft_label[src_soft_label > 0.9] = 0.9

            tgt_features = feature_extractor(
                hsi_reducer(tgt_input) if hsi_reducer else tgt_input
            )
            tgt_pred = classifier(tgt_features)
            tgt_pred = tgt_pred.div(temperature)
            tgt_soft_label = F.softmax(tgt_pred, dim=1).detach().to(device)
            tgt_soft_label[tgt_soft_label > 0.9] = 0.9

            tgt_D_pred = discriminator(tgt_features[-1], tgt_size)
            loss_adv_tgt = 0.001 * soft_label_cross_entropy(
                tgt_D_pred,
                torch.cat(
                    (tgt_soft_label, torch.zeros_like(tgt_soft_label).to(device)), dim=1
                ),
            )
            loss_adv_tgt.backward(retain_graph=True)

            if with_contrastive_loss:
                black_ring_mask = target_domain_data[1].to(device).float()
                tgt_pred_sigmoid = torch.sigmoid(tgt_pred)
                contrastive_loss = torch.mean(tgt_pred_sigmoid * black_ring_mask)
                weighted_contrastive_loss = penalize_rings_weight * contrastive_loss
                running_loss += weighted_contrastive_loss.item()
                weighted_contrastive_loss.backward()

            optimizer_fea.step()
            optimizer_cls.step()

            optimizer_D.zero_grad()
            src_D_pred = discriminator(src_features[-1].detach().to(device), src_size)
            loss_D_src = 0.5 * soft_label_cross_entropy(
                src_D_pred,
                torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1),
            )
            loss_D_src.backward()

            tgt_D_pred = discriminator(tgt_features[-1].detach().to(device), tgt_size)
            loss_D_tgt = 0.5 * soft_label_cross_entropy(
                tgt_D_pred,
                torch.cat(
                    (torch.zeros_like(tgt_soft_label).to(device), tgt_soft_label), dim=1
                ),
            )
            loss_D_tgt.backward()

            optimizer_D.step()
            # Update running losses
            running_loss += (
                loss_seg.item()
                + loss_D_src.item()
                + loss_D_tgt.item()
                + loss_adv_tgt.item()
            )
            train_loss += running_loss
            domain_loss += loss_D_src.item() + loss_D_tgt.item() + loss_adv_tgt.item()

            if (batch_idx + 1) % batch_print == 0:
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Total Loss: {running_loss/batch_print:.4f}, Segmentation Loss: {loss_seg.item():.4f}, Domain Loss Source: {loss_D_src.item():.4f}, Domain Loss Target: {loss_D_tgt.item():.4f}, Adversarial Loss Target: {loss_adv_tgt.item():.4f}"
                )
                running_loss = 0.0  # Reset running loss after printing

        train_loss = train_loss / num_batches
        domain_loss = domain_loss / num_batches
        train_losses.append(train_loss)
        domain_losses.append(domain_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train/loss": train_loss}, step=epoch + 1)
        wandb.log(
            {"epoch": epoch + 1, "train/domain_loss": domain_loss}, step=epoch + 1
        )

        # Validation
        if hsi_reducer:
            model = SegmentationWithChannelReducerFADA(
                hsi_reducer, feature_extractor, classifier
            ).to(device)
        else:
            model = SegmentationModelFADA(feature_extractor, classifier).to(device)

        model.eval()
        val_running_loss_target = 0.0
        val_running_loss_source = 0.0
        with torch.no_grad():
            # Validation Phase on target data
            for i, data in enumerate(testloader_target):
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)
                loss = criterion_segmentation(outputs, labels)
                val_running_loss_target += loss.item()
                log_segmentation_example(
                    model,
                    data,
                    device,
                    epoch,
                    title=f"Validation Overlay HSI {i}",
                    channel_reducer=hsi_reducer,
                )

            # validation phase on source data
            for i, data in enumerate(validationloader_source):
                inputs, labels = data[0].to(device), data[1].to(device).float()
                outputs = model(inputs)
                loss = criterion_segmentation(outputs, labels)
                val_running_loss_source += loss.item()
                if with_overlays and i == 0:
                    log_segmentation_example(
                        model, data, device, epoch, title="Validation Overlay FIVES"
                    )

        val_loss_source = val_running_loss_source / len(validationloader_source)
        val_loss_target = val_running_loss_target / len(testloader_target)

        wandb.log(
            {"epoch": epoch + 1, "val/loss_source": val_loss_source}, step=epoch + 1
        )
        wandb.log(
            {"epoch": epoch + 1, "val/loss_target": val_loss_target}, step=epoch + 1
        )

        val_losses_source.append(val_loss_source)
        val_losses_target.append(val_loss_target)

        # Evaluate model performance
        print("Evaluating model performance on source data")
        precision_source, _, _, _, dice_score_source = evaluate_model(
            model, testloader_source, device, with_wandb=False
        )

        print("Evaluating model performance on target data")
        precision_target, _, _, _, dice_score_target = evaluate_model(
            model, testloader_target, device, with_wandb=False
        )
        if model_name:
            if dice_score_target > highest_dice:
                highest_dice = dice_score_target
                torch.save(model.state_dict(), f"./models/{model_name}.pth")
                if save_wandb:
                    model_artifact = wandb.Artifact(f"{model_name}", type="model")
                    model_artifact.add_file(f"./models/{model_name}.pth")
                    wandb.log_artifact(model_artifact)

        print(
            f"Epoch {epoch+1}, Validation Loss Source: {val_loss_source:.4f}, Validation Loss Target: {val_loss_target:.4f}"
        )
        wandb.log(
            {"epoch": epoch + 1, "precision/source": precision_source}, step=epoch + 1
        )
        wandb.log(
            {"epoch": epoch + 1, "dice_score/source": dice_score_source}, step=epoch + 1
        )
        wandb.log(
            {"epoch": epoch + 1, "precision/target": precision_target}, step=epoch + 1
        )
        wandb.log(
            {"epoch": epoch + 1, "dice_score/target": dice_score_target}, step=epoch + 1
        )

        del model

    return train_losses, domain_losses, val_losses_source, val_losses_target


def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        rgb = config.rgb if "rgb" in config else False
        rgb_channels = (
            config.rgb_channels if "rgb_channels" in config else (425, 192, 109)
        )
        trainloader_source, validationloader_source, testloader_source = (
            build_FIVES_random_crops_dataloaders(
                batch_size=config.batch_size_source,
                num_channels=config.in_channels,
                load_from_path=config.dataset_path,
            )
        )
        window = config.window if "window" in config else None
        trainloader_target = build_hsi_dataloader(
            batch_size=config.batch_size_target,
            train_split=1,
            val_split=0,
            test_split=0,
            window=window,
            exclude_labeled_data=True,
            augmented=config.augmented,
            rgb=rgb,
            rgb_channels=rgb_channels,
        )[0]

        testloader_target = build_hsi_testloader(
            batch_size=1,
            window=window,
            rgb=rgb,
            rgb_channels=rgb_channels,
        )
        config["model"] = run.name
        model, _, _, _, _ = init_model_and_train(
            trainloader_source,
            validationloader_source,
            testloader_source,
            trainloader_target,
            testloader_target,
            config,
            device=config.device,
            batch_print=10,
            evaluate=True,
            with_overlays=True,
            save_wandb=False,
        )
        del model
        if os.path.exists(f"./models/{config.model}.pth"):
            os.remove(f"./models/{config.model}.pth")
            print(f"Removed model {config.model}.pth")
        torch.cuda.empty_cache()
