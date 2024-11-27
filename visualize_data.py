import matplotlib.pyplot as plt
import numpy as np
from dataset import build_hsi_testloader, get_wavelengths_from_metadata
import cv2
import ipywidgets as widgets
from IPython.display import display
import torch


def show_interactive_image_with_spectrum(image_id=4, rgb=True):
    # Precompute data outside of the onclick function
    testloader_HSI_windowed = build_hsi_testloader(window=(613, 615))
    testloader_HSI_all_channels = build_hsi_testloader()

    if rgb:
        input_image = (
            testloader_HSI_all_channels.dataset[image_id][0].cpu().numpy().squeeze()
        )
        input_image = np.stack(
            [input_image[425, :, :], input_image[192, :, :], input_image[109, :, :]],
            axis=-1,
        )  # Convert to 3-channel grayscale RGB
    else:
        input_image = (
            testloader_HSI_windowed.dataset[image_id][0].cpu().numpy().squeeze()
        )
        input_image = np.stack(
            [input_image, input_image, input_image], axis=-1
        )  # Convert to 3-channel grayscale RGB

    label = testloader_HSI_windowed.dataset[image_id][1].cpu().numpy().squeeze()

    image_min = input_image.min()
    image_max = input_image.max()

    # Apply min-max normalization to scale values to [0, 255]
    input_image = 255 * (input_image - image_min) / (image_max - image_min)
    input_image = input_image.astype(np.uint8)  # Convert to uint8 after scaling

    label_overlay = np.zeros_like(input_image)
    label_overlay[label == 1] = [0, 0, 255]
    combined = cv2.addWeighted(input_image, 0.7, label_overlay, 0.3, 0)

    # Precompute the spectra for all pixels
    # Assuming that the data is not too large to fit into memory
    pixel_spectra = testloader_HSI_all_channels.dataset[image_id][0].cpu().numpy()
    labels = testloader_HSI_all_channels.dataset[image_id][1].cpu().numpy()

    selected_pixels = []  # List to store selected pixel coordinates

    # Create the figure and axes
    fig, (ax_img, ax_spectrum) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image
    im_display = ax_img.imshow(combined)
    ax_img.set_title("Click on pixels to view their spectra")

    # Prepare the scatter plot for selected pixels
    scatter_plot = ax_img.scatter([], [], c=[], s=100, marker="+", edgecolors=[])

    # Set up the spectrum plot
    ax_spectrum.set_title("Pixel Spectrum")
    ax_spectrum.set_xlabel("Channel Index")
    ax_spectrum.set_ylabel("Intensity")

    # Prepare the line plots for the spectra
    lines = []

    colors = ["b", "g", "r", "c", "m", "y", "k"]  # Colors for different spectra

    def onclick(event):
        """Handle mouse click events."""
        if event.inaxes == ax_img:  # Ensure the click is within the image area
            x = int(event.xdata)
            y = int(event.ydata)
            pixel_coords = (y, x)

            # Toggle pixel selection
            if pixel_coords in selected_pixels:
                selected_pixels.remove(pixel_coords)
            else:
                selected_pixels.append(pixel_coords)

            update_plots()

    def update_plots():
        """Update the image and spectrum plots based on selected pixels."""
        # Update the scatter plot with all selected pixels
        offsets = np.array([[px[1], px[0]] for px in selected_pixels])
        scatter_plot.set_offsets(offsets)

        # Set the colors for the scatter plot markers to match the spectra colors
        marker_colors = [
            colors[idx % len(colors)] for idx in range(len(selected_pixels))
        ]
        scatter_plot.set_edgecolors(marker_colors)
        scatter_plot.set_facecolors("none")  # Keep the markers transparent inside

        # Update the spectrum plot
        # Clear existing lines
        for line in lines:
            line.remove()
        lines.clear()

        for idx, (py, px) in enumerate(selected_pixels):
            # Extract pixel spectrum
            pixel_values = pixel_spectra[:, py, px]
            label_value = labels[:, py, px].item()
            color = colors[idx % len(colors)]
            (line,) = ax_spectrum.plot(
                get_wavelengths_from_metadata(),
                pixel_values,
                marker="o",
                linestyle="-",
                color=color,
                label=f"Pixel ({py},{px}) Label={label_value}",
            )
            lines.append(line)

        ax_spectrum.set_title("Pixel Intensity Across Channels")
        ax_spectrum.set_xlabel("Wavelength (nm)")
        ax_spectrum.set_ylabel("Intensity")
        ax_spectrum.grid(axis="y", linestyle="--", alpha=0.7)
        if selected_pixels:
            ax_spectrum.legend()
        else:
            # Remove the legend if no pixels are selected
            legend = ax_spectrum.get_legend()
            if legend:
                legend.remove()

        # Redraw the updated plots
        fig.canvas.draw_idle()

    def on_clear_button_clicked(b):
        """Handle the clear button click event."""
        selected_pixels.clear()
        # Update the scatter plot to remove all markers
        scatter_plot.set_offsets(np.empty((0, 2)))
        scatter_plot.set_edgecolors([])
        # Clear existing lines in the spectrum plot
        for line in lines:
            line.remove()
        lines.clear()
        # Remove the legend if present
        legend = ax_spectrum.get_legend()
        if legend:
            legend.remove()
        # Redraw the updated plots
        fig.canvas.draw_idle()

    # Create the "Clear Selections" button
    clear_button = widgets.Button(description="Clear Selections")
    clear_button.on_click(on_clear_button_clicked)

    # Display the button
    display(clear_button)

    # Connect the click event to the callback function
    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()


def plot_intensity(dataloader, bins=50, title="Intensity Distribution of Dataset"):
    """
    Plots the intensity distribution of a PyTorch dataset.

    Args:
        dataset (Dataset): PyTorch dataset containing the data.
        batch_size (int): Batch size for DataLoader.
        bins (int): Number of bins for the histogram.
    """

    # Collect intensity values from the dataset
    intensities = []
    for batch in dataloader:
        image = batch[0]
        intensities.append(image.flatten())

    # Flatten all intensities into a single array
    intensities = torch.cat(intensities).numpy()

    # Plotting the intensity distribution
    plt.figure(figsize=(8, 6))
    plt.hist(intensities, bins=bins, alpha=0.7, color="blue")
    plt.title(title)
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()
    
def plot_intensity_for_label(dataloader, label=1, bins=50, title="Intensity Distribution for Label"):
    """
    Plots the intensity distribution of pixels with a specific label in a PyTorch dataset.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing the dataset.
        label (int): The label value to filter pixels (default is 1).
        bins (int): Number of bins for the histogram.
        title (str): Title for the plot.
    """
    # Collect intensity values for the specified label
    intensities = []
    for batch in dataloader:
        images = batch[0]  # Batch of images
        masks = batch[1]   # Corresponding segmentation masks

        # Filter intensities where the mask is equal to the specified label
        for image, mask in zip(images, masks):
            intensities.append(image[mask == label].flatten())

    # Flatten all intensities into a single array
    intensities = torch.cat(intensities).numpy()

    # Plotting the intensity distribution
    plt.figure(figsize=(8, 6))
    plt.hist(intensities, bins=bins, alpha=0.7, color="blue")
    plt.title(f"{title} (Label={label})")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()
    
    
def compute_snr(image, label):
    """
    Compute the Signal-to-Noise Ratio (SNR) for a given image and segmentation label.

    Args:
        image (torch.Tensor): The image tensor of shape (1, H, W).
        label (torch.Tensor): The segmentation label tensor of shape (1, H, W).

    Returns:
        float: The SNR value.
    """
    # Ensure image and label are in numpy format for easier indexing
    image = image.squeeze().numpy()
    label = label.squeeze().numpy()

    # Signal: Pixels where the label > 0
    signal_region = image[label > 0]

    # Background: Pixels where the label == 0
    background_region = image[label == 0]

    # Compute signal statistics
    mu_signal = signal_region.mean()

    # Compute noise as the standard deviation of the background
    sigma_noise = background_region.std()

    # Compute SNR
    if sigma_noise == 0:
        return float("inf")  # Handle edge case where noise is zero
    snr = mu_signal / sigma_noise
    return snr


def plot_snr_distribution(dataloader, bins=50, title="SNR Distribution of Dataset"):
    """
    Compute and plot the SNR distribution for a dataset.

    Args:
        dataset (Dataset): PyTorch dataset containing images and labels.
        batch_size (int): Batch size for DataLoader.
        bins (int): Number of bins for the histogram.
    """

    # Collect SNR values
    snr_values = []
    for batch in dataloader:
        images, labels = batch[0], batch[1]
        for i in range(images.size(0)):
            snr = compute_snr(images[i], labels[i])
            snr_values.append(snr)

    # Plotting the SNR distribution
    plt.figure(figsize=(8, 6))
    plt.hist(snr_values, bins=bins, alpha=0.7, color="green")
    plt.title(title)
    plt.xlabel("SNR")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()