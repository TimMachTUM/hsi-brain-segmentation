import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import spectral
import numpy as np
import cv2
import random

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None, augmentation=None):
        """
        Args:
            image_dir (string): Directory with all the original images.
            label_dir (string): Directory with all the labels.
            image_transform (callable, optional): Optional transform to be applied on a sample.
            label_transform (callable, optional): Optional transform to be applied on a sample.
            augmentation (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.augmentation = augmentation
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.images[idx])  # Assuming label and image files are named the same
        
        
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("L")  # Convert label image to grayscale if needed

        if self.augmentation:
            image, label = self.augmentation(image, label)
            
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.label_transform:
            label = self.label_transform(label)

        return image, label


class HSIDataset(Dataset):
    def __init__(self, root_dir, image_transform=None):
        """
        Initialize the dataset with the path to the data.
        Args:
        root_dir (str): Path to the directory containing subdirectories with the HSI data and labels.
        """
        self.root_dir = root_dir
        self.data_paths = []  # To store paths of the hyperspectral images and labels
        self.image_transform = image_transform

        # Iterate through each subdirectory in the root directory
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                raw_path = os.path.join(subdir_path, 'raw.hdr')
                gt_path = os.path.join(subdir_path, 'gtMap.hdr')
                img_path = os.path.join(subdir_path, 'image.jpg')
                dark_path = os.path.join(subdir_path, 'darkReference.hdr')
                white_path = os.path.join(subdir_path, 'whiteReference.hdr')
                
                if os.path.exists(raw_path) and os.path.exists(gt_path) and os.path.exists(img_path):
                    self.data_paths.append((raw_path, gt_path, img_path, dark_path, white_path))

        self.data_paths.sort()

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Retrieve the HSI image and corresponding label at the index `idx`.
        Args:
        idx (int): The index of the item.
        
        Returns:
        tuple: (image, label) where `image` is a hyperspectral image and `label` is the corresponding ground truth map.
        """
        raw_path, gt_path, img_path, dark_path, white_path = self.data_paths[idx]
        
        # Load the hyperspectral image and label
        hsi_image = spectral.open_image(raw_path).load()
        dark_reference = spectral.open_image(dark_path).load()
        white_reference = spectral.open_image(white_path).load()

        dark_full = np.tile(dark_reference, (hsi_image.shape[0], 1, 1))
        white_full = np.tile(white_reference, (hsi_image.shape[0], 1, 1))

        label = spectral.open_image(gt_path).load()
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img[:,:,[2,1,0]]

        hsi_image = (hsi_image - dark_full) / (white_full - dark_full)
        hsi_image[hsi_image <= 0] = 10**-2
        hsi_image = hsi_image.transpose(2,0,1)
        label = (label.transpose(2,0,1) == 3).astype(int)

        # Convert to PyTorch tensors
        hsi_image = torch.tensor(hsi_image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int8)

        if self.image_transform and self.label_transform:
            hsi_image = self.image_transform(hsi_image)
            label = self.label_transform(label)
            img = self.center_crop(img, (224, 224))

        return hsi_image, label, img
    
    def get_mean_std(self):
        """
        Compute the mean and standard deviation of the dataset.
        """
        if os.path.exists('./normalization_coefficients/mean.npy') and os.path.exists('./normalization_coefficients/std.npy'):
            mean = np.load('./normalization_coefficients/mean.npy')
            std = np.load('./normalization_coefficients/std.npy')
            return mean, std
        
        # Initialize variables to store the sum and sum of squares
        sum_ = 0
        sum_sq = 0
        num_pixels = 0

        # Iterate through each image in the dataset
        for i in range(len(self)):
            image, _ = self[i]
            sum_ += image.sum()
            sum_sq += (image ** 2).sum()
            num_pixels += image.numel()

        # Compute the mean and standard deviation
        mean = sum_ / num_pixels
        std = ((sum_sq / num_pixels) - (mean ** 2)) ** 0.5
        np.save('./normalization_coefficients/mean.npy', mean)
        np.save('./normalization_coefficients/std.npy', std)

        return mean, std
    
    def normalize_dataset(self):
        """
        Normalize the dataset using the given mean and standard deviation.
        Args:
        """

        mean, std = self.get_mean_std()
        self.image_transform = transforms.Compose([
            transforms.CenterCrop(224),
        ])
        self.label_transform = transforms.Compose([
            transforms.CenterCrop(224)
        ])

    def center_crop(self, image, crop_size):
        """
        Perform a center crop on the image.

        Args:
        image (numpy array): The input image.
        crop_size (tuple): The size of the crop (height, width).

        Returns:
        numpy array: The cropped image.
        """
        height, width, _ = image.shape
        new_height, new_width = crop_size

        # Calculate the coordinates for the center crop
        start_x = width // 2 - (new_width // 2)
        start_y = height // 2 - (new_height // 2)

        # Perform the crop
        cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]

        return cropped_image

class SegmentationDatasetWithRandomCrops(Dataset):
    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None, crop_height=512, crop_width=512, threshold=0.1):
        """
        Args:
            image_dir (string): Directory with all the original images.
            label_dir (string): Directory with all the labels.
            image_transform (callable, optional): Optional transform to be applied on a sample.
            label_transform (callable, optional): Optional transform to be applied on a sample.
            crop_height (int): Desired height of the cropped image.
            crop_width (int): Desired width of the cropped image.
            threshold (float): Minimum required vessel ratio in the cropped image.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.threshold = threshold
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.images[idx])
        
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("L")  # Convert label image to grayscale

        image = np.array(image)
        label = np.array(label)
        label = (label > 0).astype(np.uint8)  # Binarize the label image
        
        # Apply the random cropping with condition
        cropped_image, cropped_label = self.random_crop_with_condition(image, label, self.crop_height, self.crop_width, self.threshold)

        if cropped_image is None or cropped_label is None:
            # Handling case where no valid crop is found, you could choose to not augment or use the full image as a fallback
            cropped_image = image
            cropped_label = label

        # Convert numpy arrays back to PIL images
        cropped_image = Image.fromarray(cropped_image)
        cropped_label[cropped_label == 1] = 255

        if self.image_transform:
            cropped_image = self.image_transform(cropped_image)
        
        if self.label_transform:
            cropped_label = self.label_transform(cropped_label)


        return cropped_image, cropped_label
    
    def calculate_vessel_ratio(self, mask):
        return np.sum(mask == 1) / np.prod(mask.shape)

    def random_crop_with_condition(self, image, mask, crop_height, crop_width, threshold=0.1):
        assert image.shape[:2] == mask.shape[:2], "Image and mask must have the same dimensions."

        height, width = image.shape[:2]
        for _ in range(100):  # Try up to 100 times to find a valid crop
            x = random.randint(0, width - crop_width)
            y = random.randint(0, height - crop_height)
            cropped_image = image[y:y+crop_height, x:x+crop_width]
            cropped_mask = mask[y:y+crop_height, x:x+crop_width]

            if self.calculate_vessel_ratio(cropped_mask) >= threshold:
                return cropped_image, cropped_mask

        # If no valid crop is found, return a center crop
        center_x = (width - crop_width) // 2
        center_y = (height - crop_height) // 2
        center_cropped_image = image[center_y:center_y+crop_height, center_x:center_x+crop_width]
        center_cropped_mask = mask[center_y:center_y+crop_height, center_x:center_x+crop_width]
        
        return center_cropped_image, center_cropped_mask