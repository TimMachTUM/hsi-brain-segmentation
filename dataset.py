import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import spectral
import numpy as np
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None):
        """
        Args:
            image_dir (string): Directory with all the original images.
            label_dir (string): Directory with all the labels.
            image_transform (callable, optional): Optional transform to be applied on a sample.
            label_transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.images[idx])  # Assuming label and image files are named the same
        
        
        image = Image.open(img_name).convert("RGB")
        label = Image.open(label_name).convert("L")  # Convert label image to grayscale if needed

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
                
                if os.path.exists(raw_path) and os.path.exists(gt_path) and os.path.exists(img_path):
                    self.data_paths.append((raw_path, gt_path, img_path))

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
        raw_path, gt_path, img_path = self.data_paths[idx]
        
        # Load the hyperspectral image and label
        hsi_image = spectral.open_image(raw_path).load()
        label = spectral.open_image(gt_path).load()
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        hsi_image = hsi_image.transpose(2,0,1)
        label = (label.transpose(2,0,1) == 3).astype(int)

        # Convert to PyTorch tensors
        hsi_image = torch.tensor(hsi_image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int8)

        if self.image_transform:
            hsi_image = self.image_transform(hsi_image)

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
            transforms.Normalize(mean, std)
        ])

