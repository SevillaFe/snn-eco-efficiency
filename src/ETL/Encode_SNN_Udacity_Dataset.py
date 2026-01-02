import os, h5py, pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from snntorch.spikegen import rate
import numpy as np
import random
from codecarbon import EmissionsTracker


class DrivingDataset(Dataset):
    def __init__(self, csv_file, img_dir, resize=(66,200), transform=None, steering_correction=0.2):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            img_dir (str): Directory with all the images.
            resize (tuple): Target image size (height, width).
            transform (callable, optional): Optional transform to be applied on an image.
            steering_correction (float): Angle adjustment for left and right images.
        """
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.img_dir = img_dir
        self.steering_correction = steering_correction
        self.transform = transform if transform is not None else T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # -> [0,1]
        ])

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        data_entry = self.df.iloc[idx]
        
        # Randomly select a camera view and adjust the steering angle accordingly
        camera_choice = random.choice(["left", "center", "right"])
        
        if camera_choice == "left":
            # Left camera image (column 1 in Udacity dataset)
            img_path = os.path.join(self.img_dir, os.path.basename(data_entry.iloc[1]))
            steering_angle = float(data_entry.iloc[3]) + self.steering_correction  # Adjust for left camera
        elif camera_choice == "right":
            # Right camera image (column 2 in Udacity dataset)
            img_path = os.path.join(self.img_dir, os.path.basename(data_entry.iloc[2]))
            steering_angle = float(data_entry.iloc[3]) - self.steering_correction  # Adjust for right camera
        else:  # Center
            # Center camera image (column 0 in Udacity dataset)
            img_path = os.path.join(self.img_dir, os.path.basename(data_entry.iloc[0]))
            steering_angle = float(data_entry.iloc[3])  # No adjustment for center

        # Load and transform the image
        image = Image.open(img_path).convert('RGB')
        x = self.transform(image)  # [1,66,200]
        
        return x, steering_angle


class AugmentedDrivingDataset(Dataset):
    """Dataset with augmentation for training - includes multi-camera and flip handling"""
    def __init__(self, csv_file, img_dir, resize=(66,200), steering_correction=0.2):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            img_dir (str): Directory with all the images.
            resize (tuple): Target image size (height, width).
            steering_correction (float): Angle adjustment for left and right images.
        """
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.img_dir = img_dir
        self.steering_correction = steering_correction
        
        # Define augmentation transforms for training
        self.augment_transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            # Typical Udacity dataset augmentations
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomRotation(degrees=5),  # Small rotation for road curves
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Small translation
                scale=(0.9, 1.1),      # Small scaling
                shear=5                # Small shear transformation
            ),
            T.ToTensor()  # -> [0,1]
        ])
        
        # Separate horizontal flip for manual steering angle adjustment
        self.flip_transform = T.RandomHorizontalFlip(p=0.5)

    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        data_entry = self.df.iloc[idx]
        
        # Randomly select a camera view and adjust the steering angle accordingly
        camera_choice = random.choice(["left", "center", "right"])
        
        if camera_choice == "left":
            # Left camera image (column 1 in Udacity dataset)
            img_path = os.path.join(self.img_dir, os.path.basename(data_entry.iloc[1]))
            steering_angle = float(data_entry.iloc[3]) + self.steering_correction  # Adjust for left camera
        elif camera_choice == "right":
            # Right camera image (column 2 in Udacity dataset)
            img_path = os.path.join(self.img_dir, os.path.basename(data_entry.iloc[2]))
            steering_angle = float(data_entry.iloc[3]) - self.steering_correction  # Adjust for right camera
        else:  # Center
            # Center camera image (column 0 in Udacity dataset)
            img_path = os.path.join(self.img_dir, os.path.basename(data_entry.iloc[0]))
            steering_angle = float(data_entry.iloc[3])  # No adjustment for center

        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation transforms (except flip)
        x = self.augment_transform(image)
        
        # Handle horizontal flip manually to adjust steering angle
        if random.random() < 0.5:  # 50% chance of flip
            x = T.functional.hflip(x)
            steering_angle = -steering_angle  # Flip the steering angle
        
        return x, steering_angle


@torch.no_grad()
def encode_rate_to_h5(csv_file, img_dir, h5_path,
                      num_steps=25, gain=1.0,
                      batch_size=256, num_workers=4, device="cpu",
                      split_ratio=0.8,
                      project_name="SNN_Encoding", emissions_dir="./emissions",
                      use_augmentation=True, steering_correction=0.2):

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    os.makedirs(emissions_dir, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=emissions_dir,
        output_file=f"emissions_rate_{num_steps}_{gain}.csv"
    )
    tracker.start()

    # Create base dataset for splitting with multi-camera support
    base_ds = DrivingDataset(csv_file, img_dir, steering_correction=steering_correction)
    n = len(base_ds)
    n_train = int(split_ratio * n)
    n_val   = n - n_train
    
    # Split indices
    train_indices, val_indices = random_split(range(n), [n_train, n_val],
                                            generator=torch.Generator().manual_seed(42))
    
    # Create datasets with different transforms
    if use_augmentation:
        print("Using augmented training dataset with multi-camera support...")
        # Create augmented dataset for training
        train_ds = AugmentedDrivingDataset(csv_file, img_dir, steering_correction=steering_correction)
        # Subset with training indices
        train_ds = torch.utils.data.Subset(train_ds, train_indices.indices)
    else:
        train_ds = torch.utils.data.Subset(base_ds, train_indices.indices)
    
    # Validation dataset without augmentation but with multi-camera support
    val_ds = torch.utils.data.Subset(base_ds, val_indices.indices)

    # Shapes
    c, h, w = base_ds[0][0].shape

    # Preallocate HDF5 file
    with h5py.File(h5_path, "w") as f:
        train_x = f.create_dataset(
            "train_images", shape=(n_train, num_steps, c, h, w),
            dtype="uint8", compression=None,
            chunks=(batch_size, num_steps, c, h, w)
        )
        train_y = f.create_dataset(
            "train_labels", shape=(n_train,), dtype="float32",
            compression=None, chunks=(max(1024, batch_size),)
        )

        val_x = f.create_dataset(
            "val_images", shape=(n_val, num_steps, c, h, w),
            dtype="uint8", compression=None,
            chunks=(batch_size, num_steps, c, h, w)
        )
        val_y = f.create_dataset(
            "val_labels", shape=(n_val,), dtype="float32",
            compression=None, chunks=(max(1024, batch_size),)
        )

        # Select device
        if device == "mps" and torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        # Helper to encode + write a split
        def encode_split(subset, x_dset, y_dset, name="train"):
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, persistent_workers=(num_workers>0))
            ptr = 0
            for xb, yb in loader:
                xb = xb.to(dev, non_blocking=False)
                spikes = rate(xb, num_steps=num_steps, gain=gain)  # [T,B,1,H,W]
                spikes = spikes.permute(1,0,2,3,4).contiguous().to(torch.uint8)
                B = xb.size(0)
                x_dset[ptr:ptr+B] = spikes.cpu().numpy()
                y_dset[ptr:ptr+B] = yb.numpy().astype("float32")
                ptr += B
            print(f"{name} set done: {ptr} samples")

        # Encode both splits
        encode_split(train_ds, train_x, train_y, name="train")
        encode_split(val_ds, val_x, val_y, name="val")

    emissions: float = tracker.stop()
    print(f"Encoded dataset saved to {h5_path}")
    print(f"CO₂ emissions for encoding: {emissions:.6f} kg")
    return emissions


if __name__ == "__main__":
    data_dir = ".../Doctorado/Udacity_Dataset/Original_Images"
    csv_file = os.path.join(data_dir, "driving_log.csv")
    img_dir  = os.path.join(data_dir, "IMG")

    emissions_dir = ".../Udacity_Dataset/paper_2/Encoded_Images/Rate/Emissions"
    save_dir = ".../Udacity_Dataset/paper_2/Encoded_Images/Rate/Encoded_Datasets"
    os.makedirs(save_dir, exist_ok=True)

    num_steps_range = [5, 15, 25]      # you can add [5, 15, 25]
    gain_range = [0.5, 1.0]     # [0.5, 1.0]  test multiple values if needed

    for num_steps in num_steps_range:
        for gain in gain_range:
            out_path = os.path.join(
                save_dir,
                f"encoded_dataset_rate_numsteps_{num_steps}_gain_{gain}.h5"
            )

            emissions = encode_rate_to_h5(
                csv_file, img_dir, out_path,
                num_steps=num_steps, gain=gain,
                batch_size=320, num_workers=4, device="cpu",
                split_ratio=0.8,
                project_name="RateEncoding",
                emissions_dir=emissions_dir,
                use_augmentation=True,  # Enable augmentation
                steering_correction=0.2  # Multi-camera correction
            )

            print(f"Finished num_steps={num_steps}, gain={gain} → Emissions={emissions:.6f} kg CO₂")
