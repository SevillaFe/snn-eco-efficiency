import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import TensorBoardLogger

import os
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random

from codecarbon import EmissionsTracker

class PilotNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super(PilotNet, self).__init__()
        self.learning_rate = learning_rate

        # Define the PilotNet CNN architecture
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # Output is a single steering angle value
        )

        # Define the loss function (MSE for regression)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images).squeeze(-1)
        loss = self.loss(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images).squeeze(-1)
        val_loss = self.loss(predictions, labels)
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        test_loss = self.loss(predictions, labels)
        self.log("test_loss", test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class SteeringAngleDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, steering_correction=0.2):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            steering_correction (float): Angle adjustment for left and right images.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.steering_correction = steering_correction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_entry = self.data.iloc[idx]
        
        img_path = os.path.join(self.img_dir, data_entry.iloc[0].split('/')[-1])
        steering_angle = data_entry.iloc[1]* np.pi / 180  # To convert degrees to radians

        # Load and transform the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            # Apply transform and check for random flip afterward
            flipped = False
            for t in self.transform.transforms:
                if isinstance(t, transforms.RandomHorizontalFlip):
                    flipped = random.random() < t.p  # Determine if the flip occurs
            image = self.transform(image)
            
            # Invert the label if the image was flipped
            if flipped:
                steering_angle = -steering_angle  # Flip the steering angle

        # Convert steering angle to tensor
        label = torch.tensor(steering_angle, dtype=torch.float32)

        return image, label

# DataModule with loading logic based on the custom dataset
class PilotNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, csv_file, batch_size, split_ratio=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        # Transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((66, 200)),
            #transforms.ColorJitter(brightness=0.3, contrast=0.3),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ]) # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ToTensor()
        ]) #  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def setup(self, stage=None):
        # Load the dataset from CSV and split it
        full_dataset = SteeringAngleDataset(
            csv_file=self.csv_file,
            img_dir=os.path.join(self.data_dir, "IMG"),
            transform=self.train_transform if stage == 'fit' else self.val_transform,
            steering_correction=0.2
        )

        # Calculate split indices
        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    
    

if __name__ == "__main__":
    data_dir = '/Users/fernando/Documents/Doctorado/PaloAlto_Dataset/Original_Images/'  # Root directory containing "IMG" folder and CSV file
    csv_file = os.path.join(data_dir, "driving_log.csv")  # Path to the CSV file
    
     # Define the path to store EmissionsTracker output
    output_dir_cnn = '/Users/fernando/Documents/Doctorado/PaloAlto_Dataset/No_encoded_cnn/Emissions/Emissions_model'
    

    data_module = PilotNetDataModule(data_dir=data_dir, csv_file=csv_file, batch_size=128)
    model = PilotNet(learning_rate=1e-4)

    logger_name = 'CNN_PilotNet_PaloAlto'
    logger = TensorBoardLogger("Scripts/Scripts_Paper_2/tb_logs_paper2", name=logger_name)
    
    # Record whole carbon footprint 
    tracker_training = EmissionsTracker(
        project_name="encoding_" + logger_name,
        output_dir=output_dir_cnn,
        output_file='training_emissions_' + logger_name + '.csv'
    )
    tracker_training.start()

    trainer = pl.Trainer(max_epochs=20, accelerator= 'mps', devices='auto', precision='16-mixed', logger=logger, log_every_n_steps=50)
    trainer.fit(model, data_module)
    
    # Stop record emissions
    emissions: float = tracker_training.stop()
