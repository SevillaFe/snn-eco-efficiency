import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import TensorBoardLogger

import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random

from codecarbon import EmissionsTracker


# =============================================================================
# Model
# =============================================================================

class PilotNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super(PilotNet, self).__init__()
        self.learning_rate = learning_rate

        # PilotNet CNN architecture (identical to Udacity/Sully Chen scripts)
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
            nn.Linear(10, 1),  # Single steering angle output
        )

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


# =============================================================================
# Dataset
# =============================================================================

def load_airsim_dataframe(data_raw_dir):
    """
    Iterate over all session sub-folders inside data_raw_dir (e.g. normal_1,
    normal_2, ...) and consolidate their airsim_rec.txt files into a single
    DataFrame with columns ['img_path', 'steering'].

    Expected sub-folder layout (AirSim E2E cookbook dataset):
        Original_Images/
            normal_1/
                images/          <- PNG frames
                airsim_rec.txt   <- tab-separated sensor log
            normal_2/
                ...

    The airsim_rec.txt header is:
        Timestamp  Speed (kmph)  Throttle  Steering  Brake  Gear  ImageName
    """
    records = []

    # Collect every airsim_rec.txt found anywhere inside data_raw_dir
    pattern = os.path.join(data_raw_dir, "**", "airsim_rec.txt")
    rec_files = sorted(glob.glob(pattern, recursive=True))

    if not rec_files:
        raise FileNotFoundError(
            f"No airsim_rec.txt files found under '{data_raw_dir}'. "
            "Check that data_raw_dir points to the correct root."
        )

    for rec_path in rec_files:
        session_dir = os.path.dirname(rec_path)         # e.g. .../normal_1
        images_dir  = os.path.join(session_dir, "images")

        try:
            df = pd.read_csv(rec_path, sep="\t")
        except Exception as e:
            print(f"[WARNING] Could not read {rec_path}: {e}. Skipping.")
            continue

        # Normalise column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        # Validate expected columns
        required = {"Steering", "ImageName"}
        if not required.issubset(df.columns):
            print(
                f"[WARNING] {rec_path} is missing columns {required - set(df.columns)}. "
                "Skipping."
            )
            continue

        # Drop rows with NaN steering or missing image
        df = df.dropna(subset=["Steering", "ImageName"])

        for _, row in df.iterrows():
            img_name = str(row["ImageName"]).strip()
            img_path = os.path.join(images_dir, img_name)

            # Skip entries whose image does not exist on disk
            if not os.path.isfile(img_path):
                continue

            records.append(
                {"img_path": img_path, "steering": float(row["Steering"])}
            )

    if not records:
        raise RuntimeError(
            "DataFrame is empty after parsing all airsim_rec.txt files. "
            "Check that the images/ sub-folders exist and are populated."
        )

    consolidated = pd.DataFrame(records)
    print(
        f"[INFO] Loaded {len(consolidated):,} samples from "
        f"{len(rec_files)} session(s) in '{data_raw_dir}'."
    )
    return consolidated


class AirSimSteeringDataset(Dataset):
    """
    PyTorch Dataset for the AirSim E2E deep-learning dataset.

    Steering values in airsim_rec.txt are already in [-1, 1] (normalised),
    so no unit conversion is applied (unlike the Udacity script which converts
    degrees to radians).  If your recording uses a different convention,
    adjust the steering pre-processing below accordingly.
    """

    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Must contain 'img_path' and 'steering'.
            transform  (callable):    torchvision transform pipeline.
        """
        self.data      = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row            = self.data.iloc[idx]
        img_path       = row["img_path"]
        steering_angle = float(row["steering"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            # Track whether a RandomHorizontalFlip actually fires so we can
            # mirror the steering label consistently (same logic as Udacity).
            flipped = False
            for t in self.transform.transforms:
                if isinstance(t, transforms.RandomHorizontalFlip):
                    flipped = random.random() < t.p
            image = self.transform(image)
            if flipped:
                steering_angle = -steering_angle

        label = torch.tensor(steering_angle, dtype=torch.float32)
        return image, label


# =============================================================================
# DataModule
# =============================================================================

class AirSimDataModule(pl.LightningDataModule):
    def __init__(self, data_raw_dir, batch_size=64, split_ratio=0.8):
        """
        Args:
            data_raw_dir (str): Path to the 'data_raw' folder that contains
                                the session sub-folders (normal_1, normal_2, …).
            batch_size   (int): Mini-batch size.
            split_ratio (float): Fraction of data used for training.
        """
        super().__init__()
        self.data_raw_dir = data_raw_dir
        self.batch_size   = batch_size
        self.split_ratio  = split_ratio

        # Transformations — kept identical to the Udacity/Sully Chen pipeline
        # for a fair comparison across datasets.
        self.train_transform = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            # transforms.RandomHorizontalFlip(p=0.5),  # Uncomment if desired
            transforms.ToTensor(),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        full_df = load_airsim_dataframe(self.data_raw_dir)

        full_dataset = AirSimSteeringDataset(
            dataframe=full_df,
            transform=(
                self.train_transform if stage == "fit" else self.val_transform
            ),
        )

        train_size = int(self.split_ratio * len(full_dataset))
        val_size   = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    # Root folder that contains the session sub-folders (normal_1, normal_2 …)
    # Adjust this to wherever you extracted the AirSim E2E dataset.
    data_raw_dir = (
        "./AirSim_Dataset/Original_Images"
    )

    output_dir_cnn = (
        "./AirSim_Dataset/No_encoded_cnn/Emissions/Emissions_model"
    )
    os.makedirs(output_dir_cnn, exist_ok=True)

    # ------------------------------------------------------------------
    # DataModule & Model
    # ------------------------------------------------------------------
    data_module = AirSimDataModule(
        data_raw_dir=data_raw_dir,
        batch_size=64,
        split_ratio=0.8,
    )
    model = PilotNet(learning_rate=1e-4)

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger_name = "CNN_PilotNet_AirSim"
    logger = TensorBoardLogger(
        "./Scripts_Encoding/Scripts/Scripts_Paper_2/tb_logs_paper2", name=logger_name
        )


    # ------------------------------------------------------------------
    # Emissions tracking (CodeCarbon) — training phase
    # ------------------------------------------------------------------
    tracker_training = EmissionsTracker(
        project_name="encoding_" + logger_name,
        output_dir=output_dir_cnn,
        output_file="training_emissions_" + logger_name + ".csv",
    )
    tracker_training.start()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="mps",      # Apple Silicon GPU; change to 'gpu' or 'cpu' as needed
        devices="auto",
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=50,
    )
    trainer.fit(model, data_module)

    # ------------------------------------------------------------------
    # Stop emissions tracker
    # ------------------------------------------------------------------
    emissions: float = tracker_training.stop()
    print(f"[INFO] Total training emissions: {emissions:.6f} kg CO2eq")
