import os, h5py, pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from snntorch.spikegen import rate
import numpy as np
from codecarbon import EmissionsTracker


class DrivingDataset(Dataset):
    def __init__(self, csv_file, img_dir, resize=(66,200)):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # -> [0,1]
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.basename(str(row.iloc[0]))
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)             # [1,66,200]
        y = float(row.iloc[1]) * np.pi / 180.0
        return x, y


@torch.no_grad()
def encode_rate_to_h5(csv_file, img_dir, h5_path,
                      num_steps=25, gain=1.0,
                      batch_size=256, num_workers=4, device="cpu",
                      split_ratio=0.8,
                      project_name="SNN_Encoding", emissions_dir="./emissions"):

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    os.makedirs(emissions_dir, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=emissions_dir,
        output_file=f"emissions_rate_{num_steps}_{gain}.csv"
    )
    tracker.start()

    ds = DrivingDataset(csv_file, img_dir)
    n = len(ds)
    n_train = int(split_ratio * n)
    n_val   = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Shapes
    c, h, w = ds[0][0].shape

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
    data_dir = ".../PaloAlto_Dataset/Original_Images"
    csv_file = os.path.join(data_dir, "driving_log.csv")
    img_dir  = os.path.join(data_dir, "IMG")

    emissions_dir = ".../PaloAlto_Dataset/Encoded_Images/Rate/Emissions"
    save_dir = ".../PaloAlto_Dataset/Encoded_Images/Rate/Encoded_Datasets"
    os.makedirs(save_dir, exist_ok=True)

    num_steps_range = [25]      # you can add [5, 15, 25]
    gain_range = [0.5]      # [0.5, 1.0]  test multiple values if needed

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
                emissions_dir=emissions_dir
            )

            print(f"Finished num_steps={num_steps}, gain={gain} → Emissions={emissions:.6f} kg CO₂")
            
