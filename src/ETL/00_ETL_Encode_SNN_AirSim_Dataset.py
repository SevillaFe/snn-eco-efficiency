import os, glob, h5py
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from snntorch.spikegen import rate
import numpy as np
from codecarbon import EmissionsTracker


# =============================================================================
# Data loading — multi-session AirSim structure
# =============================================================================

def load_airsim_dataframe(data_raw_dir):
    """
    Iterate over all session sub-folders inside data_raw_dir (e.g. normal_1,
    normal_2, ...) and consolidate their airsim_rec.txt files into a single
    DataFrame with columns ['img_path', 'steering'].

    Expected layout:
        data_raw/
            normal_1/
                images/          <- PNG frames
                airsim_rec.txt   <- tab-separated: Timestamp Speed Throttle
                                                   Steering Brake Gear ImageName
            normal_2/
                ...
    """
    records = []

    pattern  = os.path.join(data_raw_dir, "**", "airsim_rec.txt")
    rec_files = sorted(glob.glob(pattern, recursive=True))

    if not rec_files:
        raise FileNotFoundError(
            f"No airsim_rec.txt files found under '{data_raw_dir}'. "
            "Check that data_raw_dir points to the correct root."
        )

    for rec_path in rec_files:
        session_dir = os.path.dirname(rec_path)
        images_dir  = os.path.join(session_dir, "images")

        try:
            df = pd.read_csv(rec_path, sep="\t")
        except Exception as e:
            print(f"[WARNING] Could not read {rec_path}: {e}. Skipping.")
            continue

        df.columns = [c.strip() for c in df.columns]

        required = {"Steering", "ImageName"}
        if not required.issubset(df.columns):
            print(
                f"[WARNING] {rec_path} is missing columns "
                f"{required - set(df.columns)}. Skipping."
            )
            continue

        df = df.dropna(subset=["Steering", "ImageName"])

        for _, row in df.iterrows():
            img_name = str(row["ImageName"]).strip()
            img_path = os.path.join(images_dir, img_name)
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


# =============================================================================
# Dataset
# =============================================================================

class AirSimDrivingDataset(Dataset):
    """
    Dataset for the AirSim E2E dataset.

    Unlike PaloAlto/Udacity, steering values are already in [-1, 1]
    (AirSim normalised convention), so no unit conversion is applied.
    Preprocessing is otherwise identical to the other datasets:
    grayscale → resize 66×200 → ToTensor → [0, 1].
    """

    def __init__(self, dataframe, resize=(66, 200)):
        self.data = dataframe.reset_index(drop=True)
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),   # → [0, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        img_path = row["img_path"]
        steering = float(row["steering"])   # already in [-1, 1], no conversion

        img = Image.open(img_path).convert("RGB")
        x   = self.transform(img)           # [1, 66, 200]
        return x, steering


# =============================================================================
# Encoding pipeline
# =============================================================================

@torch.no_grad()
def encode_rate_to_h5(data_raw_dir, h5_path,
                      num_steps=25, gain=1.0,
                      batch_size=256, num_workers=4, device="cpu",
                      split_ratio=0.8,
                      project_name="SNN_Encoding", emissions_dir="./emissions"):
    """
    Load the consolidated AirSim dataset, apply rate encoding with snnTorch,
    and write the result to an HDF5 file.  CodeCarbon tracks emissions for the
    entire encoding stage (loading + encoding + I/O).

    Args:
        data_raw_dir  (str):   Path to the folder containing session sub-folders.
        h5_path       (str):   Destination HDF5 file path.
        num_steps     (int):   Temporal depth S ∈ {5, 15, 25}.
        gain          (float): Rate-encoding gain G ∈ {0.5, 1.0}.
        batch_size    (int):   Batch size for the encoding DataLoader.
        num_workers   (int):   DataLoader worker processes.
        device        (str):   'mps', 'cuda', or 'cpu'.
        split_ratio   (float): Train/val split (same seed=42 as other datasets).
        project_name  (str):   CodeCarbon project label.
        emissions_dir (str):   Directory for CodeCarbon CSV output.

    Returns:
        float: Total CO₂ emissions in kg for the encoding stage.
    """

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    os.makedirs(emissions_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Start emissions tracker                                              #
    # ------------------------------------------------------------------ #
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=emissions_dir,
        output_file=f"emissions_rate_{num_steps}_{gain}.csv"
    )
    tracker.start()

    # ------------------------------------------------------------------ #
    # Build consolidated dataset and deterministic train/val split        #
    # ------------------------------------------------------------------ #
    full_df = load_airsim_dataframe(data_raw_dir)
    ds      = AirSimDrivingDataset(full_df)

    n       = len(ds)
    n_train = int(split_ratio * n)
    n_val   = n - n_train
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Infer tensor shape from a single sample
    c, h, w = ds[0][0].shape   # (1, 66, 200)

    # ------------------------------------------------------------------ #
    # Select compute device                                                #
    # ------------------------------------------------------------------ #
    if device == "mps" and torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Preallocate HDF5 datasets (uint8, no compression — same strategy    #
    # as PaloAlto/Udacity scripts for reproducibility)                    #
    # ------------------------------------------------------------------ #
    with h5py.File(h5_path, "w") as f:
        train_x = f.create_dataset(
            "train_images",
            shape=(n_train, num_steps, c, h, w),
            dtype="uint8", compression=None,
            chunks=(min(batch_size, n_train), num_steps, c, h, w)
        )
        train_y = f.create_dataset(
            "train_labels", shape=(n_train,), dtype="float32",
            compression=None, chunks=(max(1024, batch_size),)
        )
        val_x = f.create_dataset(
            "val_images",
            shape=(n_val, num_steps, c, h, w),
            dtype="uint8", compression=None,
            chunks=(min(batch_size, n_val), num_steps, c, h, w)
        )
        val_y = f.create_dataset(
            "val_labels", shape=(n_val,), dtype="float32",
            compression=None, chunks=(max(1024, batch_size),)
        )

        # -------------------------------------------------------------- #
        # Inner helper: encode one split and stream it to HDF5            #
        # -------------------------------------------------------------- #
        def encode_split(subset, x_dset, y_dset, name="train"):
            loader = DataLoader(
                subset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0)
            )
            ptr = 0
            for xb, yb in loader:
                xb     = xb.to(dev, non_blocking=False)
                # rate() output shape: [T, B, C, H, W]
                spikes = rate(xb, num_steps=num_steps, gain=gain)
                # Reorder to [B, T, C, H, W] and cast to uint8
                spikes = spikes.permute(1, 0, 2, 3, 4).contiguous().to(torch.uint8)
                B = xb.size(0)
                x_dset[ptr:ptr + B] = spikes.cpu().numpy()
                y_dset[ptr:ptr + B] = np.array(yb, dtype="float32")
                ptr += B
            print(f"  [{name}] {ptr:,} samples encoded and written.")

        # -------------------------------------------------------------- #
        # Encode train and validation splits                               #
        # -------------------------------------------------------------- #
        print(f"\n[INFO] Encoding  S={num_steps}, G={gain}  →  {h5_path}")
        encode_split(train_ds, train_x, train_y, name="train")
        encode_split(val_ds,   val_x,   val_y,   name="val")

    # ------------------------------------------------------------------ #
    # Stop tracker and report                                              #
    # ------------------------------------------------------------------ #
    emissions: float = tracker.stop()
    print(f"[INFO] Encoded dataset saved  →  {h5_path}")
    print(f"[INFO] Encoding CO₂ emissions: {emissions:.6f} kg\n")
    return emissions


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    # Root of the AirSim E2E dataset (contains the session sub-folders)
    data_raw_dir = (
        "/Users/fernando/Documents/Doctorado/"
        "AirSim_Dataset/Original_Images"
    )

    emissions_dir = (
        "/Users/fernando/Documents/Doctorado/"
        "AirSim_Dataset/Encoded_Images/Rate/Emissions"
    )
    save_dir = (
        "/Users/fernando/Documents/Doctorado/"
        "AirSim_Dataset/Encoded_Images/Rate/Encoded_Datasets"
    )
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Hyperparameter grid — identical to Udacity / Sully Chen experiments
    # ------------------------------------------------------------------
    num_steps_range = [15]
    gain_range      = [0.5, 1.0]

    for num_steps in num_steps_range:
        for gain in gain_range:
            out_path = os.path.join(
                save_dir,
                f"encoded_dataset_rate_numsteps_{num_steps}_gain_{gain}.h5"
            )

            emissions = encode_rate_to_h5(
                data_raw_dir  = data_raw_dir,
                h5_path       = out_path,
                num_steps     = num_steps,
                gain          = gain,
                batch_size    = 320,
                num_workers   = 4,
                device        = "cpu",        # change to 'mps' for Apple Silicon GPU
                split_ratio   = 0.8,
                project_name  = "RateEncoding_AirSim",
                emissions_dir = emissions_dir,
            )

            print(
                f"[DONE] S={num_steps}, G={gain} → "
                f"Emissions={emissions:.6f} kg CO₂\n"
                f"{'─'*60}"
            )