import os
#from prometheus_client import h
import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import h5py
from codecarbon import EmissionsTracker
from torchmetrics import MeanAbsolutePercentageError
#import Roger_gradient_substitut_TFM as new_surrogate


class SNNLaskNet(pl.LightningModule):
    def __init__(self, learning_rate=0.01, beta=0.9, 
                spike_grad=surrogate.fast_sigmoid(slope=25), threshold=0.5,learn_beta=True): # original learning_rate=1e-4, beta=0.9 threshold = 0.5
                #spike_grad=new_surrogate.new_surrogate(slope=25), threshold=0.5,learn_beta=True): # original learning_rate=1e-4, beta=0.9 threshold = 0.5

        super(SNNLaskNet, self).__init__()
        self.learning_rate = learning_rate


        # Define the SNN with spiking layers
        # Define the SNN with spiking layers
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(48, 64, kernel_size=3),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(3840, 120),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(120, 20),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(20, 1),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=spike_grad,
              init_hidden=True, output=True, reset_mechanism="none")
)
        
        #1152
        self.loss_fn = nn.MSELoss()
    


    def forward(self, x):
        utils.reset(self.net)
        mem_rec = []  # Record membrane potentials for analysis if needed
        if x.dim() < 5: # Add a dummy time dimension to the input if it's missing (case delta encoding)
            x = x.unsqueeze(1)
        for step in range(x.size(1)):
            step_data = x[:, step, :, :, :]
            spk_out, mem_out = self.net(step_data)
            #print("mem_out dtype:", mem_out.dtype)
            mem_rec.append(mem_out)
        return mem_out  # Return final time step output

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        #print("Predictions dtype:", predictions.dtype)
        #print("Labels dtype:", labels.dtype)
        labels = labels.unsqueeze(1)
        loss = self.loss_fn(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        labels = labels.unsqueeze(1)
        val_loss = self.loss_fn(predictions, labels)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        #return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.5492389157972004) # original 
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class H5SpikingDataset(Dataset):
    def __init__(self, h5_file_path, split="train"):
        assert split in ("train", "val")
        self.h5_file_path = h5_file_path
        self.split = split

        self.h5_file = h5py.File(self.h5_file_path, 'r')
        if self.split == "train":
            self.images = self.h5_file['train_images']
            self.labels = self.h5_file['train_labels']
        else:
            self.images = self.h5_file['val_images']
            self.labels = self.h5_file['val_labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load as numpy, cast to float32
        image = torch.from_numpy(self.images[idx]).float()  # cast uint8 → float32
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

    def close(self):
        self.h5_file.close()
        
# DataModule with loading logic based on the custom dataset
class H5PilotNetDataModule(pl.LightningDataModule):
    def __init__(self, h5_file_path, batch_size=64):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = H5SpikingDataset(self.h5_file_path, split="train")
        self.val_dataset = H5SpikingDataset(self.h5_file_path, split="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage=None):
        # Close the H5 file after training/validation
        self.train_dataset.close()
        self.val_dataset.close()
    
    

if __name__ == "__main__":
    
    # Define the path to the encoded dataset
    h5_dir_rate = '.../PaloAlto_Dataset/Encoded_Images/Rate/Encoded_Datasets'
    h5_file_rate = 'encoded_dataset_rate_numsteps_15_gain_0.5.h5'
    h5_file_path_rate = os.path.join(h5_dir_rate, h5_file_rate)

    #h5_dir_latency = '.../PaloAlto_Dataset/Encoded_Images/Latency/Encoded_Datasets'
    #h5_file_latency = 'encoded_dataset_latency_numsteps_50_tau_5.0.h5'
    #h5_file_path_latency = os.path.join(h5_dir_latency, h5_file_latency)
#
    #h5_dir_delta = '.../PaloAlto_Dataset/Encoded_Images/Delta/Encoded_Datasets'
    #h5_file_delta = 'encoded_dataset_delta_threshold_1.0.h5'
    #h5_file_path_delta = os.path.join(h5_dir_delta, h5_file_delta)
    
    # Define the path to store EmissionsTracker output for 20 epochs
    output_dir_rate = '.../PaloAlto_Dataset/Encoded_Images/Rate/Emissions/Emissions_model/LaskNet'
    #output_dir_latency = '.../PaloAlto_Dataset/Encoded_Images/Latency/Emissions/Emissions_model/LaskNet'
    #output_dir_delta = '.../PaloAlto_Dataset/Encoded_Images/Delta/Emissions/Emissions_model/LaskNet'


    # Create the data module and model
    data_module = H5PilotNetDataModule(h5_file_path=h5_file_path_rate, batch_size=64)
    model = SNNLaskNet(learning_rate=1e-4, beta=0.9) #model = SNNPilotNet(learning_rate=1e-4, beta=0.9) originales
    
    # Create a logger for TensorBoard
    logger_name = f"SNN_LaskNet_PaloAlto_{h5_file_rate.split('_', 3)[-1].replace('.h5', '')}"
    logger = TensorBoardLogger("Scripts/Scripts_Paper_2/tb_logs_paper2", name=logger_name)

    # Record whole carbon footprint 
    tracker_training = EmissionsTracker(
        project_name="encoding_" + logger_name,
        output_dir=output_dir_rate,
        output_file='training_emissions_' + logger_name + '.csv'
    )
    tracker_training.start()

    # Train the model
    trainer = pl.Trainer(max_epochs=20, accelerator= 'mps', devices='auto', precision='16-mixed', logger=logger, log_every_n_steps=50)
    trainer.fit(model, data_module)
    
    # Stop record emissions
    emissions: float = tracker_training.stop()
