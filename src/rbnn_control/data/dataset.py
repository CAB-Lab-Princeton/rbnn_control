# Name: Justice Mason
# Project: RBNN + Control
# Date: 02/10/2024

import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Measurement Dataset Class

class LowDimDataset(Dataset):
    """
    Measurement dataset for Rigid Body Neural Network (RBNN) with control.
    
    ...
    
    Attributes
    ----------
    data_dir : str
        Path to data directory (full trajectories generated using Euler's equations + control vector)
        
    seq_len : int, default=2
        The length of each sample trajectory in the dataset
        
    Methods
    -------
    __len__()
    __getitem__(idx)
    
    Notes
    -----
    
    """
    def __init__(self, data_dir, seq_len: int = 2):
        self.data_dir = data_dir
        filename = glob.glob(os.path.join(self.data_dir, "*.npz"))[0]
        self.seq_len = seq_len

        # Load files
        npzfiles = np.load(filename)
        self.data_R = npzfiles['R']
        self.data_omega = npzfiles['omega']
        self.data_u = npzfiles['u']
        
    def __len__(self):
        """
        Computes total number of trajectory samples of length seq_len in the dataset.
        
        ...
        
        """
        num_traj, traj_len, _, _  = self.data_R.shape
        num_samples = num_traj * (traj_len - self.seq_len + 1)
        return num_samples
    
    def __getitem__(self, idx):
        """
        Returns specific trajectory subsequence corresponding to idx.
        
        ...
        
        Parameters
        ----------
        idx : int
            Sample index.
        
        Returns
        -------
        sample : torch.Tensor
            Returns sample of length seq_len with 
        
        Notes
        -----
        
        """
        assert idx < self.__len__(), "Index is out of range."
        _, traj_len, _, _  = self.data_R.shape
        
        traj_idx, seq_idx = divmod(idx, traj_len - self.seq_len + 1)

        # Sample data
        sample_R = self.data_R[traj_idx, seq_idx:seq_idx+self.seq_len, ...]
        sample_omega = self.data_omega[traj_idx, seq_idx:seq_idx+self.seq_len, ...]
        sample_u = self.data_u[traj_idx, seq_idx:seq_idx+self.seq_len, ...]
        
        sample = (sample_R, sample_omega, sample_u)
        return sample
    
# High-dimensional data + control dataset
class RBNNControlDataset(Dataset):
    """
    Dataset class for the RBNN + Control project.
    
    ...
    
    Attributes
    ----------
    data : torch.Tensor
        N-D array of images for training/testing.
        
    seq_len : int, default=3
        Number of observations representating a sequence of images -- input to the network.
        
    Methods
    -------
    __len__()
    __getitem__()
    
    Notes
    -----
    
    """
    # This will be changed.
    def __init__(self, data_x: np.ndarray, data_u: np.ndarray, seq_len: int = 3):
        super().__init__()

        self.data_x = data_x
        self.data_u = data_u
        self.seq_len = seq_len
        
    def __len__(self):
        """
        """
        num_traj, traj_len, _, _, _ = self.data_x.shape
        length = num_traj * (traj_len - self.seq_len + 1)   
        return length
        
    def __getitem__(self, idx):
        """
        """
        assert idx < self.__len__(),  "Index is out of range."
        traj_len = self.data_x.shape[1]
        
        traj_idx, seq_idx = divmod(idx, traj_len - self.seq_len + 1)
        
        sample_x = self.data_x[traj_idx, seq_idx:seq_idx+self.seq_len,...]
        sample_u = self.data_u[traj_idx, seq_idx:seq_idx+self.seq_len,...]
        return sample_x, sample_u