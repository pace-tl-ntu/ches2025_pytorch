import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils import load_ctf_2025
import torch

class Custom_Dataset(Dataset):
    def __init__(self, root = './', dataset = "CHES_2025", leakage = "HW",transform = None):

        if dataset == "CHES_2025":
            byte = 2
            data_root = 'Dataset/CHES_2025/CHES_Challenge_v0.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
                self.plt_profiling, self.plt_attack), self.correct_key = load_ctf_2025(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=10000)

        print("The dataset we using: ", data_root)
        self.transform = transform
        self.scaler_std = StandardScaler()
        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)

        self.X_attack_test, self.X_attack_val, self.Y_attack_test, self.Y_attack_val = train_test_split(self.X_attack,self.Y_attack,test_size=0.1,random_state=0)



    def choose_phase(self,phase):
        if phase == 'train':
            self.X, self.Y = np.expand_dims(self.X_profiling, 1), self.Y_profiling
        elif phase == 'validation':
            self.X, self.Y = np.expand_dims(self.X_attack_val, 1), self.Y_attack_val
        elif phase == 'test':
            self.X, self.Y =np.expand_dims(self.X_attack_test, 1), self.Y_attack_test



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]
        # plaintext = self.Plaintext[idx]
        sample = {'trace': trace, 'sensitive': sensitive} #, 'plaintext': plaintext}
        # print(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor_trace(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # trace, label, plaintext= sample['trace'], sample['sensitive'], sample['plaintext']
        trace, label= sample['trace'], sample['sensitive']#, sample['plaintext']

        return torch.from_numpy(trace).float(), torch.from_numpy(np.array(label)).long()