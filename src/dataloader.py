import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.utils import load_chipwhisperer, generate_traces, calculate_HW, load_aes_hd_ext, \
    load_ascad, load_ctf, load_rasp
import torch

class Custom_Dataset(Dataset):
    def __init__(self, root = './', dataset = "Chipwhisperer", leakage = "HW",transform = None, byte = 2):
        if dataset == 'simulated_traces_order_0' or dataset == 'simulated_traces_order_1' or dataset == 'simulated_traces_order_2' or dataset == 'simulated_traces_order_3':
            data_root = './Dataset/' + dataset + '/'
            if not os.path.exists(data_root):
                os.mkdir(data_root)
            if dataset == 'simulated_traces_order_0':
                order = 0
            elif dataset == 'simulated_traces_order_1':
                order = 1
            elif dataset == 'simulated_traces_order_2':
                order = 2
            elif dataset == 'simulated_traces_order_3':
                order = 3
            save_data = True
            self.correct_key = 0x03
            trace_size = 24 #24,64,88,128,216,256, 512
            if save_data == True:
                self.X_profiling, self.Y_profiling, self.plt_profiling = generate_traces(n_traces=14000, n_features=trace_size, order=order)
                self.X_attack, self.Y_attack, self.plt_attack = generate_traces(n_traces=30000, n_features=trace_size, order=order)
                np.save(data_root + "X_profiling.npy", self.X_profiling)
                np.save(data_root + "Y_profiling.npy", self.Y_profiling)
                np.save(data_root + "plt_profiling.npy", self.plt_profiling)
                np.save(data_root + "X_attack.npy", self.X_attack)
                np.save(data_root + "Y_attack.npy", self.Y_attack)
                np.save(data_root + "plt_attack.npy", self.plt_attack)
                if leakage == 'HW':
                    self.Y_profiling = np.array(calculate_HW(self.Y_profiling))
                    self.Y_attack = np.array(calculate_HW(self.Y_attack))
            else:
                self.X_profiling = np.load(data_root + "X_profiling.npy")
                self.Y_profiling = np.load(data_root + "Y_profiling.npy")
                self.plt_profiling = np.load(data_root + "plt_profiling.npy")
                self.X_attack = np.load(data_root + "X_attack.npy")
                self.Y_attack = np.load(data_root + "Y_attack.npy")
                self.plt_attack = np.load(data_root + "plt_attack.npy")

                if leakage == 'HW':
                    self.Y_profiling = np.array(calculate_HW(self.Y_profiling))
                    self.Y_attack = np.array(calculate_HW(self.Y_attack))
        elif dataset == 'Chipwhisperer':
            data_root = 'Dataset/Chipwhisperer/'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack), self.correct_key = load_chipwhisperer(root + data_root + '/', leakage_model=leakage)


        elif dataset == 'AES_HD_ext':
            data_root = 'Dataset/AES_HD_ext/aes_hd_ext.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), self.correct_key = load_aes_hd_ext(root + data_root, leakage_model=leakage,
                                                                      train_begin=0, train_end=45000,
                                                                      test_begin=0,
                                                                      test_end=10000)
        elif dataset == 'ASCAD':
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
                self.plt_profiling, self.plt_attack), self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=10000)
        elif dataset == "Raspberry_PI":
            byte = 2
            data_root = 'Dataset/Raspberry_PI/CHES_Challenge_v0.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
                self.plt_profiling, self.plt_attack), self.correct_key = load_rasp(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=10000)


        elif dataset == 'ASCAD_desync50':
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_desync50.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack),  self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=50000, test_begin=0,
                test_end=10000)
        elif dataset == 'ASCAD_desync100':
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_desync100.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack),  self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=50000, test_begin=0,
                test_end=10000)
        elif dataset == 'ASCAD_variable':
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_variable.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
                self.plt_profiling, self.plt_attack), self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=10000)
        elif dataset == 'ASCAD_variable_desync50':
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_variable_desync50.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack),  self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=20000)

        elif dataset == 'ASCAD_variable_desync100':
            byte = 2
            data_root = 'Dataset/ASCAD/ASCAD_variable_desync100.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack),  self.correct_key = load_ascad(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=20000)
        elif dataset == 'AES_HD_ext':
            data_root = 'Dataset/AES_HD_ext/aes_hd_ext.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (
            self.plt_profiling, self.plt_attack), correct_key = load_aes_hd_ext(
                root + data_root, leakage_model=leakage, train_begin=0, train_end=45000, test_begin=0,
                test_end=10000)
        elif dataset == 'CTF2018':
            byte = 0
            data_root = 'Dataset/CTF2018/ches_ctf.h5'
            (self.X_profiling, self.X_attack), (self.Y_profiling, self.Y_attack), (self.plt_profiling, self.plt_attack), self.correct_key = load_ctf(
                root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
                test_end=10000)
        print("The dataset we using: ", data_root)
        self.transform = transform
        self.scaler_std = StandardScaler()
        self.scaler = MinMaxScaler()
        self.X_profiling = self.scaler_std.fit_transform(self.X_profiling)
        self.X_attack = self.scaler_std.transform(self.X_attack)

        # self.scaler_std_latent = StandardScaler()
        # self.scaler_latent = MinMaxScaler()
        self.X_attack_test, self.X_attack_val, self.Y_attack_test, self.Y_attack_val = train_test_split(self.X_attack,self.Y_attack,test_size=0.1,random_state=0)

        # print("X_profiling:", self.X_profiling)
        print("X_profiling max:", np.max(self.X_profiling))
        print("X_profiling min:", np.min(self.X_profiling))
        # print("plt_profiling:", self.plt_profiling)
    def apply_MinMaxScaler(self):
        self.X_profiling = self.scaler.fit_transform(self.X_profiling)
        self.X_attack = self.scaler.transform(self.X_attack)
        print("After minmaxscaler X_profiling max:", np.max(self.X_profiling))
        print("After minmaxscaler X_profiling min:", np.min(self.X_profiling))


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