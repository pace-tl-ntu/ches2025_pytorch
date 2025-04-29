# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import random
import numpy as np
import torch
from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset

dataset = "Raspberry_PI"
model_type = "mlp" #mlp, cnn
leakage = "HW" #ID, HW
byte = 0
num_epochs = 50
total_num_models = 100
nb_traces_attacks = 1700
total_nb_traces_attacks = 2000

if not os.path.exists('./Result/'):
    os.mkdir('./Result/')

root = "./Result/"
save_root = root+dataset+"_"+model_type+ "_byte"+str(byte)+"_"+leakage+"/"
model_root = save_root+"models/"
print("root:", root)
print("save_time_path:", save_root)
if not os.path.exists(root):
    os.mkdir(root)
if not os.path.exists(save_root):
    os.mkdir(save_root)
if not os.path.exists(model_root):
    os.mkdir(model_root)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nb_attacks = 100
if leakage == 'HW':
    classes = 9
elif leakage == 'ID':
    classes = 256


dataloadertrain = Custom_Dataset(root='./../', dataset=dataset, leakage=leakage,
                                             transform=transforms.Compose([ToTensor_trace()]), byte = byte)