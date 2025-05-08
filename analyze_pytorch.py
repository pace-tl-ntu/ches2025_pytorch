import os
import random
from copy import deepcopy
import numpy as np
import torch

from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN
from src.trainer import trainer
from src.utils import evaluate, AES_Sbox, calculate_HW

if __name__=="__main__":
    dataset = "CHES_2025"
    leakage = "HW"
    nb_traces_attacks = 1700
    total_nb_traces_attacks = 2000

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


    ##################please do not touch this code here###################
    dataloadertest = Custom_Dataset(root='./../', dataset=dataset, leakage="ID", #change root to where you download your dataset.
                                                 transform=transforms.Compose([ToTensor_trace()]))
    #########################################################################
    if leakage == 'ID':
        def leakage_fn(att_plt, k):
            return AES_Sbox[k ^ int(att_plt)]
        classes = 256
    elif leakage == 'HW':
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]
            return hw[AES_Sbox[k ^ int(att_plt)]]
        classes = 9
        dataloadertest.Y_attack = calculate_HW(dataloadertest.Y_attack)
    else:
        ####TODO: You can change the code here if you want to create your own leakage model.
        pass


    ##################please do not touch this code here###################
    dataloadertest.choose_phase("test")
    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    Y_attack = dataloadertest.Y_attack
    plt_attack = dataloadertest.plt_attack
    num_sample_pts = X_attack.shape[-1]
    #########################################################################


    ##TODO: Load your model (note, you have to create your model in this file and new function should be in this file.) ########################
    ############## Below is an example ############################################
    model_type = "mlp"
    root = "./Result/"
    save_root = root + dataset + "_" + model_type + "_" + leakage + "/"
    model_root = save_root + "models/"
    config = np.load(model_root + "model_configuration_0.npy", allow_pickle=True).item()
    model = MLP(config, num_sample_pts, classes).to(device)
    model.load_state_dict(torch.load(model_root + "model_0.pth"))
    ###############################################################################


    ####All model will be evaluated based on this function, if it does not adhere to the following, it will be eliminated. ##################
    GE, NTGE = evaluate(device, model, X_attack, plt_attack, correct_key, leakage_fn=leakage_fn, nb_attacks=100,
                        total_nb_traces_attacks=2000, nb_traces_attacks=1700)

