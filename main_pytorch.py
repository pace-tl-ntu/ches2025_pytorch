# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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
    model_type = "mlp" #mlp, cnn
    leakage = "HW" #ID, HW
    train_models = True
    num_epochs = 50
    total_num_models = 2
    nb_traces_attacks = 1700
    total_nb_traces_attacks = 2000


    if not os.path.exists('./Result/'):
        os.mkdir('./Result/')

    root = "./Result/"
    save_root = root+dataset+"_"+model_type+"_"+leakage+"/"
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
    if leakage == 'ID':
        def leakage_fn(att_plt, k):
            return AES_Sbox[k ^ int(att_plt)]
        classes = 256

    elif leakage == 'HW':
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]
            return hw[AES_Sbox[k ^ int(att_plt)]]
        classes = 9
    ####You can change the code above if you want to create your own leakage model.



    dataloadertrain = Custom_Dataset(root='./../', dataset=dataset, leakage="ID",
                                                 transform=transforms.Compose([ToTensor_trace()]))
    ##########################################################################

    if leakage == "HW":

        dataloadertrain.Y_profiling = calculate_HW(dataloadertrain.Y_profiling)
        dataloadertrain.Y_attack = calculate_HW(dataloadertrain.Y_attack)


    dataloadertrain.choose_phase("train")
    dataloadertest = deepcopy(dataloadertrain)
    dataloadertest.choose_phase("test")
    dataloaderval = deepcopy(dataloadertrain)
    dataloaderval.choose_phase("validation")

    correct_key = dataloadertrain.correct_key
    X_attack = dataloadertrain.X_attack
    Y_attack = dataloadertrain.Y_attack
    plt_attack = dataloadertrain.plt_attack
    num_sample_pts = X_attack.shape[-1]
    #Random Search
    for num_models in range(total_num_models):
        if train_models == True:
            config = create_hyperparameter_space(model_type)
            np.save(model_root + "model_configuration_"+str(num_models)+".npy", config)
            batch_size = config["batch_size"]
            num_workers = 2
            dataloaders = {"train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size,
                                                                shuffle=True,
                                                                num_workers=num_workers),
                           "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size,
                                                              shuffle=True, num_workers=num_workers)
                           }
            dataset_sizes = {"train": len(dataloadertrain), "val": len(dataloaderval)}



            model = trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device)
            torch.save(model.state_dict(), model_root + "model_"+str(num_models)+".pth")
        else:
            config = np.load(model_root + "model_configuration_"+str(num_models)+".npy", allow_pickle=True).item()
            if model_type == "mlp":
                model = MLP(config, num_sample_pts, classes).to(device)
            elif model_type == "cnn":
                model = CNN(config, num_sample_pts, classes).to(device)
            model.load_state_dict(torch.load(model_root + "model_"+str(num_models)+".pth"))
        #Evaluate
        GE, NTGE = evaluate(device, model, X_attack, plt_attack, correct_key,leakage_fn=leakage_fn, nb_attacks=100, total_nb_traces_attacks=2000, nb_traces_attacks=1700)
        np.save(model_root + "/result_"+str(num_models), {"GE": GE, "NTGE": NTGE})