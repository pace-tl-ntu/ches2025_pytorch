import random

import numpy as np
import torch

if __name__=="__main__":
    dataset = "CHES_2025"

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
