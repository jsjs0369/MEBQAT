import torch
import torch.optim as optim

PHASE1_SCHEME = 'mebqat-maml'

WORKERS = 4
DATASET = 'omniglot'
DATASET_PATH = <Path to the dataset>
NET_ARCH = 'maml-conv-net'
QUANT_SCHEME = 'lsq'
QUANT_SCHEME_KWARGS = {}
QB_W_FIRST = 32     # 32: no quantization (equal to None in non_few_shot)
QB_A_FIRST = 32
QB_W_LAST = 32
QB_A_LAST = 32
LAST_EPOCH = 0
LAST_BEST_AVG_VAL_ACC_IFR = 0.0
EPOCHS = 75
INTER_UNIFORM = True
N_WAY = 20
K_SHOT_SUP = 1
K_SHOT_QRY = 15
OUTER_OPTIM = optim.Adam
OUTER_OPTIM_KWARGS = {'lr': 1e-4}
INNER_CL_SUBTASKS = 16
INNER_OPTIM = optim.SGD
INNER_OPTIM_KWARGS = {'lr': 0.1}
INNER_UPDATES = 5
VAL_UPDATES = 5
SAVE_PERIOD = 1
