import torch
import torch.optim as optim

PHASE1_SCHEME = 'mebqat-pn'

WORKERS = 4
DATASET = 'omniglot'
DATASET_PATH = <Path to the dataset>
NET_ARCH = 'proto-conv-net'
QUANT_SCHEME = 'lsq'
QUANT_SCHEME_KWARGS = {}
QB_W_FIRST = 32     # 32: no quantization
QB_A_FIRST = 32
LAST_EPOCH = 0
LAST_BEST_AVG_VAL_ACC_QRY = 0.0
EPOCHS = 600
INTER_UNIFORM = True
N_WAY_VAL = 20
K_SHOT_SUP = 1
K_SHOT_QRY_VAL = 15
OUTER_OPTIM = optim.Adam
OUTER_OPTIM_KWARGS = {'lr': 1e-3}
OUTER_LR_SCH = optim.lr_scheduler.StepLR
OUTER_LR_SCH_KWARGS = {'step_size': 40, 'gamma': 0.5}
INNER_QB_SUBTASKS = 4
SAVE_PERIOD = 1
TRACK_RUNNING_STATS = False
