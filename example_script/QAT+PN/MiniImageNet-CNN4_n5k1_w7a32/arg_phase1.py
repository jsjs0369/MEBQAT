import torch
import torch.optim as optim

PHASE1_SCHEME = 'qat-pn'

WORKERS = 8
DATASET = 'miniimagenet'
DATASET_PATH = '/root/jsyoun_data/MiniImageNet'
NET_ARCH = 'proto-conv-net'
QUANT_SCHEME = 'lsq'
QUANT_SCHEME_KWARGS = {}
QB_W_FIRST = 32     # 32: no quantization
QB_A_FIRST = 32
LAST_EPOCH = 0
LAST_BEST_AVG_VAL_ACC_QRY = 0.0
EPOCHS = 200
INTER_QB_TUPLE_LIST_GIVEN = [(7, 32)] * 3
N_WAY_VAL = 5
K_SHOT_SUP = 1
K_SHOT_QRY_VAL = 15
OPTIM = optim.Adam
OPTIM_KWARGS = {'lr': 1e-3}
LR_SCH = optim.lr_scheduler.StepLR
LR_SCH_KWARGS = {'step_size': 40, 'gamma': 0.5}
SAVE_PERIOD = 1
