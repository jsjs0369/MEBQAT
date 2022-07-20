import torch
import torch.optim as optim

PHASE2AND3_CONFIG = 'qa-ft-and-ifr'

PHASE1_SCHEME = 'qat-fomaml'
WORKERS = 4
DATASET = 'omniglot'
DATASET_PATH = '../jsyoun_data/Omniglot/'
NET_ARCH = 'maml-conv-net'
QUANT_SCHEME = 'lsq'
QUANT_SCHEME_KWARGS = {}
QB_W_FIRST = 32     # 32: no quantization (equal to None in non_few_shot)
QB_A_FIRST = 32
QB_W_LAST = 32
QB_A_LAST = 32
LAST_EPOCH = 65
CL_SUBTASKS = 600
SEED_FOR_CL_SUBTASKS = None
QB_SUBTASKS = 9
INTER_QB_TUPLE_LIST_LIST_GIVEN = [
    [(2, 2)] * 3,
    [(3, 3)] * 3,
    [(4, 4)] * 3,
    [(5, 5)] * 3,
    [(6, 6)] * 3,
    [(7, 7)] * 3,
    [(8, 8)] * 3,
    [(16, 16)] * 3,
    [(32, 32)] * 3]
SEED_FOR_REMAINING_QB_SUBTASKS = None
N_WAY = 20
K_SHOT_FT = 1
K_SHOT_IFR = 15
FT_OPTIM = optim.SGD
FT_OPTIM_KWARGS = {'lr': 0.1}
FT_UPDATES = 5
