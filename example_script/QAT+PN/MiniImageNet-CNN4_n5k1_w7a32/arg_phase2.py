import torch
import torch.optim as optim

PHASE2AND3_CONFIG = 'qa-prototype-and-infer'

PHASE1_SCHEME = 'qat-pn'
WORKERS = 8
DATASET = 'miniimagenet'
DATASET_PATH = '/root/jsyoun_data/MiniImageNet'
NET_ARCH = 'proto-conv-net'
QUANT_SCHEME = 'lsq'
QUANT_SCHEME_KWARGS = {}
QB_W_FIRST = 32
QB_A_FIRST = 32
LAST_EPOCH = 188
CL_SUBTASKS = 600
SEED_FOR_CL_SUBTASKS = None
QB_SUBTASKS = 1
INTER_QB_TUPLE_LIST_LIST_GIVEN = [
    [(7, 32)] * 3]
SEED_FOR_REMAINING_QB_SUBTASKS = None
N_WAY = 5
K_SHOT_SUP = 1
K_SHOT_QRY = 15
TRACK_RUNNING_STATS = True
