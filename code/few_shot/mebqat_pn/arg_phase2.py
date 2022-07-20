import torch
import torch.optim as optim

PHASE2AND3_CONFIG = 'qa-prototype-and-infer'

PHASE1_SCHEME = 'mebqat-pn'
WORKERS = 4
DATASET = 'omniglot'
DATASET_PATH = <Path to the dataset>
NET_ARCH = 'proto-conv-net'
QUANT_SCHEME = 'lsq'
QUANT_SCHEME_KWARGS = {}
QB_W_FIRST = 32     # 32: no quantization
QB_A_FIRST = 32
LAST_EPOCH = <Last epoch>
CL_SUBTASKS = 600
SEED_FOR_CL_SUBTASKS = None
QB_SUBTASKS = 10
INTER_QB_TUPLE_LIST_LIST_GIVEN = [
    [(2, 2)] * 3,
    [(3, 3)] * 3,
    [(4, 4)] * 3,
    [(5, 5)] * 3,
    [(6, 6)] * 3,
    [(7, 7)] * 3,
    [(8, 8)] * 3,
    [(16, 16)] * 3,
    [(32, 32)] * 3,
    [(2, 8)] * 3]
SEED_FOR_REMAINING_QB_SUBTASKS = None
N_WAY = 20
K_SHOT_SUP = 1
K_SHOT_QRY = 15
TRACK_RUNNING_STATS = False
