import torch
import torch.optim as optim

PHASE1_SCHEME = 'MEBQAT-NonFewShot'

WORKERS = 4
DATASET = 'SVHN'
DATASET_PATH = <Path to the dataset>
MODEL_ARCH = 'CNNForSVHN'
PRETRAINED = False
Q_METHOD = 'Yu21'   # QAT method used in [AAAI'21] Any-precision DNN
Q_BITS_W_FIRST = None   # None: no quantization
Q_BITS_W_LAST = None
Q_BITS_A_LAST = None
LAST_EPOCH = 0
LAST_BEST_AVG_ACC_VAL = 0.0
EPOCHS = 100
BATCH_SIZE = 256
OUTER_OPTIM = optim.Adam
OUTER_OPTIM_KWARGS = {'lr': 0.001}
OUTER_LR_SCH = optim.lr_scheduler.MultiStepLR
OUTER_LR_SCH_KWARGS = {'milestones': [50, 75, 90], 'gamma': 0.1}
INNER_Q_SUBTASKS = 4
Q_BITS_WA_GUARANTEED_LIST_GIVEN = [(None, None), (1, 1)]          # [(q_bits_w_guaranteed1, q_bits_a_guaranteed1), ...]
DISTILL_KNOWLEDGE = True
SAVE_PERIOD = 1
REPORT_PERIOD = 1000
