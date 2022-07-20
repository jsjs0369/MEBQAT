import torch
import torch.optim as optim

PHASE1_SCHEME = 'QAT-NonFewShot'

WORKERS = 4
DATASET = 'CIFAR10'
DATASET_PATH = '../jsyoun_data/CIFAR-10/'
MODEL_ARCH = 'MobileNetV2ForCIFAR'
PRETRAINED = False
Q_METHOD = 'SAT-modifiedW'
Q_BITS_W_FIRST = 8
Q_BITS_W_LAST = 8
Q_BITS_A_LAST = 'same'
LAST_EPOCH = 0
LAST_BEST_AVG_ACC_VAL = 0.0
EPOCHS = 600
BATCH_SIZE = 256
OPTIM = optim.Adam
OPTIM_KWARGS = {'lr': 5e-2}
LR_SCH = optim.lr_scheduler.CosineAnnealingLR
LR_SCH_KWARGS = {'T_max': 600}
Q_BITS_W_INTER_LIST = [2] * 51
Q_BITS_A_INTER_LIST = [2] * 34
SAVE_PERIOD = 1
REPORT_PERIOD = 100
