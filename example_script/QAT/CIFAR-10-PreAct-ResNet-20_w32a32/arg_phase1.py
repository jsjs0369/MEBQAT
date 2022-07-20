import torch
import torch.optim as optim

PHASE1_SCHEME = 'QAT-NonFewShot'

WORKERS = 8
DATASET = 'CIFAR10'
DATASET_PATH = '/root/jsyoun_data/CIFAR-10/'
MODEL_ARCH = 'PreActResNet20'
PRETRAINED = False
Q_METHOD = 'Yu21'   # QAT method used in [AAAI'21] Any-precision DNN
Q_BITS_W_FIRST = None   # None: no quantization
Q_BITS_W_LAST = None
Q_BITS_A_LAST = None
LAST_EPOCH = 0
LAST_BEST_AVG_ACC_VAL = 0.0
EPOCHS = 400
BATCH_SIZE = 256
OPTIM = optim.Adam
OPTIM_KWARGS = {'lr': 1e-3}
LR_SCH = optim.lr_scheduler.MultiStepLR
LR_SCH_KWARGS = {'milestones': [150, 250, 350], 'gamma': 0.1}
Q_BITS_W_INTER_LIST = [None] * 20
Q_BITS_A_INTER_LIST = [None] * 17
SAVE_PERIOD = 1
REPORT_PERIOD = 100
