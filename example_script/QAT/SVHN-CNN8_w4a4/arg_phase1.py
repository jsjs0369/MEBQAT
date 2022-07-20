import torch
import torch.optim as optim

PHASE1_SCHEME = 'QAT-NonFewShot'

WORKERS = 0
DATASET = 'SVHN'
DATASET_PATH = '/root/jsyoun_data/SVHN/'
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
OPTIM = optim.Adam
OPTIM_KWARGS = {'lr': 0.001}
LR_SCH = optim.lr_scheduler.MultiStepLR
LR_SCH_KWARGS = {'milestones': [50, 75, 90], 'gamma': 0.1}
Q_BITS_W_INTER_LIST = [4] * 6
Q_BITS_A_INTER_LIST = [4] * 6
SAVE_PERIOD = 1
REPORT_PERIOD = 1000
