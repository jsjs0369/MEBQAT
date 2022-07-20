import torch
import torch.optim as optim

PHASE2_SCHEME = 'NonFewShot'

PHASE1_SCHEME = 'QAT-NonFewShot'     # None when LAST_EPOCH is 0
WORKERS = 0
DATASET = 'SVHN'
DATASET_PATH = '/root/jsyoun_data/SVHN/'
MODEL_ARCH = 'CNNForSVHN'
PRETRAINED = False
Q_METHOD = 'Yu21'
Q_BITS_W_FIRST = None
Q_BITS_W_INTER_LIST_LIST_GIVEN = [
        [1] * 6,
        [2] * 6,
        [3] * 6,
        [4] * 6,
        [5] * 6,
        [6] * 6,
        [7] * 6,
        [8] * 6,
        [16] * 6,
        [None] * 6]
Q_BITS_A_INTER_LIST_LIST_GIVEN = [
        [1] * 6,
        [2] * 6,
        [3] * 6,
        [4] * 6,
        [5] * 6,
        [6] * 6,
        [7] * 6,
        [8] * 6,
        [16] * 6,
        [None] * 6]
Q_BITS_W_LAST = None
Q_BITS_A_LAST = None
LAST_EPOCH = 67
BATCH_SIZE = 256
Q_SUBTASKS = 10
C_BATCHES = 12345   # Dummy large number → automatically changed to len(test_loader)
FIRST_SEED_UNGIVEN_Q_SUBTASKS = None
TRACK_RUNNING_STATS = True