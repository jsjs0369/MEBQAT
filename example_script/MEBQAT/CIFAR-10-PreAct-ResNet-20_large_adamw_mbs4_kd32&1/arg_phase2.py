import torch
import torch.optim as optim

PHASE2_SCHEME = 'NonFewShot'

PHASE1_SCHEME = 'MEBQAT-NonFewShot'     # None when LAST_EPOCH is 0
WORKERS = 8
DATASET = 'CIFAR10'
DATASET_PATH = '/root/jsyoun_data/CIFAR-10/'
MODEL_ARCH = 'PreActResNet20'
PRETRAINED = False
Q_METHOD = 'Yu21'
Q_BITS_W_FIRST = None
Q_BITS_W_INTER_LIST_LIST_GIVEN = [
        [1] * 20,
        [2] * 20,
        [3] * 20,
        [4] * 20,
        [5] * 20,
        [6] * 20,
        [7] * 20,
        [8] * 20,
        [16] * 20,
        [None] * 20]
Q_BITS_A_INTER_LIST_LIST_GIVEN = [
        [1] * 17,
        [2] * 17,
        [3] * 17,
        [4] * 17,
        [5] * 17,
        [6] * 17,
        [7] * 17,
        [8] * 17,
        [16] * 17,
        [None] * 17]
Q_BITS_W_LAST = None
Q_BITS_A_LAST = None
LAST_EPOCH = 280
BATCH_SIZE = 256
Q_SUBTASKS = 10
C_BATCHES = 12345   # Dummy large number → automatically changed to len(test_loader)
FIRST_SEED_UNGIVEN_Q_SUBTASKS = None
TRACK_RUNNING_STATS = False
