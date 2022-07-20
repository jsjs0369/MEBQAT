import torch
import torch.optim as optim

PHASE2_SCHEME = 'NonFewShot'

PHASE1_SCHEME = 'QAT-NonFewShot'     # None when LAST_EPOCH is 0
WORKERS = 4
DATASET = 'CIFAR10'
DATASET_PATH = '../jsyoun_data/CIFAR-10/'
MODEL_ARCH = 'MobileNetV2ForCIFAR'
PRETRAINED = False
Q_METHOD = 'SAT-modifiedW'
Q_BITS_W_FIRST = 8
Q_BITS_W_INTER_LIST_LIST_GIVEN = [
        [2] * 51,
        [3] * 51,
        [4] * 51,
        [5] * 51,
        [6] * 51,
        [7] * 51,
        [8] * 51,
        [16] * 51,
        [None] * 51]
Q_BITS_A_INTER_LIST_LIST_GIVEN = [
        [2] * 34,
        [3] * 34,
        [4] * 34,
        [5] * 34,
        [6] * 34,
        [7] * 34,
        [8] * 34,
        [16] * 34,
        [None] * 34]
Q_BITS_W_LAST = 8
Q_BITS_A_LAST = 'same'
LAST_EPOCH = 600
BATCH_SIZE = 256
Q_SUBTASKS = 9
C_BATCHES = 12345   # Dummy large number → automatically changed to len(test_loader)
FIRST_SEED_UNGIVEN_Q_SUBTASKS = None
TRACK_RUNNING_STATS = True
