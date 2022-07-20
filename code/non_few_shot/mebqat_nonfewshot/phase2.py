# From own code
from arch import ResNet18, ResNet50, MobileNetV2ForImageNet, MobileNetV2ForCIFAR, PreActResNet50, PreActResNet20, CNNForSVHN, CNNForMAML, CNNForProtoNet
from arg_phase2 import *
from util import create_log_func, rgetattr, rdelattr, rsetparam
from loss import prototypical_loss
# From outside
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if PHASE2_SCHEME == 'NonFewShot-MultiGPU':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
import apex
import csv
import datetime
import math
import pprint
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmeta
import torchvision
from apex.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchvision import datasets
from torchvision.transforms import transforms

"""
Written referring to:
    https://chaelin0722.github.io/etc/gpu_utility/
"""

def nonfewshot_multigpu(
    phase1_scheme,
    workers,
    dataset, dataset_path,
    model_arch, pretrained,
    q_method,
    q_bits_w_first,
    q_bits_w_inter_list_list_given, q_bits_a_inter_list_list_given,
    q_bits_w_last, q_bits_a_last,
    last_epoch,
    batch_size,
    q_subtasks, c_batches,
    first_seed_ungiven_q_subtasks,
    track_running_stats):
    torch.distributed.init_process_group(backend='nccl', init_method="env://", rank=0, world_size=1)  # rank should be 0 ~ world_size-1

    phase2_scheme_lowercase = 'nonfewshot_multigpu'
    phase1_scheme_lowercase_dict = {
        'MEBQAT-NonFewShot-MultiGPU': 'mebqat_nonfewshot_multigpu'}
    if phase1_scheme is None:
        phase1_scheme_lowercase = 'none'
    else:
        phase1_scheme_lowercase = phase1_scheme_lowercase_dict[phase1_scheme]
    if not os.path.exists(f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/'):
        os.makedirs(f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/')

    log_path = f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    f_csv = open(f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/{now}.csv', 'w', encoding='utf-8', newline='')
    writer_csv = csv.writer(f_csv)
    writer_csv.writerow(['avg_acc_test'])

    f_txt = open(f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/{now}_q_subtasks.txt', 'a')

    num_classes_dict = {
        'ImageNet': 1000,
        'CIFAR100': 100,
        'CIFAR10': 10,
        'SVHN': 10}
    num_classes = num_classes_dict[dataset]

    if model_arch in ['MobileNetV2ForCIFAR', 'PreActResNet50', 'PreActResNet20', 'CNNForSVHN']:    # Cannot be pretrained
        pretrained = False

    if phase1_scheme == 'MEBQAT-NonFewShot-MultiGPU':
        track_running_stats = False

    # Dataset
    if dataset == 'ImageNet':
        temp_transform_list = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        test_transform = transforms.Compose(temp_transform_list)
        test_set = datasets.ImageNet(
            root=dataset_path, split='val',
            transform=test_transform)
    elif dataset == 'CIFAR100':
        temp_transform_list = [
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]))
        test_transform = transforms.Compose(temp_transform_list)
        test_set = datasets.CIFAR100(
            dataset_path, train=False,
            download=True,
            transform=test_transform)       
    elif dataset == 'CIFAR10':
        temp_transform_list = [
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
        test_transform = transforms.Compose(temp_transform_list)
        test_set = datasets.CIFAR10(
            dataset_path, train=False,
            download=True,
            transform=test_transform)      
    elif dataset == 'SVHN':
        temp_transform_list = [
            transforms.Resize(40),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]))
        test_transform = transforms.Compose(temp_transform_list)
        test_set = datasets.SVHN(
            dataset_path, split='test', 
            download=True,
            transform=test_transform)
    test_sampler = DistributedSampler(test_set)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True, sampler=test_sampler)
    if q_method == 'LSQ':
        num_gpus = torch.cuda.device_count()
        temp_loader = DataLoader(test_set, batch_size=num_gpus, shuffle=False, pin_memory=True, sampler=test_sampler)
    c_batches = min(c_batches, len(test_loader))

    log('******************************          Arguments          ******************************')
    log(f'Time: {now}')
    log(f'Phase 2: {phase2_scheme_lowercase}')

    log(f'Phase 1: {phase1_scheme_lowercase}')
    log(f'# of workers: {workers}')
    log(f'Dataset: {dataset}, with path {dataset_path}')
    temp_text = ('pretrained' if pretrained else 'from scratch') + ', in ' + ('inductive setting' if track_running_stats else 'transductive setting')
    log(f'Model architecture: {model_arch}, {temp_text}')
    log(f'Method of quantization and/or QAT: {q_method}')
    log(f'For the first layer, use {q_bits_w_first}-bit weight')    
    log(f'For the last layer, use {q_bits_w_last}-bit weight and {q_bits_a_last}-bit activation') 
    log(f'Check {now}_q_subtasks.txt to get the full information of quantization subtasks')
    log(f'Last phase 1 epoch: {last_epoch}')
    log(f'Batch size: {batch_size}')
    log(f'# of quantization subtasks at test: {q_subtasks} (each one includes {c_batches} classification batches)')
    log(f'1st (or starting) random seed for ungiven quantization subtasks: {first_seed_ungiven_q_subtasks}')
    log('Not mixed precision')    

    log('\n******************************           Phase 2           ******************************')
    # Loss function
    loss_func = nn.CrossEntropyLoss().to(device)

    # Last thing of dataset
    log(f'# of test tasks: {q_subtasks} * 1 = {q_subtasks}\n')

    ## Quantization
    assert len(q_bits_w_inter_list_list_given) == len(q_bits_a_inter_list_list_given)
    assert 0 <= len(q_bits_w_inter_list_list_given) <= q_subtasks   # 0 <= q_subtasks_given <= q_subtasks
    q_subtasks_given = len(q_bits_w_inter_list_list_given)
    if q_subtasks_given != q_subtasks:  
        # One or more ungiven quantization subtask exist(s)
        q_bits_cand_list = [2, 3, 4, 5, 6, 7, 8, 16, None]  # NOTE: it should be in ascending order
        if q_method in ['Yu21', 'Sun21']:
            q_bits_cand_list.insert(0, 1)
        log(f'List of quantization bitwidth candidates: {q_bits_cand_list}\n')
        seed_ungiven_q_subtasks = first_seed_ungiven_q_subtasks
    q_bits_wsas_dict = {
        # model_arch: (q_bits_ws, q_bits_as)
        'ResNet18': (21, 17),
        'ResNet50': (54, 49),
        'MobileNetV2ForImageNet': (53, 35),
        'MobileNetV2ForCIFAR': (53, 35),
        'PreActResNet50': (54, 49),
        'PreActResNet20': (22, 18),
        'CNNForSVHN': (8, 7)}
    q_bits_ws, q_bits_as = q_bits_wsas_dict[model_arch]
    if q_subtasks_given != 0:
        assert all(len(q_bits_w_inter_list) == (q_bits_ws - 2) for q_bits_w_inter_list in q_bits_w_inter_list_list_given)
        assert all(len(q_bits_a_inter_list) == (q_bits_as - 1) for q_bits_a_inter_list in q_bits_a_inter_list_list_given)
    ##

    for q_st in range(1, q_subtasks + 1):
        sum_acc_test = 0.0

        if q_st <= q_subtasks_given:
            q_bits_w_list = q_bits_w_inter_list_list_given[q_st - 1]
            q_bits_a_list = q_bits_a_inter_list_list_given[q_st - 1]
        else:
            if seed_ungiven_q_subtasks is not None:
                random.seed(seed_ungiven_q_subtasks)    # Set 

            while True:
                b = random.randint(0, len(q_bits_cand_list) - 1)
                if seed_ungiven_q_subtasks is not None:
                    seed_ungiven_q_subtasks += 2
                    random.seed(seed_ungiven_q_subtasks)
                q_bits_w = q_bits_cand_list[b]

                b = random.randint(0, len(q_bits_cand_list) - 1)
                if seed_ungiven_q_subtasks is not None:
                    seed_ungiven_q_subtasks += 2
                    random.seed(seed_ungiven_q_subtasks)
                q_bits_a = q_bits_cand_list[b]

                reselect_a = any([
                    (q_bits_w is None) and (q_bits_a is not None),
                    (q_bits_w == 1) and (q_bits_a not in [1, None]),
                    (q_bits_w != 1) and (q_bits_a == 1)])

                if reselect_a: continue
                break                    
            q_bits_w_list = [q_bits_w] * (q_bits_ws - 2)
            q_bits_a_list = [q_bits_a] * (q_bits_as - 1)

            if seed_ungiven_q_subtasks is not None:
                random.seed()                           # Reset
        # At this time, q_bits_w_list and q_bits_a_list don't include first- and last-layer bitwidths
        full_precision = all(w is None for w in q_bits_w_list) and all(a is None for a in q_bits_a_list)
        q_bits_w_first_, q_bits_w_last_, q_bits_a_last_ = q_bits_w_first, q_bits_w_last, q_bits_a_last
        if full_precision:
            q_bits_w_first_ = None
            q_bits_w_last_, q_bits_a_last_ = None, None 
        else:
            if q_bits_w_first == 'same':
                q_bits_w_first_ = q_bits_w_list[0]
            if q_bits_w_last == 'same':
                q_bits_w_last_ = q_bits_w_list[-1]
            if q_bits_a_last == 'same':
                q_bits_a_last_ = q_bits_a_list[-1]
        q_bits_w_list.insert(0, q_bits_w_first_)
        q_bits_w_list.append(q_bits_w_last_)
        q_bits_a_list.append(q_bits_a_last_)

        f_txt.write(f'-------------- Quantization subtask {q_st} --------------\n')
        f_txt.write(f'q_bits_w_list: {q_bits_w_list}\n')
        f_txt.write(f'q_bits_a_list: {q_bits_a_list}\n\n')

        ## Test model
        if model_arch == 'ResNet18':
            model_test = ResNet18(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained)
        elif model_arch == 'ResNet50':
            model_test = ResNet50(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained)
        elif model_arch == 'MobileNetV2ForImageNet':
            model_test = MobileNetV2ForImageNet(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained)
        elif model_arch == 'MobileNetV2ForCIFAR':
            model_test = MobileNetV2ForCIFAR(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained)
        elif model_arch == 'PreActResNet50':
            model_test = PreActResNet50(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained)
        elif model_arch == 'PreActResNet20':
            model_test = PreActResNet20(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained)
        elif model_arch == 'CNNForSVHN':
            model_test = CNNForSVHN(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes, 
                pretrained=pretrained)
        model_test = nn.DataParallel(model_test).to(device)
        model_test = DDP(model_test, delay_allreduce=True)
        if q_method == 'LSQ':
            with torch.no_grad():
                if q_st == 1:
                    temp_imgs, _ = next(iter(temp_loader))
                    temp_imgs = temp_imgs.to(device)
                _ = model_test(temp_imgs)

        if last_epoch != 0:
            model_test.load_state_dict(
                torch.load(f'./{phase1_scheme_lowercase}/checkpoints/model_outer_e{last_epoch}.pth'), 
                strict=False)
            if q_st == 1:
                log(f'Successfully loaded model with {last_epoch}-th phase 1 epoch.\n')
        ##

        model_test.eval()

        test_iter = iter(test_loader)
        for c_bat in range(1, c_batches + 1):
            imgs, labels = next(test_iter)
            imgs, labels = imgs.to(device), labels.to(device)

            ## Forward
            with torch.no_grad():
                outputs = model_test(imgs)
                loss = loss_func(outputs, labels)

                _, preds = torch.max(outputs.data, 1)
                total_examples = labels.size(0)
                correct_examples = (preds == labels).sum().item()
            sum_acc_test += 100 * correct_examples / float(total_examples)
            ##

        avg_acc_test = sum_acc_test / float(c_batches)
        log(f'\n  Phase 2 | {q_st}-th quantization subtask')
        log(f'  Test | Avg over {c_batches} batches | accuracy: {avg_acc_test:.2f}%')
        writer_csv.writerow([avg_acc_test])

    log('\n*****************************************************************************************')
    print(f'{now}\n')
    f_csv.close()
    f_txt.close()

def nonfewshot(
    phase1_scheme,
    workers,
    dataset, dataset_path,
    model_arch, pretrained,
    q_method,
    q_bits_w_first,
    q_bits_w_inter_list_list_given, q_bits_a_inter_list_list_given,
    q_bits_w_last, q_bits_a_last,
    last_epoch,
    batch_size,
    q_subtasks, c_batches,
    first_seed_ungiven_q_subtasks,
    track_running_stats):
    phase2_scheme_lowercase = 'nonfewshot'
    phase1_scheme_lowercase_dict = {
        'MEBQAT-NonFewShot': 'mebqat_nonfewshot'}
    if phase1_scheme is None:
        phase1_scheme_lowercase = 'none'
    else:
        phase1_scheme_lowercase = phase1_scheme_lowercase_dict[phase1_scheme]
    if not os.path.exists(f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/'):
        os.makedirs(f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/')

    log_path = f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    f_csv = open(f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/{now}.csv', 'w', encoding='utf-8', newline='')
    writer_csv = csv.writer(f_csv)
    writer_csv.writerow(['avg_acc_test'])

    f_txt = open(f'./{phase1_scheme_lowercase}/reports/phase2/{phase2_scheme_lowercase}/{now}_q_subtasks.txt', 'a')

    num_classes_dict = {
        'ImageNet': 1000,
        'CIFAR100': 100,
        'CIFAR10': 10,
        'SVHN': 10}
    num_classes = num_classes_dict[dataset]

    if model_arch in ['MobileNetV2ForCIFAR', 'PreActResNet50', 'PreActResNet20', 'CNNForSVHN']:    # Cannot be pretrained
        pretrained = False

    if phase1_scheme == 'MEBQAT-NonFewShot':
        track_running_stats = False

    # Dataset
    if dataset == 'ImageNet':
        temp_transform_list = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        test_transform = transforms.Compose(temp_transform_list)
        test_set = datasets.ImageNet(
            root=dataset_path, split='val',
            transform=test_transform)
    elif dataset == 'CIFAR100':
        temp_transform_list = [
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]))
        test_transform = transforms.Compose(temp_transform_list)
        test_set = datasets.CIFAR100(
            dataset_path, train=False,
            download=True,
            transform=test_transform)       
    elif dataset == 'CIFAR10':
        temp_transform_list = [
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
        test_transform = transforms.Compose(temp_transform_list)
        test_set = datasets.CIFAR10(
            dataset_path, train=False,
            download=True,
            transform=test_transform)      
    elif dataset == 'SVHN':
        temp_transform_list = [
            transforms.Resize(40),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]))
        test_transform = transforms.Compose(temp_transform_list)
        test_set = datasets.SVHN(
            dataset_path, split='test', 
            download=True,
            transform=test_transform)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True)
    if q_method == 'LSQ':
        temp_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    c_batches = min(c_batches, len(test_loader))

    log('******************************          Arguments          ******************************')
    log(f'Time: {now}')
    log(f'Phase 2: {phase2_scheme_lowercase}')

    log(f'Phase 1: {phase1_scheme_lowercase}')
    log(f'# of workers: {workers}')
    log(f'Dataset: {dataset}, with path {dataset_path}')
    temp_text = ('pretrained' if pretrained else 'from scratch') + ', in ' + ('inductive setting' if track_running_stats else 'transductive setting')
    log(f'Model architecture: {model_arch}, {temp_text}')
    log(f'Method of quantization and/or QAT: {q_method}')
    log(f'For the first layer, use {q_bits_w_first}-bit weight')    
    log(f'For the last layer, use {q_bits_w_last}-bit weight and {q_bits_a_last}-bit activation') 
    log(f'Check {now}_q_subtasks.txt to get the full information of quantization subtasks')
    log(f'Last phase 1 epoch: {last_epoch}')
    log(f'Batch size: {batch_size}')
    log(f'# of quantization subtasks at test: {q_subtasks} (each one includes {c_batches} classification batches)')
    log(f'1st (or starting) random seed for ungiven quantization subtasks: {first_seed_ungiven_q_subtasks}')
    log('Not mixed precision')    

    log('\n******************************           Phase 2           ******************************')
    # Loss function
    loss_func = nn.CrossEntropyLoss().to(device)

    # Last thing of dataset
    log(f'# of test tasks: {q_subtasks} * 1 = {q_subtasks}\n')

    ## Quantization
    assert len(q_bits_w_inter_list_list_given) == len(q_bits_a_inter_list_list_given)
    assert 0 <= len(q_bits_w_inter_list_list_given) <= q_subtasks   # 0 <= q_subtasks_given <= q_subtasks
    q_subtasks_given = len(q_bits_w_inter_list_list_given)
    if q_subtasks_given != q_subtasks:  
        # One or more ungiven quantization subtask exist(s)
        q_bits_cand_list = [2, 3, 4, 5, 6, 7, 8, 16, None]  # NOTE: it should be in ascending order
        if q_method in ['Yu21', 'Sun21']:
            q_bits_cand_list.insert(0, 1)
        log(f'List of quantization bitwidth candidates: {q_bits_cand_list}\n')
        seed_ungiven_q_subtasks = first_seed_ungiven_q_subtasks

    q_bits_wsas_dict = {
        # model_arch: (q_bits_ws, q_bits_as)
        'ResNet18': (21, 17),
        'ResNet50': (54, 49),
        'MobileNetV2ForImageNet': (53, 35),
        'MobileNetV2ForCIFAR': (53, 35),
        'PreActResNet50': (54, 49),
        'PreActResNet20': (22, 18),
        'CNNForSVHN': (8, 7)}
    q_bits_ws, q_bits_as = q_bits_wsas_dict[model_arch]
    if q_subtasks_given != 0:
        assert all(len(q_bits_w_inter_list) == (q_bits_ws - 2) for q_bits_w_inter_list in q_bits_w_inter_list_list_given)
        assert all(len(q_bits_a_inter_list) == (q_bits_as - 1) for q_bits_a_inter_list in q_bits_a_inter_list_list_given)
    ##

    for q_st in range(1, q_subtasks + 1):
        sum_acc_test = 0.0

        if q_st <= q_subtasks_given:
            q_bits_w_list = q_bits_w_inter_list_list_given[q_st - 1]
            q_bits_a_list = q_bits_a_inter_list_list_given[q_st - 1]
        else:
            if seed_ungiven_q_subtasks is not None:
                random.seed(seed_ungiven_q_subtasks)    # Set 

            while True:
                b = random.randint(0, len(q_bits_cand_list) - 1)
                if seed_ungiven_q_subtasks is not None:
                    seed_ungiven_q_subtasks += 2
                    random.seed(seed_ungiven_q_subtasks)
                q_bits_w = q_bits_cand_list[b]

                b = random.randint(0, len(q_bits_cand_list) - 1)
                if seed_ungiven_q_subtasks is not None:
                    seed_ungiven_q_subtasks += 2
                    random.seed(seed_ungiven_q_subtasks)
                q_bits_a = q_bits_cand_list[b]

                reselect_a = any([
                    (q_bits_w is None) and (q_bits_a is not None),
                    (q_bits_w == 1) and (q_bits_a not in [1, None]),
                    (q_bits_w != 1) and (q_bits_a == 1)])

                if reselect_a: continue
                break                    
            q_bits_w_list = [q_bits_w] * (q_bits_ws - 2)
            q_bits_a_list = [q_bits_a] * (q_bits_as - 1)

            if seed_ungiven_q_subtasks is not None:
                random.seed()                           # Reset
        # At this time, q_bits_w_list and q_bits_a_list don't include first- and last-layer bitwidths
        full_precision = all(w is None for w in q_bits_w_list) and all(a is None for a in q_bits_a_list)
        q_bits_w_first_, q_bits_w_last_, q_bits_a_last_ = q_bits_w_first, q_bits_w_last, q_bits_a_last
        if full_precision:
            q_bits_w_first_ = None
            q_bits_w_last_, q_bits_a_last_ = None, None 
        else:
            if q_bits_w_first == 'same':
                q_bits_w_first_ = q_bits_w_list[0]
            if q_bits_w_last == 'same':
                q_bits_w_last_ = q_bits_w_list[-1]
            if q_bits_a_last == 'same':
                q_bits_a_last_ = q_bits_a_list[-1]
        q_bits_w_list.insert(0, q_bits_w_first_)
        q_bits_w_list.append(q_bits_w_last_)
        q_bits_a_list.append(q_bits_a_last_)

        f_txt.write(f'-------------- Quantization subtask {q_st} --------------\n')
        f_txt.write(f'q_bits_w_list: {q_bits_w_list}\n')
        f_txt.write(f'q_bits_a_list: {q_bits_a_list}\n\n')

        ## Test model
        if model_arch == 'ResNet18':
            model_test = ResNet18(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained).to(device)
        elif model_arch == 'ResNet50':
            model_test = ResNet50(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained).to(device)
        elif model_arch == 'MobileNetV2ForImageNet':
            model_test = MobileNetV2ForImageNet(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained).to(device)
        elif model_arch == 'MobileNetV2ForCIFAR':
            model_test = MobileNetV2ForCIFAR(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained).to(device)
        elif model_arch == 'PreActResNet50':
            model_test = PreActResNet50(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained).to(device)
        elif model_arch == 'PreActResNet20':
            model_test = PreActResNet20(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes,
                pretrained=pretrained).to(device)
        elif model_arch == 'CNNForSVHN':
            model_test = CNNForSVHN(
                q_method,
                q_bits_w_list, q_bits_a_list,
                track_running_stats,
                num_classes=num_classes, 
                pretrained=pretrained).to(device)
        if q_method == 'LSQ':
            with torch.no_grad():
                if q_st == 1:
                    temp_imgs, _ = next(iter(temp_loader))
                    temp_imgs = temp_imgs.to(device)
                _ = model_test(temp_imgs)

        if last_epoch != 0:
            model_test.load_state_dict(
                torch.load(f'./{phase1_scheme_lowercase}/checkpoints/model_outer_e{last_epoch}.pth'), 
                strict=False)
            if q_st == 1:
                log(f'Successfully loaded model with {last_epoch}-th phase 1 epoch.\n')
        ##

        model_test.eval()

        test_iter = iter(test_loader)
        for c_bat in range(1, c_batches + 1):
            imgs, labels = next(test_iter)
            imgs, labels = imgs.to(device), labels.to(device)

            ## Forward
            with torch.no_grad():
                outputs = model_test(imgs)
                loss = loss_func(outputs, labels)

                _, preds = torch.max(outputs.data, 1)
                total_examples = labels.size(0)
                correct_examples = (preds == labels).sum().item()
            sum_acc_test += 100 * correct_examples / float(total_examples)
            ##

        avg_acc_test = sum_acc_test / float(c_batches)
        log(f'\n  Phase 2 | {q_st}-th quantization subtask')
        log(f'  Test | Avg over {c_batches} batches | accuracy: {avg_acc_test:.2f}%')
        writer_csv.writerow([avg_acc_test])

    log('\n*****************************************************************************************')
    print(f'{now}\n')
    f_csv.close()
    f_txt.close()

def main():
    if PHASE2_SCHEME == 'NonFewShot-MultiGPU':
        nonfewshot_multigpu(
            PHASE1_SCHEME,
            WORKERS,
            DATASET, DATASET_PATH,
            MODEL_ARCH, PRETRAINED,
            Q_METHOD,
            Q_BITS_W_FIRST,
            Q_BITS_W_INTER_LIST_LIST_GIVEN, Q_BITS_A_INTER_LIST_LIST_GIVEN,
            Q_BITS_W_LAST, Q_BITS_A_LAST,
            LAST_EPOCH,
            BATCH_SIZE,
            Q_SUBTASKS, C_BATCHES,
            FIRST_SEED_UNGIVEN_Q_SUBTASKS,
            TRACK_RUNNING_STATS)
    elif PHASE2_SCHEME == 'NonFewShot':
        nonfewshot(
            PHASE1_SCHEME,
            WORKERS,
            DATASET, DATASET_PATH,
            MODEL_ARCH, PRETRAINED,
            Q_METHOD,
            Q_BITS_W_FIRST,
            Q_BITS_W_INTER_LIST_LIST_GIVEN, Q_BITS_A_INTER_LIST_LIST_GIVEN,
            Q_BITS_W_LAST, Q_BITS_A_LAST,
            LAST_EPOCH,
            BATCH_SIZE,
            Q_SUBTASKS, C_BATCHES,
            FIRST_SEED_UNGIVEN_Q_SUBTASKS,
            TRACK_RUNNING_STATS)

if __name__ == '__main__':
    # Detect anomaly for debugging
    autograd.set_detect_anomaly(True)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate timestamp
    now = datetime.datetime.utcnow()
    time_gap = datetime.timedelta(hours=9)
    now += time_gap
    now = now.strftime("%Y%m%d_%H%M%S") 

    assert (PHASE1_SCHEME is None) == (LAST_EPOCH == 0)

    main()
