# From own code
from quant import WeightQuantizer, ActivationQuantizer
from arch import BatchNorm2d, BatchNorm2dOnlyBeta
from arch import ResNet18, ResNet50, MobileNetV2ForImageNet, MobileNetV2ForCIFAR, PreActResNet50, PreActResNet20, CNNForSVHN, CNNForMAML, CNNForProtoNet
from arg_phase1 import *
from util import create_log_func, rgetattr, rdelattr, rsetparam
from loss import CrossEntropyLossSoft, prototypical_loss
# From outside
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if PHASE1_SCHEME == 'MEBQAT-NonFewShot-MultiGPU':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
import apex
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
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchvision import datasets
from torchvision.transforms import transforms

"""
Written referring to:
    https://chaelin0722.github.io/etc/gpu_utility/
"""

def mebqat_nonfewshot_multigpu(
    workers,
    dataset, dataset_path,
    model_arch, pretrained,
    q_method,
    q_bits_w_first,
    q_bits_w_last, q_bits_a_last,
    last_epoch,
    last_best_avg_acc_val,
    epochs,
    batch_size,
    outer_optim, outer_optim_kwargs,
    outer_lr_sch, outer_lr_sch_kwargs,
    inner_q_subtasks,
    q_bits_wa_guaranteed_list_given,
    distill_knowledge,
    save_period, report_period):
    torch.distributed.init_process_group(backend='nccl', init_method="env://", rank=0, world_size=1)  # rank should be 0 ~ world_size-1

    phase1_scheme_lowercase = 'mebqat_nonfewshot_multigpu'
    if not os.path.exists(f'./{phase1_scheme_lowercase}/reports/phase1/'):
        os.makedirs(f'./{phase1_scheme_lowercase}/reports/phase1/')
    if not os.path.exists(f'./{phase1_scheme_lowercase}/checkpoints/'):
        os.makedirs(f'./{phase1_scheme_lowercase}/checkpoints/')

    log_path = f'./{phase1_scheme_lowercase}/reports/phase1/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    num_classes_dict = {
        'ImageNet': 1000,
        'CIFAR100': 100,
        'CIFAR10': 10,
        'SVHN': 10}
    num_classes = num_classes_dict[dataset]

    if model_arch in ['MobileNetV2ForCIFAR', 'PreActResNet50', 'PreActResNet20', 'CNNForSVHN']:    # Cannot be pretrained
        pretrained = False

    log('******************************          Arguments          ******************************')
    log(f'Time: {now}')
    log(f'Phase 1: {phase1_scheme_lowercase}')

    log(f'# of workers: {workers}')
    log(f'Dataset: {dataset}, with path {dataset_path}')
    log(f'Model architecture: {model_arch}, transductive setting, whether to start with a pretrained model: {pretrained}')
    log(f'Method of quantization and/or QAT: {q_method}')
    log(f'For the first layer, use {q_bits_w_first}-bit weight')
    log(f'For the last layer, use {q_bits_w_last}-bit weight and {q_bits_a_last}-bit activation')     
    log(f'Last phase 1 epoch: {last_epoch}')
    log(f'Last best validation accuracy: {last_best_avg_acc_val}')
    log(f'Phase 1 epochs: {epochs}')
    log(f'Batch size: {batch_size}')
    log(f'Outer-loop optimizer: {outer_optim}, with keyword arguments {outer_optim_kwargs}')
    log(f'Learning rate scheduler: {outer_lr_sch}, with keyword arguments {outer_lr_sch_kwargs}')
    log(f'# of quantization bitwidth subtasks per outer-loop: {inner_q_subtasks}')
    log(f'List of quantization bitwidths (in uniform-precision) guaranteed to appear in an inner-loop at every outer-loop: {q_bits_wa_guaranteed_list_given}')
    log(f'Whether to distill knowledge: {distill_knowledge}') 
    log(f'Save period: {save_period} epoch(s), report period: {report_period} step(s)')
    log('Not mixed precision')

    log('\n******************************           Phase 1           ******************************')
    # Loss function
    loss_func = nn.CrossEntropyLoss().to(device)
    if distill_knowledge:
        distill_loss_func = CrossEntropyLossSoft().to(device)

    # Dataset
    if dataset == 'ImageNet':
        temp_transform_list = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        train_transform = transforms.Compose(temp_transform_list)
        train_set = datasets.ImageNet(
            root=dataset_path, split='train',
            transform=train_transform)
        temp_transform_list = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        val_transform = transforms.Compose(temp_transform_list)
        val_set = datasets.ImageNet(
            root=dataset_path, split='val',
            transform=val_transform)
    elif dataset == 'CIFAR100':
        temp_transform_list = [
            transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]))
        train_transform = transforms.Compose(temp_transform_list)
        train_set = datasets.CIFAR100(
            dataset_path, train=True, 
            download=True,
            transform=train_transform)
        temp_transform_list = [
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]))
        val_transform = transforms.Compose(temp_transform_list)
        val_set = datasets.CIFAR100(
            dataset_path, train=False,
            download=True,
            transform=val_transform)      
    elif dataset == 'CIFAR10':
        temp_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
        train_transform = transforms.Compose(temp_transform_list)
        train_set = datasets.CIFAR10(
            dataset_path, train=True, 
            download=True,
            transform=train_transform)
        temp_transform_list = [
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
        val_transform = transforms.Compose(temp_transform_list)
        val_set = datasets.CIFAR10(
            dataset_path, train=False,
            download=True,
            transform=val_transform)       
    elif dataset == 'SVHN':
        temp_transform_list = [
            transforms.Resize(40),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]))
        train_transform = transforms.Compose(temp_transform_list)
        train_set = datasets.SVHN(
            dataset_path, split='train', 
            download=True,
            transform=train_transform)
        extra_train_set = datasets.SVHN(
            dataset_path, split='extra', 
            download=True,
            transform=train_transform)
        train_set = ConcatDataset([train_set, extra_train_set])
        temp_transform_list = [
            transforms.Resize(40),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]))
        val_transform = transforms.Compose(temp_transform_list)
        val_set = datasets.SVHN(
            dataset_path, split='test', 
            download=True,
            transform=val_transform)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True, sampler=train_sampler)
    if q_method == 'LSQ':
        num_gpus = torch.cuda.device_count()
        temp_loader = DataLoader(train_set, batch_size=num_gpus, shuffle=False, pin_memory=True, sampler=train_sampler)
    val_sampler = DistributedSampler(val_set)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True, sampler=val_sampler)
    steps_per_epoch = len(train_loader)
    val_tasks_per_epoch = len(val_loader)
    log(f'# of steps per epoch: {steps_per_epoch}')
    log(f'# of validation tasks per epoch: {val_tasks_per_epoch}\n')

    # Quantization
    q_bits_cand_list = [2, 3, 4, 5, 6, 7, 8, 16, None]  # NOTE: it should be in ascending order
    if q_method in ['Yu21', 'Sun21']:
        q_bits_cand_list.insert(0, 1)
    log(f'List of quantization bitwidth candidates: {q_bits_cand_list}\n')
    if q_bits_wa_guaranteed_list_given is not None:
        for q_bits_wa_guaranteed in q_bits_wa_guaranteed_list_given:
            assert q_bits_wa_guaranteed[0] in q_bits_cand_list
            assert q_bits_wa_guaranteed[1] in q_bits_cand_list        
    q_bits_wsas_dict = {
        # model_arch: (q_bits_ws, q_bits_as)
        'ResNet18': (21, 17),
        'ResNet50': (54, 49),
        'MobileNetV2ForImageNet': (53, 35),
        'MobileNetV2ForCIFAR': (53, 35),
        'PreActResNet50': (54, 49),
        'PreActResNet20': (22, 18),
        'CNNForSVHN': (8, 7)}
    if model_arch in q_bits_wsas_dict:
        q_bits_ws, q_bits_as = q_bits_wsas_dict[model_arch]
    else:
        q_bits_ws, q_bits_as = 1234, 1234     # Dummy large number

    ## Outer-loop model, optimizer, and lr scheduler
    if model_arch == 'ResNet18':
        model_outer = ResNet18(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained)
    elif model_arch == 'ResNet50':
        model_outer = ResNet50(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained)
    elif model_arch == 'MobileNetV2ForImageNet':
        model_outer = MobileNetV2ForImageNet(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained)
    elif model_arch == 'MobileNetV2ForCIFAR':
        model_outer = MobileNetV2ForCIFAR(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained)
    elif model_arch == 'PreActResNet50':
        model_outer = PreActResNet50(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained)
    elif model_arch == 'PreActResNet20':
        model_outer = PreActResNet20(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained)
    elif model_arch == 'CNNForSVHN':
        model_outer = CNNForSVHN(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes, 
            pretrained=pretrained)
    model_outer = nn.DataParallel(model_outer).to(device)
    model_outer = DDP(model_outer, delay_allreduce=True) 
    if q_method == 'LSQ':
        with torch.no_grad():
            temp_imgs, _ = next(iter(temp_loader))
            temp_imgs = temp_imgs.to(device)
            _ = model_outer(temp_imgs)

    optimizer_outer = outer_optim(model_outer.parameters(), **outer_optim_kwargs)

    if outer_lr_sch == 'Warmup-StepwiseCosine':
        total_iters = (last_epoch + epochs) * steps_per_epoch
        warmup_iters = 5 * steps_per_epoch
        last_iters = last_epoch * steps_per_epoch
        lr_dict = {}
        bs_ratio = 256 / batch_size
        for i in range(warmup_iters):
            if i >= last_iters:
                lr_dict[i - last_iters] = (1 - bs_ratio) / warmup_iters * i + bs_ratio
        for i in range(warmup_iters, total_iters):
            if i >= last_iters:            
                lr_dict[i - last_iters] = (1.0 + math.cos((i - warmup_iters) * math.pi / (total_iters - warmup_iters))) / 2
        lr_lambda = lambda itr: lr_dict[itr]
        lr_scheduler_outer = optim.lr_scheduler.LambdaLR(optimizer_outer, lr_lambda=lr_lambda)    # Neither to save nor to load state
    else:
        lr_scheduler_outer = outer_lr_sch(optimizer_outer, **outer_lr_sch_kwargs)

    if last_epoch != 0:
        model_outer.load_state_dict(
            torch.load(f'./{phase1_scheme_lowercase}/checkpoints/model_outer_e{last_epoch}.pth'),
            strict=True)
        optimizer_outer.load_state_dict(
            torch.load(f'./{phase1_scheme_lowercase}/checkpoints/optimizer_outer_e{last_epoch}.pth'))
        if outer_lr_sch != 'Warmup-StepwiseCosine':
            lr_scheduler_outer.load_state_dict(
                torch.load(f'./{phase1_scheme_lowercase}/checkpoints/lr_scheduler_outer_e{last_epoch}.pth'))
        temp_text = ', and lr scheduler' if outer_lr_sch != 'Warmup-StepwiseCosine' else ''
        log(f'Successfully loaded outer-loop model, optimizer{temp_text} with {last_epoch}-th phase 1 epoch.\n')
    ##

    if model_arch not in q_bits_wsas_dict: 
        q_bits_ws, q_bits_as = 0, 0
        for module in model_outer.modules():
            if type(module) == WeightQuantizer:
                q_bits_ws += 1
            if type(module) == ActivationQuantizer:
                q_bits_as += 1
        print(f'\nModel architecture: {model_arch}, q_bits_ws: {q_bits_ws}, q_bits_as: {q_bits_as}\n')

    best_avg_acc_val = last_best_avg_acc_val

    for epoch in range(last_epoch + 1, last_epoch + 1 + epochs):    # Outer-loop
        for step, (imgs, labels) in enumerate(train_loader, start=1):
            sum_outer_grad_dict = {}
            sum_outer_cnt_dict = {}
            optimizer_outer.zero_grad()

            imgs, labels = imgs.to(device), labels.to(device)

            for t in range(1, inner_q_subtasks + 1):                # Inner-loop
                if (q_bits_wa_guaranteed_list_given is not None) and (t in range(1, len(q_bits_wa_guaranteed_list_given) + 1)):
                    q_bits_w_list = [q_bits_wa_guaranteed_list_given[t - 1][0]] * (q_bits_ws - 2)
                    q_bits_a_list = [q_bits_wa_guaranteed_list_given[t - 1][1]] * (q_bits_as - 1)                    
                else:
                    while True:
                        b = random.randint(0, len(q_bits_cand_list) - 1)
                        q_bits_w = q_bits_cand_list[b]

                        b = random.randint(0, len(q_bits_cand_list) - 1)
                        q_bits_a = q_bits_cand_list[b]

                        reselect_a = any([
                            (q_bits_w is None) and (q_bits_a is not None),
                            (q_bits_w == 1) and (q_bits_a not in [1, None]),
                            (q_bits_w != 1) and (q_bits_a == 1)])

                        if reselect_a: continue
                        break                    
                    q_bits_w_list = [q_bits_w] * (q_bits_ws - 2)
                    q_bits_a_list = [q_bits_a] * (q_bits_as - 1)
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

                if (not distill_knowledge) or full_precision:       # No distillation for this inner-loop
                    ## Inner-loop model
                    if model_arch == 'ResNet18':
                        model_inner = ResNet18(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'ResNet50':
                        model_inner = ResNet50(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'MobileNetV2ForImageNet':
                        model_inner = MobileNetV2ForImageNet(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'MobileNetV2ForCIFAR':
                        model_inner = MobileNetV2ForCIFAR(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'PreActResNet50':
                        model_inner = PreActResNet50(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'PreActResNet20':
                        model_inner = PreActResNet20(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'CNNForSVHN':
                        model_inner = CNNForSVHN(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes, 
                            pretrained=pretrained)
                    model_inner = nn.DataParallel(model_inner).to(device)
                    model_inner = DDP(model_inner, delay_allreduce=True) 
                    if q_method == 'LSQ':
                        with torch.no_grad():
                            _ = model_inner(temp_imgs)

                    model_inner.load_state_dict(model_outer.state_dict(), strict=False)
                    ##

                    ## Forward
                    model_inner.zero_grad()
                    outputs = model_inner(imgs)
                    loss = loss_func(outputs, labels)

                    with torch.no_grad():
                        _, preds = torch.max(outputs.data, 1)
                        total_examples = labels.size(0)
                        correct_examples = (preds == labels).sum().item()

                    outer_grad_group = autograd.grad(loss, model_inner.parameters())
                    outer_grad_dict = {}
                    outer_cnt_dict = {}
                    for i, (key, _) in enumerate(model_inner.named_parameters()):
                        if outer_grad_group[i] is not None:
                            outer_grad_dict[key] = outer_grad_group[i]
                            outer_cnt_dict[key] = 1.0
                    ##
                else:                                               # Distillation for this inner-loop
                    ## Inner-loop teacher model
                    if model_arch == 'ResNet18':
                        model_inner = ResNet18(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'ResNet50':
                        model_inner = ResNet50(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'MobileNetV2ForImageNet':
                        model_inner = MobileNetV2ForImageNet(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'MobileNetV2ForCIFAR':
                        model_inner = MobileNetV2ForCIFAR(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'PreActResNet50':
                        model_inner = PreActResNet50(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'PreActResNet20':
                        model_inner = PreActResNet20(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'CNNForSVHN':
                        model_inner = CNNForSVHN(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes, 
                            pretrained=pretrained)
                    model_inner = nn.DataParallel(model_inner).to(device)
                    model_inner = DDP(model_inner, delay_allreduce=True)
                    if q_method == 'LSQ':
                        with torch.no_grad():
                            _ = model_inner(temp_imgs)

                    model_inner.load_state_dict(model_outer.state_dict(), strict=False)
                    ##

                    ## Forward
                    model_inner.zero_grad()
                    outputs = model_inner(imgs)

                    targets_soft = F.softmax(outputs.detach(), dim=1)
                    ##

                    ## Inner-loop student model
                    if model_arch == 'ResNet18':
                        model_inner = ResNet18(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'ResNet50':
                        model_inner = ResNet50(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'MobileNetV2ForImageNet':
                        model_inner = MobileNetV2ForImageNet(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'MobileNetV2ForCIFAR':
                        model_inner = MobileNetV2ForCIFAR(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'PreActResNet50':
                        model_inner = PreActResNet50(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'PreActResNet20':
                        model_inner = PreActResNet20(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained)
                    elif model_arch == 'CNNForSVHN':
                        model_inner = CNNForSVHN(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes, 
                            pretrained=pretrained)
                    model_inner = nn.DataParallel(model_inner).to(device)
                    model_inner = DDP(model_inner, delay_allreduce=True)
                    if q_method == 'LSQ':
                        with torch.no_grad():
                            _ = model_inner(temp_imgs)

                    model_inner.load_state_dict(model_outer.state_dict(), strict=False)
                    ##

                    ## Forward
                    model_inner.zero_grad()
                    outputs = model_inner(imgs)
                    loss = loss_func(outputs, labels)
                    loss += distill_loss_func(outputs, targets_soft)

                    with torch.no_grad():
                        _, preds = torch.max(outputs.data, 1)
                        total_examples = labels.size(0)
                        correct_examples = (preds == labels).sum().item()

                    outer_grad_group = autograd.grad(loss, model_inner.parameters())
                    outer_grad_dict = {}
                    outer_cnt_dict = {}
                    for i, (key, _) in enumerate(model_inner.named_parameters()):
                        if outer_grad_group[i] is not None:
                            outer_grad_dict[key] = outer_grad_group[i]
                            outer_cnt_dict[key] = 1.0
                    ##

                if t == 1:
                    sum_loss = loss.item()
                    sum_acc = 100 * correct_examples / float(total_examples)

                    for key, param in model_inner.named_parameters():
                        if key in outer_grad_dict:
                            sum_outer_grad_dict[key] = outer_grad_dict[key]
                            sum_outer_cnt_dict[key] = outer_cnt_dict[key]
                else:
                    sum_loss += loss.item()
                    sum_acc += 100 * correct_examples / float(total_examples)

                    for key, param in model_inner.named_parameters():
                        if key in outer_grad_dict:
                            if key in sum_outer_grad_dict:
                                sum_outer_grad_dict[key] += outer_grad_dict[key]
                                sum_outer_cnt_dict[key] += outer_cnt_dict[key]
                            else:
                                sum_outer_grad_dict[key] = outer_grad_dict[key]
                                sum_outer_cnt_dict[key] = outer_cnt_dict[key]

            if (step % report_period == 1) or (step == steps_per_epoch):
                avg_loss = sum_loss / float(inner_q_subtasks)
                avg_acc = sum_acc / float(inner_q_subtasks)
                log(f'\n  Phase 1 | {epoch}-th epoch, {step}-th step')
                log(f'  Training | Avg over {inner_q_subtasks} tasks | loss: {avg_loss:.3f}, accuracy: {avg_acc:.2f}%')

            for key, param in model_outer.named_parameters():
                if key in sum_outer_grad_dict:
                    param.grad = torch.clamp(
                        sum_outer_grad_dict[key] / float(sum_outer_cnt_dict[key]),
                        min=-10.0, max=10.0)

            optimizer_outer.step()

            if outer_lr_sch == 'Warmup-StepwiseCosine':
                lr_scheduler_outer.step()

        if outer_lr_sch != 'Warmup-StepwiseCosine':
            lr_scheduler_outer.step()

        if epoch % save_period == 0:
            torch.save(model_outer.state_dict(), f'./{phase1_scheme_lowercase}/checkpoints/model_outer_e{epoch}.pth')
            torch.save(optimizer_outer.state_dict(), f'./{phase1_scheme_lowercase}/checkpoints/optimizer_outer_e{epoch}.pth')
            if outer_lr_sch != 'Warmup-StepwiseCosine':
                torch.save(lr_scheduler_outer.state_dict(), f'./{phase1_scheme_lowercase}/checkpoints/lr_scheduler_outer_e{epoch}.pth')            
            temp_text = ', and lr scheduler' if outer_lr_sch != 'Warmup-StepwiseCosine' else ''
            log(f'Successfully saved outer-loop model, optimizer{temp_text} with {epoch}-th phase 1 epoch.')

        ## Validation per epoch
        for t, (imgs, labels) in enumerate(val_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)

            while True:
                b = random.randint(0, len(q_bits_cand_list) - 1)
                q_bits_w = q_bits_cand_list[b]

                b = random.randint(0, len(q_bits_cand_list) - 1)
                q_bits_a = q_bits_cand_list[b]

                reselect_a = any([
                    (q_bits_w is None) and (q_bits_a is not None),
                    (q_bits_w == 1) and (q_bits_a not in [1, None]),
                    (q_bits_w != 1) and (q_bits_a == 1)])

                if reselect_a: continue
                break                    
            q_bits_w_list = [q_bits_w] * (q_bits_ws - 2)
            q_bits_a_list = [q_bits_a] * (q_bits_as - 1)
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

            ## Validation model
            if model_arch == 'ResNet18':
                model_inner = ResNet18(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained)
            elif model_arch == 'ResNet50':
                model_inner = ResNet50(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained)
            elif model_arch == 'MobileNetV2ForImageNet':
                model_inner = MobileNetV2ForImageNet(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained)
            elif model_arch == 'MobileNetV2ForCIFAR':
                model_inner = MobileNetV2ForCIFAR(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained)
            elif model_arch == 'PreActResNet50':
                model_inner = PreActResNet50(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained)
            elif model_arch == 'PreActResNet20':
                model_inner = PreActResNet20(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained)
            elif model_arch == 'CNNForSVHN':
                model_inner = CNNForSVHN(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes, 
                    pretrained=pretrained)
            model_inner = nn.DataParallel(model_inner).to(device)
            model_inner = DDP(model_inner, delay_allreduce=True)
            if q_method == 'LSQ':
                with torch.no_grad():
                    _ = model_inner(temp_imgs)

            model_inner.load_state_dict(model_outer.state_dict(), strict=False)
            ##

            ## Forward
            model_inner.eval()

            with torch.no_grad():
                outputs = model_inner(imgs)
                loss = loss_func(outputs, labels)

                _, preds = torch.max(outputs.data, 1)
                total_examples = labels.size(0)
                correct_examples = (preds == labels).sum().item()
            loss_val = loss.item() 
            acc_val = 100 * correct_examples / float(total_examples)
            ##

            if t == 1:
                sum_loss_val = loss_val
                sum_acc_val = acc_val
            else:
                sum_loss_val += loss_val
                sum_acc_val += acc_val

        avg_loss_val = sum_loss_val / float(val_tasks_per_epoch)
        avg_acc_val = sum_acc_val / float(val_tasks_per_epoch)
        log(f'\n  Phase 2 | {epoch}-th epoch')
        log(f'  Validation | Avg over {val_tasks_per_epoch} tasks | loss: {avg_loss_val:.3f}, accuracy: {avg_acc_val:.2f}%')

        if best_avg_acc_val < avg_acc_val:
            best_avg_acc_val = avg_acc_val
            log('  Achieved best validation accuracy.\n')
        else:
            log('\n')
        ##

    log('\n*****************************************************************************************')
    print(f'{now}\n')

def mebqat_nonfewshot(
    workers,
    dataset, dataset_path,
    model_arch, pretrained,
    q_method,
    q_bits_w_first,
    q_bits_w_last, q_bits_a_last,
    last_epoch,
    last_best_avg_acc_val,
    epochs,
    batch_size,
    outer_optim, outer_optim_kwargs,
    outer_lr_sch, outer_lr_sch_kwargs,
    inner_q_subtasks,
    q_bits_wa_guaranteed_list_given,
    distill_knowledge,
    save_period, report_period):
    phase1_scheme_lowercase = 'mebqat_nonfewshot'
    if not os.path.exists(f'./{phase1_scheme_lowercase}/reports/phase1/'):
        os.makedirs(f'./{phase1_scheme_lowercase}/reports/phase1/')
    if not os.path.exists(f'./{phase1_scheme_lowercase}/checkpoints/'):
        os.makedirs(f'./{phase1_scheme_lowercase}/checkpoints/')

    log_path = f'./{phase1_scheme_lowercase}/reports/phase1/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    num_classes_dict = {
        'ImageNet': 1000,
        'CIFAR100': 100,
        'CIFAR10': 10,
        'SVHN': 10}
    num_classes = num_classes_dict[dataset]

    if model_arch in ['MobileNetV2ForCIFAR', 'PreActResNet50', 'PreActResNet20', 'CNNForSVHN']:    # Cannot be pretrained
        pretrained = False

    log('******************************          Arguments          ******************************')
    log(f'Time: {now}')
    log(f'Phase 1: {phase1_scheme_lowercase}')

    log(f'# of workers: {workers}')
    log(f'Dataset: {dataset}, with path {dataset_path}')
    log(f'Model architecture: {model_arch}, transductive setting, whether to start with a pretrained model: {pretrained}')
    log(f'Method of quantization and/or QAT: {q_method}')
    log(f'For the first layer, use {q_bits_w_first}-bit weight')
    log(f'For the last layer, use {q_bits_w_last}-bit weight and {q_bits_a_last}-bit activation')     
    log(f'Last phase 1 epoch: {last_epoch}')
    log(f'Last best validation accuracy: {last_best_avg_acc_val}')
    log(f'Phase 1 epochs: {epochs}')
    log(f'Batch size: {batch_size}')
    log(f'Outer-loop optimizer: {outer_optim}, with keyword arguments {outer_optim_kwargs}')
    log(f'Learning rate scheduler: {outer_lr_sch}, with keyword arguments {outer_lr_sch_kwargs}')
    log(f'# of quantization bitwidth subtasks per outer-loop: {inner_q_subtasks}')
    log(f'List of quantization bitwidths (in uniform-precision) guaranteed to appear in an inner-loop at every outer-loop: {q_bits_wa_guaranteed_list_given}')
    log(f'Whether to distill knowledge: {distill_knowledge}') 
    log(f'Save period: {save_period} epoch(s), report period: {report_period} step(s)')
    log('Not mixed precision')

    log('\n******************************           Phase 1           ******************************')
    # Loss function
    loss_func = nn.CrossEntropyLoss().to(device)
    if distill_knowledge:
        distill_loss_func = CrossEntropyLossSoft().to(device)

    # Dataset
    if dataset == 'ImageNet':
        temp_transform_list = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        train_transform = transforms.Compose(temp_transform_list)
        train_set = datasets.ImageNet(
            root=dataset_path, split='train',
            transform=train_transform)
        temp_transform_list = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        val_transform = transforms.Compose(temp_transform_list)
        val_set = datasets.ImageNet(
            root=dataset_path, split='val',
            transform=val_transform)
    elif dataset == 'CIFAR100':
        temp_transform_list = [
            transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]))
        train_transform = transforms.Compose(temp_transform_list)
        train_set = datasets.CIFAR100(
            dataset_path, train=True, 
            download=True,
            transform=train_transform)
        temp_transform_list = [
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]))
        val_transform = transforms.Compose(temp_transform_list)
        val_set = datasets.CIFAR100(
            dataset_path, train=False,
            download=True,
            transform=val_transform)      
    elif dataset == 'CIFAR10':
        temp_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
        train_transform = transforms.Compose(temp_transform_list)
        train_set = datasets.CIFAR10(
            dataset_path, train=True, 
            download=True,
            transform=train_transform)
        temp_transform_list = [
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
        val_transform = transforms.Compose(temp_transform_list)
        val_set = datasets.CIFAR10(
            dataset_path, train=False,
            download=True,
            transform=val_transform)       
    elif dataset == 'SVHN':
        temp_transform_list = [
            transforms.Resize(40),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]))
        train_transform = transforms.Compose(temp_transform_list)
        train_set = datasets.SVHN(
            dataset_path, split='train', 
            download=True,
            transform=train_transform)
        extra_train_set = datasets.SVHN(
            dataset_path, split='extra', 
            download=True,
            transform=train_transform)
        train_set = ConcatDataset([train_set, extra_train_set])
        temp_transform_list = [
            transforms.Resize(40),
            transforms.ToTensor()]
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            temp_transform_list.append(transforms.Lambda(lambda x: torch.round(255.0 * x) / 255.0))
        else:
            temp_transform_list.append(transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]))
        val_transform = transforms.Compose(temp_transform_list)
        val_set = datasets.SVHN(
            dataset_path, split='test', 
            download=True,
            transform=val_transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True)
    if q_method == 'LSQ':
        temp_loader = DataLoader(train_set, batch_size=1, pin_memory=True, shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True)
    steps_per_epoch = len(train_loader)
    val_tasks_per_epoch = len(val_loader)
    log(f'# of steps per epoch: {steps_per_epoch}')
    log(f'# of validation tasks per epoch: {val_tasks_per_epoch}\n')

    # Quantization
    q_bits_cand_list = [2, 3, 4, 5, 6, 7, 8, 16, None]  # NOTE: it should be in ascending order
    if q_method in ['Yu21', 'Sun21']:
        q_bits_cand_list.insert(0, 1)
    log(f'List of quantization bitwidth candidates: {q_bits_cand_list}\n')
    if q_bits_wa_guaranteed_list_given is not None:
        for q_bits_wa_guaranteed in q_bits_wa_guaranteed_list_given:
            assert q_bits_wa_guaranteed[0] in q_bits_cand_list
            assert q_bits_wa_guaranteed[1] in q_bits_cand_list        
    q_bits_wsas_dict = {
        # model_arch: (q_bits_ws, q_bits_as)
        'ResNet18': (21, 17),
        'ResNet50': (54, 49),
        'MobileNetV2ForImageNet': (53, 35),
        'MobileNetV2ForCIFAR': (53, 35),
        'PreActResNet50': (54, 49),
        'PreActResNet20': (22, 18),
        'CNNForSVHN': (8, 7)}
    if model_arch in q_bits_wsas_dict:
        q_bits_ws, q_bits_as = q_bits_wsas_dict[model_arch]
    else:
        q_bits_ws, q_bits_as = 1234, 1234     # Dummy large number

    ## Outer-loop model, optimizer, and lr scheduler
    if model_arch == 'ResNet18':
        model_outer = ResNet18(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'ResNet50':
        model_outer = ResNet50(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'MobileNetV2ForImageNet':
        model_outer = MobileNetV2ForImageNet(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'MobileNetV2ForCIFAR':
        model_outer = MobileNetV2ForCIFAR(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'PreActResNet50':
        model_outer = PreActResNet50(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'PreActResNet20':
        model_outer = PreActResNet20(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'CNNForSVHN':
        model_outer = CNNForSVHN(
            q_method,
            [q_bits_cand_list[0]] * q_bits_ws, [q_bits_cand_list[0]] * q_bits_as,
            False,
            num_classes=num_classes, 
            pretrained=pretrained).to(device)
    if q_method == 'LSQ':
        with torch.no_grad():
            temp_imgs, _ = next(iter(temp_loader))
            temp_imgs = temp_imgs.to(device)
            _ = model_outer(temp_imgs)

    optimizer_outer = outer_optim(model_outer.parameters(), **outer_optim_kwargs)

    if outer_lr_sch == 'Warmup-StepwiseCosine':
        total_iters = (last_epoch + epochs) * steps_per_epoch
        warmup_iters = 5 * steps_per_epoch
        last_iters = last_epoch * steps_per_epoch
        lr_dict = {}
        bs_ratio = 256 / batch_size
        for i in range(warmup_iters):
            if i >= last_iters:
                lr_dict[i - last_iters] = (1 - bs_ratio) / warmup_iters * i + bs_ratio
        for i in range(warmup_iters, total_iters):
            if i >= last_iters:            
                lr_dict[i - last_iters] = (1.0 + math.cos((i - warmup_iters) * math.pi / (total_iters - warmup_iters))) / 2
        lr_lambda = lambda itr: lr_dict[itr]
        lr_scheduler_outer = optim.lr_scheduler.LambdaLR(optimizer_outer, lr_lambda=lr_lambda)    # Neither to save nor to load state
    else:
        lr_scheduler_outer = outer_lr_sch(optimizer_outer, **outer_lr_sch_kwargs)

    if last_epoch != 0:
        model_outer.load_state_dict(
            torch.load(f'./{phase1_scheme_lowercase}/checkpoints/model_outer_e{last_epoch}.pth'),
            strict=True)
        optimizer_outer.load_state_dict(
            torch.load(f'./{phase1_scheme_lowercase}/checkpoints/optimizer_outer_e{last_epoch}.pth'))
        if outer_lr_sch != 'Warmup-StepwiseCosine':
            lr_scheduler_outer.load_state_dict(
                torch.load(f'./{phase1_scheme_lowercase}/checkpoints/lr_scheduler_outer_e{last_epoch}.pth'))
        temp_text = ', and lr scheduler' if outer_lr_sch != 'Warmup-StepwiseCosine' else ''
        log(f'Successfully loaded outer-loop model, optimizer{temp_text} with {last_epoch}-th phase 1 epoch.\n')
    ##

    if model_arch not in q_bits_wsas_dict: 
        q_bits_ws, q_bits_as = 0, 0
        for module in model_outer.modules():
            if type(module) == WeightQuantizer:
                q_bits_ws += 1
            if type(module) == ActivationQuantizer:
                q_bits_as += 1
        print(f'\nModel architecture: {model_arch}, q_bits_ws: {q_bits_ws}, q_bits_as: {q_bits_as}\n')

    best_avg_acc_val = last_best_avg_acc_val

    for epoch in range(last_epoch + 1, last_epoch + 1 + epochs):    # Outer-loop
        for step, (imgs, labels) in enumerate(train_loader, start=1):
            sum_outer_grad_dict = {}
            sum_outer_cnt_dict = {}
            optimizer_outer.zero_grad()

            imgs, labels = imgs.to(device), labels.to(device)

            for t in range(1, inner_q_subtasks + 1):                # Inner-loop
                if (q_bits_wa_guaranteed_list_given is not None) and (t in range(1, len(q_bits_wa_guaranteed_list_given) + 1)):
                    q_bits_w_list = [q_bits_wa_guaranteed_list_given[t - 1][0]] * (q_bits_ws - 2)
                    q_bits_a_list = [q_bits_wa_guaranteed_list_given[t - 1][1]] * (q_bits_as - 1)                    
                else:
                    while True:
                        b = random.randint(0, len(q_bits_cand_list) - 1)
                        q_bits_w = q_bits_cand_list[b]

                        b = random.randint(0, len(q_bits_cand_list) - 1)
                        q_bits_a = q_bits_cand_list[b]

                        reselect_a = any([
                            (q_bits_w is None) and (q_bits_a is not None),
                            (q_bits_w == 1) and (q_bits_a not in [1, None]),
                            (q_bits_w != 1) and (q_bits_a == 1)])

                        if reselect_a: continue
                        break                    
                    q_bits_w_list = [q_bits_w] * (q_bits_ws - 2)
                    q_bits_a_list = [q_bits_a] * (q_bits_as - 1)
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

                if (not distill_knowledge) or full_precision:       # No distillation for this inner-loop
                    ## Inner-loop model
                    if model_arch == 'ResNet18':
                        model_inner = ResNet18(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'ResNet50':
                        model_inner = ResNet50(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'MobileNetV2ForImageNet':
                        model_inner = MobileNetV2ForImageNet(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'MobileNetV2ForCIFAR':
                        model_inner = MobileNetV2ForCIFAR(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'PreActResNet50':
                        model_inner = PreActResNet50(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'PreActResNet20':
                        model_inner = PreActResNet20(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'CNNForSVHN':
                        model_inner = CNNForSVHN(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes, 
                            pretrained=pretrained).to(device)
                    if q_method == 'LSQ':
                        with torch.no_grad():
                            _ = model_inner(temp_imgs)

                    model_inner.load_state_dict(model_outer.state_dict(), strict=False)
                    ##

                    ## Forward
                    model_inner.zero_grad()
                    outputs = model_inner(imgs)
                    loss = loss_func(outputs, labels)

                    with torch.no_grad():
                        _, preds = torch.max(outputs.data, 1)
                        total_examples = labels.size(0)
                        correct_examples = (preds == labels).sum().item()

                    outer_grad_group = autograd.grad(loss, model_inner.parameters())
                    outer_grad_dict = {}
                    outer_cnt_dict = {}
                    for i, (key, _) in enumerate(model_inner.named_parameters()):
                        if outer_grad_group[i] is not None:
                            outer_grad_dict[key] = outer_grad_group[i]
                            outer_cnt_dict[key] = 1.0
                    ##
                else:                                               # Distillation for this inner-loop
                    ## Inner-loop teacher model
                    if model_arch == 'ResNet18':
                        model_inner = ResNet18(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'ResNet50':
                        model_inner = ResNet50(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'MobileNetV2ForImageNet':
                        model_inner = MobileNetV2ForImageNet(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'MobileNetV2ForCIFAR':
                        model_inner = MobileNetV2ForCIFAR(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'PreActResNet50':
                        model_inner = PreActResNet50(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'PreActResNet20':
                        model_inner = PreActResNet20(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'CNNForSVHN':
                        model_inner = CNNForSVHN(
                            q_method,
                            [None] * q_bits_ws, [None] * q_bits_as,
                            False,
                            num_classes=num_classes, 
                            pretrained=pretrained).to(device)
                    if q_method == 'LSQ':
                        with torch.no_grad():
                            _ = model_inner(temp_imgs)

                    model_inner.load_state_dict(model_outer.state_dict(), strict=False)
                    ##

                    ## Forward
                    model_inner.zero_grad()
                    outputs = model_inner(imgs)

                    targets_soft = F.softmax(outputs.detach(), dim=1)
                    ##

                    ## Inner-loop student model
                    if model_arch == 'ResNet18':
                        model_inner = ResNet18(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'ResNet50':
                        model_inner = ResNet50(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'MobileNetV2ForImageNet':
                        model_inner = MobileNetV2ForImageNet(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'MobileNetV2ForCIFAR':
                        model_inner = MobileNetV2ForCIFAR(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'PreActResNet50':
                        model_inner = PreActResNet50(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'PreActResNet20':
                        model_inner = PreActResNet20(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes,
                            pretrained=pretrained).to(device)
                    elif model_arch == 'CNNForSVHN':
                        model_inner = CNNForSVHN(
                            q_method,
                            q_bits_w_list, q_bits_a_list,
                            False,
                            num_classes=num_classes, 
                            pretrained=pretrained).to(device)
                    if q_method == 'LSQ':
                        with torch.no_grad():
                            _ = model_inner(temp_imgs)

                    model_inner.load_state_dict(model_outer.state_dict(), strict=False)
                    ##

                    ## Forward
                    model_inner.zero_grad()
                    outputs = model_inner(imgs)
                    loss = loss_func(outputs, labels)
                    loss += distill_loss_func(outputs, targets_soft)

                    with torch.no_grad():
                        _, preds = torch.max(outputs.data, 1)
                        total_examples = labels.size(0)
                        correct_examples = (preds == labels).sum().item()

                    outer_grad_group = autograd.grad(loss, model_inner.parameters())
                    outer_grad_dict = {}
                    outer_cnt_dict = {}
                    for i, (key, _) in enumerate(model_inner.named_parameters()):
                        if outer_grad_group[i] is not None:
                            outer_grad_dict[key] = outer_grad_group[i]
                            outer_cnt_dict[key] = 1.0
                    ##

                if t == 1:
                    sum_loss = loss.item()
                    sum_acc = 100 * correct_examples / float(total_examples)

                    for key, param in model_inner.named_parameters():
                        if key in outer_grad_dict:
                            sum_outer_grad_dict[key] = outer_grad_dict[key]
                            sum_outer_cnt_dict[key] = outer_cnt_dict[key]
                else:
                    sum_loss += loss.item()
                    sum_acc += 100 * correct_examples / float(total_examples)

                    for key, param in model_inner.named_parameters():
                        if key in outer_grad_dict:
                            if key in sum_outer_grad_dict:
                                sum_outer_grad_dict[key] += outer_grad_dict[key]
                                sum_outer_cnt_dict[key] += outer_cnt_dict[key]
                            else:
                                sum_outer_grad_dict[key] = outer_grad_dict[key]
                                sum_outer_cnt_dict[key] = outer_cnt_dict[key]

            if (step % report_period == 1) or (step == steps_per_epoch):
                avg_loss = sum_loss / float(inner_q_subtasks)
                avg_acc = sum_acc / float(inner_q_subtasks)
                log(f'\n  Phase 1 | {epoch}-th epoch, {step}-th step')
                log(f'  Training | Avg over {inner_q_subtasks} tasks | loss: {avg_loss:.3f}, accuracy: {avg_acc:.2f}%')

            for key, param in model_outer.named_parameters():
                if key in sum_outer_grad_dict:
                    param.grad = torch.clamp(
                        sum_outer_grad_dict[key] / float(sum_outer_cnt_dict[key]),
                        min=-10.0, max=10.0)

            optimizer_outer.step()

            if outer_lr_sch == 'Warmup-StepwiseCosine':
                lr_scheduler_outer.step()

        if outer_lr_sch != 'Warmup-StepwiseCosine':
            lr_scheduler_outer.step()

        if epoch % save_period == 0:
            torch.save(model_outer.state_dict(), f'./{phase1_scheme_lowercase}/checkpoints/model_outer_e{epoch}.pth')
            torch.save(optimizer_outer.state_dict(), f'./{phase1_scheme_lowercase}/checkpoints/optimizer_outer_e{epoch}.pth')
            if outer_lr_sch != 'Warmup-StepwiseCosine':
                torch.save(lr_scheduler_outer.state_dict(), f'./{phase1_scheme_lowercase}/checkpoints/lr_scheduler_outer_e{epoch}.pth')            
            temp_text = ', and lr scheduler' if outer_lr_sch != 'Warmup-StepwiseCosine' else ''
            log(f'Successfully saved outer-loop model, optimizer{temp_text} with {epoch}-th phase 1 epoch.')

        ## Validation per epoch
        for t, (imgs, labels) in enumerate(val_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)

            while True:
                b = random.randint(0, len(q_bits_cand_list) - 1)
                q_bits_w = q_bits_cand_list[b]

                b = random.randint(0, len(q_bits_cand_list) - 1)
                q_bits_a = q_bits_cand_list[b]

                reselect_a = any([
                    (q_bits_w is None) and (q_bits_a is not None),
                    (q_bits_w == 1) and (q_bits_a not in [1, None]),
                    (q_bits_w != 1) and (q_bits_a == 1)])

                if reselect_a: continue
                break                    
            q_bits_w_list = [q_bits_w] * (q_bits_ws - 2)
            q_bits_a_list = [q_bits_a] * (q_bits_as - 1)
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

            ## Validation model
            if model_arch == 'ResNet18':
                model_inner = ResNet18(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'ResNet50':
                model_inner = ResNet50(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'MobileNetV2ForImageNet':
                model_inner = MobileNetV2ForImageNet(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'MobileNetV2ForCIFAR':
                model_inner = MobileNetV2ForCIFAR(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'PreActResNet50':
                model_inner = PreActResNet50(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'PreActResNet20':
                model_inner = PreActResNet20(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'CNNForSVHN':
                model_inner = CNNForSVHN(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    False,
                    num_classes=num_classes, 
                    pretrained=pretrained).to(device)
            if q_method == 'LSQ':
                with torch.no_grad():
                    _ = model_inner(temp_imgs)

            model_inner.load_state_dict(model_outer.state_dict(), strict=False)
            ##

            ## Forward
            model_inner.eval()

            with torch.no_grad():
                outputs = model_inner(imgs)
                loss = loss_func(outputs, labels)

                _, preds = torch.max(outputs.data, 1)
                total_examples = labels.size(0)
                correct_examples = (preds == labels).sum().item()
            loss_val = loss.item() 
            acc_val = 100 * correct_examples / float(total_examples)
            ##

            if t == 1:
                sum_loss_val = loss_val
                sum_acc_val = acc_val
            else:
                sum_loss_val += loss_val
                sum_acc_val += acc_val

        avg_loss_val = sum_loss_val / float(val_tasks_per_epoch)
        avg_acc_val = sum_acc_val / float(val_tasks_per_epoch)
        log(f'\n  Phase 2 | {epoch}-th epoch')
        log(f'  Validation | Avg over {val_tasks_per_epoch} tasks | loss: {avg_loss_val:.3f}, accuracy: {avg_acc_val:.2f}%')

        if best_avg_acc_val < avg_acc_val:
            best_avg_acc_val = avg_acc_val
            log('  Achieved best validation accuracy.\n')
        else:
            log('\n')
        ##

    log('\n*****************************************************************************************')
    print(f'{now}\n')

def main():
    if PHASE1_SCHEME == 'MEBQAT-NonFewShot-MultiGPU':
        mebqat_nonfewshot_multigpu(
            WORKERS,
            DATASET, DATASET_PATH,
            MODEL_ARCH, PRETRAINED,
            Q_METHOD,
            Q_BITS_W_FIRST,
            Q_BITS_W_LAST, Q_BITS_A_LAST,
            LAST_EPOCH,
            LAST_BEST_AVG_ACC_VAL,
            EPOCHS,
            BATCH_SIZE,
            OUTER_OPTIM, OUTER_OPTIM_KWARGS,
            OUTER_LR_SCH, OUTER_LR_SCH_KWARGS,
            INNER_Q_SUBTASKS,
            Q_BITS_WA_GUARANTEED_LIST_GIVEN,
            DISTILL_KNOWLEDGE,
            SAVE_PERIOD, REPORT_PERIOD)
    elif PHASE1_SCHEME == 'MEBQAT-NonFewShot':
        mebqat_nonfewshot(
            WORKERS,
            DATASET, DATASET_PATH,
            MODEL_ARCH, PRETRAINED,
            Q_METHOD,
            Q_BITS_W_FIRST,
            Q_BITS_W_LAST, Q_BITS_A_LAST,
            LAST_EPOCH,
            LAST_BEST_AVG_ACC_VAL,
            EPOCHS,
            BATCH_SIZE,
            OUTER_OPTIM, OUTER_OPTIM_KWARGS,
            OUTER_LR_SCH, OUTER_LR_SCH_KWARGS,
            INNER_Q_SUBTASKS,
            Q_BITS_WA_GUARANTEED_LIST_GIVEN,
            DISTILL_KNOWLEDGE,
            SAVE_PERIOD, REPORT_PERIOD)

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

    main()
