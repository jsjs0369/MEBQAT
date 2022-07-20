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

def qat_nonfewshot(
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
    optim, optim_kwargs,
    lr_sch, lr_sch_kwargs,
    q_bits_w_inter_list, q_bits_a_inter_list,
    save_period, report_period):
    phase1_scheme_lowercase = 'qat_nonfewshot'
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
    log(f'Model architecture: {model_arch}, inductive setting, whether to start with a pretrained model: {pretrained}')
    log(f'Method of quantization and/or QAT: {q_method}')
    log(f'For the first layer, use {q_bits_w_first}-bit weight')
    log(f'For the last layer, use {q_bits_w_last}-bit weight and {q_bits_a_last}-bit activation')     
    log(f'Last phase 1 epoch: {last_epoch}')
    log(f'Last best validation accuracy: {last_best_avg_acc_val}')
    log(f'Phase 1 epochs: {epochs}')
    log(f'Batch size: {batch_size}')
    log(f'Optimizer: {optim}, with keyword arguments {optim_kwargs}')
    log(f'Learning rate scheduler: {lr_sch}, with keyword arguments {lr_sch_kwargs}')
    log(f'Target weight bitwidth list: {q_bits_w_inter_list}, target activation bitwidth list: {q_bits_a_inter_list}')
    log(f'Save period: {save_period} epoch(s), report period: {report_period} step(s)')
    log('Not mixed precision')

    log('\n******************************           Phase 1           ******************************')
    # Loss function
    loss_func = nn.CrossEntropyLoss().to(device)

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

    q_bits_w_list, q_bits_a_list = q_bits_w_inter_list, q_bits_a_inter_list
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

    ## Base model, optimizer, and lr scheduler
    if model_arch == 'ResNet18':
        model_base = ResNet18(
            q_method,
            q_bits_w_list, q_bits_a_list,
            True,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'ResNet50':
        model_base = ResNet50(
            q_method,
            q_bits_w_list, q_bits_a_list,
            True,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'MobileNetV2ForImageNet':
        model_base = MobileNetV2ForImageNet(
            q_method,
            q_bits_w_list, q_bits_a_list,
            True,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'MobileNetV2ForCIFAR':
        model_base = MobileNetV2ForCIFAR(
            q_method,
            q_bits_w_list, q_bits_a_list,
            True,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'PreActResNet50':
        model_base = PreActResNet50(
            q_method,
            q_bits_w_list, q_bits_a_list,
            True,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'PreActResNet20':
        model_base = PreActResNet20(
            q_method,
            q_bits_w_list, q_bits_a_list,
            True,
            num_classes=num_classes,
            pretrained=pretrained).to(device)
    elif model_arch == 'CNNForSVHN':
        model_base = CNNForSVHN(
            q_method,
            q_bits_w_list, q_bits_a_list,
            True,
            num_classes=num_classes, 
            pretrained=pretrained).to(device)
    if q_method == 'LSQ':
        with torch.no_grad():
            temp_imgs, _ = next(iter(temp_loader))
            temp_imgs = temp_imgs.to(device)
            _ = model_base(temp_imgs)

    optimizer_base = optim(model_base.parameters(), **optim_kwargs)

    if lr_sch == 'Warmup-StepwiseCosine':
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
        lr_scheduler_base = optim.lr_scheduler.LambdaLR(optimizer_base, lr_lambda=lr_lambda)    # Neither to save nor to load state
    else:
        lr_scheduler_base = lr_sch(optimizer_base, **lr_sch_kwargs)

    if last_epoch != 0:
        model_base.load_state_dict(
            torch.load(f'./{phase1_scheme_lowercase}/checkpoints/model_base_e{last_epoch}.pth'),
            strict=True)
        optimizer_base.load_state_dict(
            torch.load(f'./{phase1_scheme_lowercase}/checkpoints/optimizer_base_e{last_epoch}.pth'))
        if lr_sch != 'Warmup-StepwiseCosine':
            lr_scheduler_base.load_state_dict(
                torch.load(f'./{phase1_scheme_lowercase}/checkpoints/lr_scheduler_base_e{last_epoch}.pth'))
        temp_text = ', and lr scheduler' if lr_sch != 'Warmup-StepwiseCosine' else ''
        log(f'Successfully loaded base model, optimizer{temp_text} with {last_epoch}-th phase 1 epoch.\n')
    ##

    best_avg_acc_val = last_best_avg_acc_val

    for epoch in range(last_epoch + 1, last_epoch + 1 + epochs):
        for step, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer_base.zero_grad()

            if model_arch == 'ResNet18':
                model = ResNet18(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'ResNet50':
                model = ResNet50(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'MobileNetV2ForImageNet':
                model = MobileNetV2ForImageNet(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'MobileNetV2ForCIFAR':
                model = MobileNetV2ForCIFAR(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'PreActResNet50':
                model = PreActResNet50(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'PreActResNet20':
                model = PreActResNet20(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'CNNForSVHN':
                model = CNNForSVHN(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes, 
                    pretrained=pretrained).to(device)
            if q_method == 'LSQ':
                with torch.no_grad():
                    _ = model(temp_imgs)

            model.load_state_dict(model_base.state_dict(), strict=True)

            model.zero_grad()

            outputs = model(imgs)
            loss = loss_func(outputs, labels)

            with torch.no_grad():
                _, preds = torch.max(outputs.data, 1)
                total_examples = labels.size(0)
                correct_examples = (preds == labels).sum().item()
                acc = 100 * correct_examples / float(total_examples)

            base_grad_group = autograd.grad(loss, model.parameters())
            base_grad_dict = {}
            for i, (key, _) in enumerate(model.named_parameters()):
                if base_grad_group[i] is not None:
                    base_grad_dict[key] = base_grad_group[i]

            for key, m in model_base.named_modules():
                if type(m) == BatchNorm2d:
                    model_bn_layer = rgetattr(model, key)
                    m.running_mean = model_bn_layer.running_mean
                    m.running_var = model_bn_layer.running_var
                    m.num_batches_tracked = model_bn_layer.num_batches_tracked

            for key, param in model_base.named_parameters():
                if key in base_grad_dict:
                    param.grad = torch.clamp(
                        base_grad_dict[key],
                        min=-10.0, max=10.0)

            optimizer_base.step()

            if (step % report_period == 1) or (step == steps_per_epoch):
                log(f'\n  Phase 1 | {epoch}-th epoch, {step}-th step')
                log(f'  Training | specific-precision task | loss: {loss.item():.3f}, accuracy: {acc:.2f}%')

            if step >= steps_per_epoch:
                break

            if lr_sch == 'Warmup-StepwiseCosine':
                lr_scheduler_base.step()

        if lr_sch != 'Warmup-StepwiseCosine':
            lr_scheduler_base.step()

        if epoch % save_period == 0:
            torch.save(model_base.state_dict(), f'./{phase1_scheme_lowercase}/checkpoints/model_base_e{epoch}.pth')
            torch.save(optimizer_base.state_dict(), f'./{phase1_scheme_lowercase}/checkpoints/optimizer_base_e{epoch}.pth')
            if lr_sch != 'Warmup-StepwiseCosine':
                torch.save(lr_scheduler_base.state_dict(), f'./{phase1_scheme_lowercase}/checkpoints/lr_scheduler_base_e{epoch}.pth')            
            temp_text = ', and lr scheduler' if lr_sch != 'Warmup-StepwiseCosine' else ''
            log(f'Successfully saved base model, optimizer{temp_text} with {epoch}-th phase 1 epoch.')

        ## Validation per epoch
        sum_loss_val = 0.0
        sum_acc_val = 0.0

        for t, (imgs, labels) in enumerate(val_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)

            ## Validation model
            if model_arch == 'ResNet18':
                model = ResNet18(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'ResNet50':
                model = ResNet50(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'MobileNetV2ForImageNet':
                model = MobileNetV2ForImageNet(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'MobileNetV2ForCIFAR':
                model = MobileNetV2ForCIFAR(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'PreActResNet50':
                model = PreActResNet50(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'PreActResNet20':
                model = PreActResNet20(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes,
                    pretrained=pretrained).to(device)
            elif model_arch == 'CNNForSVHN':
                model = CNNForSVHN(
                    q_method,
                    q_bits_w_list, q_bits_a_list,
                    True,
                    num_classes=num_classes, 
                    pretrained=pretrained).to(device)
            if q_method == 'LSQ':
                with torch.no_grad():
                    _ = model(temp_imgs)

            model.load_state_dict(model_base.state_dict(), strict=True)
            ##

            ## Forward
            model.eval()

            with torch.no_grad():
                outputs = model(imgs)
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

            if t >= val_tasks_per_epoch:
                break

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
    if PHASE1_SCHEME == 'QAT-NonFewShot':
        qat_nonfewshot(
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
            OPTIM, OPTIM_KWARGS,
            LR_SCH, LR_SCH_KWARGS,
            Q_BITS_W_INTER_LIST, Q_BITS_A_INTER_LIST,
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
