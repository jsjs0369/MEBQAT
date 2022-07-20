import csv
import datetime
import numpy as np
import os
import pprint
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torchmeta
from torch.utils.data import DataLoader
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from collections import OrderedDict
from tensorboardX import SummaryWriter
# From own code(s)
from arch import BatchNorm2d, BatchNorm2dOnlyBeta, MAMLConvNet, ProtoConvNet, resnet, mobilenet_v2
from arg_phase2 import *
from util import create_log_func, plot_prediction, rdelattr, rsetparam
from loss import prototypical_loss

# FOMAML
def qa_finetune_and_infer(
    workers,
    dataset,
    dataset_path,
    net_arch,
    quant_scheme,
    quant_scheme_kwargs,
    qb_w_first, qb_a_first,
    qb_w_last, qb_a_last,
    phase1_scheme,
    last_epoch,
    cl_subtasks,
    seed_for_cl_subtasks,
    qb_subtasks,
    inter_qb_tuple_list_list_given,
    seed_for_remaining_qb_subtasks,
    n_way,
    k_shot_ft,
    k_shot_ifr,
    ft_optim,
    ft_optim_kwargs,
    ft_updates):
    ## Prepare to report
    middle_dir = 'qa-ft-and-ifr'

    # Make directory if not exist
    if not os.path.exists(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/'):
        os.makedirs(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/')

    # Add logger
    log_path = f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    # Add Tensorboard summary writer
    tb_writer = SummaryWriter(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/{now}_tb')

    # Add CSV writer
    f = open(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/{now}.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    if ft_updates == 0:
        csv_writer.writerow(['test_acc_ifr'])
    else:
        csv_writer.writerow(['test_acc_ft_u1', 'test_acc_ifr'])

    # Add txt writer
    f2 = open(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/{now}_qb_subtasks.txt', 'a')
    ## Prepare to report, end

    log('******************************          Arguments          ******************************')
    log(f'Time: {now}')

    log(f'# of workers: {workers}')
    log(f'Dataset: {dataset}')
    log(f'Path to dataset: {dataset_path}')

    log(f'Network model architecture: {net_arch}')
    log('Transductive setting (i.e., no tracking of running statistics in BN layer)')

    log(f'QAT/quantization scheme: {quant_scheme}')
    log(f'QAT/quantization scheme keyword arguments: {quant_scheme_kwargs}')
    log(f'For the first layer, use {qb_w_first}-bit weight and {qb_a_first}-bit activation')
    log(f'For the last layer, use {qb_w_last}-bit weight and {qb_a_last}-bit activation')

    log(f'Used phase 1 scheme: {phase1_scheme}')
    log(f'Last phase 1 epoch: {last_epoch}')

    log('Phase 2~3: (QA-)fine-tuning & inference (after quantization)')
    log(f'# of classification subtasks: {cl_subtasks}')
    log(f'Random seed for classification subtasks: {seed_for_cl_subtasks}') 
    log(f'# of quantization bitwidth subtasks: {qb_subtasks}')
    log(f'Used quantization bitwidth subtasks: see {now}_qb_subtasks.txt')
    log(f'Random seed for remaining quantization bitwidth subtasks: {seed_for_remaining_qb_subtasks}')
    log(f'# of classes: {n_way}') 
    log(f'{k_shot_ft} for phase 2 mini-batch, {k_shot_ifr} for phase 3 mini-batch') 
    log(f'Phase2 optimizer: {ft_optim}, {ft_optim_kwargs}')
    log(f'# of updates per test task in phase 2: {ft_updates}')

    log('\n******************************        Phase 2 ~ 3        ******************************')
    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    ## Prepare dataset
    # Numbers
    test_tasks = int(cl_subtasks * qb_subtasks)
    log(f'# of test tasks: {cl_subtasks} * {qb_subtasks} = {test_tasks}\n')

    # Data augmentation & sets
    if dataset == 'omniglot':
        # Test dataset
        test_set = omniglot(
            dataset_path, 
            k_shot_ft, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_ifr,
            meta_split='test',
            download=True)
    elif dataset == 'miniimagenet':
        # Test dataset
        test_set = miniimagenet(
            dataset_path, 
            k_shot_ft, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_ifr,
            meta_split='test',
            download=True)
    else:
        raise Exception(f'Not supported dataset called {dataset}')

    # Data loaders
    test_loader = BatchMetaDataLoader(
        test_set,
        batch_size=1, 
        shuffle=True, 
        num_workers=workers,
        pin_memory=True, 
        drop_last=True)
    ## Prepare dataset, end

    # Configure model architecture, referring to dataset when needed
    if net_arch == 'maml-conv-net':
        inter_qb_tuples = MAMLConvNet.inter_qb_tuples

        if dataset == 'omniglot':
            in_channels = 1
            hidden_channels = 64
            classifier_in_features = 64
        elif dataset == 'miniimagenet':
            in_channels = 3
            hidden_channels = 32
            classifier_in_features = 32*5*5
    else:
        raise Exception(f'Not supported model architecture called {net_arch}')

    ## Prepare classification subtasks
    # Set random seed
    if seed_for_cl_subtasks is not None:
        torch.manual_seed(seed_for_cl_subtasks)
        torch.cuda.manual_seed(seed_for_cl_subtasks)
        torch.cuda.manual_seed_all(seed_for_cl_subtasks) # if use multi-GPU
        np.random.seed(seed_for_cl_subtasks)
        random.seed(seed_for_cl_subtasks)

    # cl_subtask_list has length cl_subtasks
    cl_subtask_list = []
    for clst, meta_batch in enumerate(test_loader, start=1):
        x_ft, y_ft = meta_batch['train']
        x_ifr, y_ifr = meta_batch['test'] 

        imgs_ft, labels_ft = x_ft[0], y_ft[0]
        imgs_ifr, labels_ifr = x_ifr[0], y_ifr[0]

        cl_subtask_list.append((imgs_ft, labels_ft, imgs_ifr, labels_ifr))

        if clst >= cl_subtasks:
            break

    # Reset random seed
    if seed_for_cl_subtasks is not None:
        torch.seed()
        torch.cuda.seed()
        torch.cuda.seed_all()
        np.random.seed()
        random.seed()
    ## Prepare classification subtasks, end

    ## Prepare quantization bitwidth subtasks
    # inter_qb_tuple_list_list_given has length M where 0 <= M <= qb_subtasks
    # Each element of inter_qb_tuple_list_list_given (i.e., a quantization bitwidth setting except first & last layers) has length inter_qb_tuples
    assert 0 <= len(inter_qb_tuple_list_list_given) <= qb_subtasks

    qb_subtask_list = inter_qb_tuple_list_list_given

    # If required, fill remaining w/ randomly sampled quantization bitwidth settings
    if len(inter_qb_tuple_list_list_given) < qb_subtasks:
        if quant_scheme in ['lsq', 'dorefa']:
            qb_tuple_cand_list = [
                (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 16), (2, 32),
                (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 16), (3, 32),
                (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 16), (4, 32),
                (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 16), (5, 32),
                (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 16), (6, 32),
                (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 16), (7, 32),
                (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 16), (8, 32),
                (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8), (16, 16), (16, 32),
                (32, 32)]
        else:
            raise Exception(f'Not supported QAT/quantization scheme called {quant_scheme}')

        for _ in range(qb_subtasks - len(inter_qb_tuple_list_list_given)):
            inter_qb_tuple_list = []
            for _ in range(inter_qb_tuples):
                if seed_for_remaining_qb_subtasks is not None:
                    random.seed(seed_for_remaining_qb_subtasks)     # Set random seed
                b = random.randint(1, len(qb_tuple_cand_list))
                inter_qb_tuple_list.append(qb_tuple_cand_list[b-1])
                if seed_for_remaining_qb_subtasks is not None:
                    seed_for_remaining_qb_subtasks += 100             # Increase random seed
            qb_subtask_list.append(inter_qb_tuple_list)

        # Reset random seed
        if seed_for_remaining_qb_subtasks is not None:
            random.seed()

    assert len(qb_subtask_list) == qb_subtasks

    for qbst, inter_qb_tuple_list in enumerate(qb_subtask_list, start=1):
        f2.write(f'-------------- Quantization subtask {qbst} --------------\n')

        qb_w_first_temp, qb_a_first_temp = qb_w_first, qb_a_first
        qb_w_last_temp, qb_a_last_temp = qb_w_last, qb_a_last

        if qb_w_first == 'same':
            qb_w_first_temp = inter_qb_tuple_list[0][0]
        if qb_a_first == 'same':
            qb_a_first_temp = inter_qb_tuple_list[0][1]
        if qb_w_last == 'same':
            qb_w_last_temp = inter_qb_tuple_list[-1][0]
        if qb_a_last == 'same':
            qb_a_last_temp = inter_qb_tuple_list[-1][1]

        full_precision = True
        for (qb_w, qb_a) in inter_qb_tuple_list:
            if (qb_w < 32) or (qb_a < 32):
                full_precision = False
                break

        if full_precision:
            qb_w_first_temp, qb_a_first_temp = 32, 32
            qb_w_last_temp, qb_a_last_temp = 32, 32

        f2.write(f'[({qb_w_first_temp}, {qb_a_first_temp}), {", ".join(map(str, inter_qb_tuple_list))}, ({qb_w_last_temp}, {qb_a_last_temp})]\n\n')            
    f2.close()
    ## Prepare quantization bitwidth subtasks, end

    # Main test
    for t in range(1, test_tasks + 1):
        # Select classification subtask
        remainder = int((t - 1) % cl_subtasks) 
        imgs_ft, labels_ft, imgs_ifr, labels_ifr = cl_subtask_list[remainder]
        imgs_ft, labels_ft, imgs_ifr, labels_ifr = \
            imgs_ft.to(device), labels_ft.to(device), imgs_ifr.to(device), labels_ifr.to(device)

        # Select quantization subtask
        quotient = int((t - 1) // cl_subtasks) 
        inter_qb_tuple_list = qb_subtask_list[quotient]

        ## Prepare fine-tuning (model, optimizer)
        if net_arch == 'maml-conv-net':
            net = MAMLConvNet(
                quant_scheme,
                quant_scheme_kwargs,
                qb_w_first, qb_a_first,
                inter_qb_tuple_list, 
                qb_w_last, qb_a_last,
                in_channels,
                hidden_channels,
                classifier_in_features,
                n_way,
                track_running_stats=False).to(device)

        # Prepare some things first
        if t == 1:
            bn_param_key_list = []
            for key, m in net.named_modules():
                if type(m) in [BatchNorm2d, BatchNorm2dOnlyBeta]:
                    bn_param_key_list.append(key + '.bias')
            if last_epoch != 0:
                base_state_dict = torch.load(f'./{phase1_scheme}/checkpoints/net_base_e{last_epoch}.pth')

        scope_dict = OrderedDict(net.named_parameters())
        for key in list(scope_dict.keys()):
            if key in bn_param_key_list:
                scope_dict.pop(key)
        optimizer = ft_optim(list(scope_dict.values()), **ft_optim_kwargs)
    
        if last_epoch != 0:
            if quant_scheme == 'lsq':
                for key, param in base_state_dict.items():
                    if 'scale_w' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    elif 'scale_a' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                net.load_state_dict(base_state_dict, strict=False)
            elif quant_scheme == 'dorefa':
                net.load_state_dict(base_state_dict, strict=False)
        ## Prepare fine-tuning, end

        ## Fine-tuning
        for u in range(1, ft_updates + 1):
            optimizer.zero_grad()
            outputs_ft = net(imgs_ft)
            loss_ft = loss_func(outputs_ft, labels_ft)

            if u == 1:
                with torch.no_grad():
                    _, test_preds_ft_u1 = torch.max(outputs_ft.data, 1)

                    total_examples = labels_ft.size(0)
                    correct_examples = (test_preds_ft_u1 == labels_ft).sum().item()
                    test_acc_ft_u1 = 100 * correct_examples / float(total_examples)
            
            loss_ft.backward()
            optimizer.step()
        ## Fine-tuning, end

        ## Inference
        net.quant_w()
        net.eval()

        with torch.no_grad():
            outputs_ifr = net(imgs_ifr)

            _, test_preds_ifr = torch.max(outputs_ifr.data, 1)

            total_examples = labels_ifr.size(0)
            correct_examples = (test_preds_ifr == labels_ifr).sum().item()  
            test_acc_ifr = 100 * correct_examples / float(total_examples)
        ## Inference, end

        ## Sum up this test task
        if t == 1:
            log(f'  (QA-)Fine-tuning & inference (after quantization) (phase 2~3)')
        else:
            log('')

        if ft_updates == 0:
            log(f'  Inference (after quantization) | {t}-th test task | accuracy: {test_acc_ifr:.2f}%')

            tb_writer.add_scalar('test_acc_ifr', test_acc_ifr, t)

            """
            tb_writer.add_figure(
                f'prediction_vs_label/ifr_t{t}',
                plot_prediction(imgs_ifr, test_preds_ifr, labels_ifr, n_way, k_shot_ifr))
            """

            csv_writer.writerow([test_acc_ifr])
        else:
            log(f'  (QA-)Fine-tuning, 1st iteration | {t}-th test task | accuracy: {test_acc_ft_u1:.2f}%')

            log(f'  Inference (after quantization) | {t}-th test task | accuracy: {test_acc_ifr:.2f}%')

            tb_writer.add_scalar('test_acc_ft_u1', test_acc_ft_u1, t)

            tb_writer.add_scalar('test_acc_ifr', test_acc_ifr, t)

            """
            tb_writer.add_figure(
                f'prediction_vs_label/ft_u1_t{t}',
                plot_prediction(imgs_ft, test_preds_ft_u1, labels_ft, n_way, k_shot_ft))

            tb_writer.add_figure(
                f'prediction_vs_label/ifr_t{t}',
                plot_prediction(imgs_ifr, test_preds_ifr, labels_ifr, n_way, k_shot_ifr))
            """

            csv_writer.writerow([test_acc_ft_u1, test_acc_ifr])
        ## Sum up this test task, end      

    log('\n*****************************************************************************************')
    print(f'{now}\n')   # Print to check
    tb_writer.close()
    f.close()

# Prototypical Networks
def qa_prototype_and_infer(
    workers,
    dataset,
    dataset_path,
    net_arch,
    quant_scheme,
    quant_scheme_kwargs,
    qb_w_first, qb_a_first,
    phase1_scheme,
    last_epoch,
    cl_subtasks,
    seed_for_cl_subtasks,
    qb_subtasks,
    inter_qb_tuple_list_list_given,
    seed_for_remaining_qb_subtasks,
    n_way,
    k_shot_sup,
    k_shot_qry,
    track_running_stats):
    ## Prepare to report
    middle_dir = 'qa-prototype-and-infer'

    # Make directory if not exist
    if not os.path.exists(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/'):
        os.makedirs(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/')

    # Add logger
    log_path = f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    # Add Tensorboard summary writer
    tb_writer = SummaryWriter(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/{now}_tb')

    # Add CSV writer
    f = open(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/{now}.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['test_acc_qry'])

    # Add txt writer
    f2 = open(f'./{phase1_scheme}/reports/phase2and3/{middle_dir}/{now}_qb_subtasks.txt', 'a')
    ## Prepare to report, end

    log('******************************          Arguments          ******************************')
    log(f'Time: {now}')

    log(f'# of workers: {workers}')
    log(f'Dataset: {dataset}')
    log(f'Path to dataset: {dataset_path}')

    log(f'Network model architecture: {net_arch}')
    if track_running_stats:
        log('Inductive setting (i.e., tracking running statistics in BN layer)')
    else:
        log('Transductive setting (i.e., no tracking of running statistics in BN layer)')

    log(f'QAT/quantization scheme: {quant_scheme}')
    log(f'QAT/quantization scheme keyword arguments: {quant_scheme_kwargs}')
    log(f'For the first layer, use {qb_w_first}-bit weight and {qb_a_first}-bit activation')

    log(f'Used phase 1 scheme: {phase1_scheme}')
    log(f'Last phase 1 epoch: {last_epoch}')

    log('Phase 2~3: (QA-)Prototype calculation & inference (after quantization)')
    log(f'# of classification subtasks: {cl_subtasks}')
    log(f'Random seed for classification subtasks: {seed_for_cl_subtasks}') 
    log(f'# of quantization bitwidth subtasks: {qb_subtasks}')
    log(f'Used quantization bitwidth subtasks: see {now}_qb_subtasks.txt')
    log(f'Random seed for remaining quantization bitwidth subtasks: {seed_for_remaining_qb_subtasks}')
    log(f'# of classes: {n_way}') 
    log(f'{k_shot_sup} for phase 2 mini-batch, {k_shot_qry} for phase 3 mini-batch') 

    log('\n******************************        Phase 2 ~ 3        ******************************')
    # Define loss function
    loss_func = prototypical_loss

    ## Prepare dataset
    # Numbers
    test_tasks = int(cl_subtasks * qb_subtasks)
    log(f'# of test tasks: {cl_subtasks} * {qb_subtasks} = {test_tasks}\n')

    # Data augmentation & sets
    if dataset == 'omniglot':
        # Test dataset
        test_set = omniglot(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            meta_split='test',
            download=True)
    elif dataset == 'miniimagenet':
        # Test dataset
        test_set = miniimagenet(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            meta_split='test',
            download=True)
    else:
        raise Exception(f'Not supported dataset called {dataset}')

    # Data loaders
    test_loader = BatchMetaDataLoader(
        test_set,
        batch_size=1, 
        shuffle=True, 
        num_workers=workers,
        pin_memory=True, 
        drop_last=True)
    ## Prepare dataset, end

    # Configure model architecture, referring to dataset when needed
    if net_arch == 'proto-conv-net':
        inter_qb_tuples = ProtoConvNet.inter_qb_tuples

        if dataset == 'omniglot':
            in_channels = 1
        elif dataset == 'miniimagenet':
            in_channels = 3
    else:
        raise Exception(f'Not supported model architecture called {net_arch}')

    ## Prepare classification subtasks
    # Set random seed
    if seed_for_cl_subtasks is not None:
        torch.manual_seed(seed_for_cl_subtasks)
        torch.cuda.manual_seed(seed_for_cl_subtasks)
        torch.cuda.manual_seed_all(seed_for_cl_subtasks) # if use multi-GPU
        np.random.seed(seed_for_cl_subtasks)
        random.seed(seed_for_cl_subtasks)

    # cl_subtask_list has length cl_subtasks
    cl_subtask_list = []
    for clst, meta_batch in enumerate(test_loader, start=1):
        x_sup, _ = meta_batch['train']
        x_qry, _ = meta_batch['test'] 

        imgs_sup = x_sup[0]
        imgs_qry = x_qry[0]

        cl_subtask_list.append((imgs_sup, imgs_qry))

        if clst >= cl_subtasks:
            break

    # Reset random seed
    if seed_for_cl_subtasks is not None:
        torch.seed()
        torch.cuda.seed()
        torch.cuda.seed_all()
        np.random.seed()
        random.seed()
    ## Prepare classification subtasks, end

    ## Prepare quantization bitwidth subtasks
    # inter_qb_tuple_list_list_given has length M where 0 <= M <= qb_subtasks
    # Each element of inter_qb_tuple_list_list_given (i.e., a quantization bitwidth setting except first layer) has length inter_qb_tuples
    assert 0 <= len(inter_qb_tuple_list_list_given) <= qb_subtasks

    qb_subtask_list = inter_qb_tuple_list_list_given

    # If required, fill remaining w/ randomly sampled quantization bitwidth settings
    if len(inter_qb_tuple_list_list_given) < qb_subtasks:
        if quant_scheme in ['lsq', 'dorefa']:
            qb_tuple_cand_list = [
                (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 16), (2, 32),
                (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 16), (3, 32),
                (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 16), (4, 32),
                (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 16), (5, 32),
                (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 16), (6, 32),
                (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 16), (7, 32),
                (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 16), (8, 32),
                (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 8), (16, 16), (16, 32),
                (32, 32)]
        else:
            raise Exception(f'Not supported QAT/quantization scheme called {quant_scheme}')

        for _ in range(qb_subtasks - len(inter_qb_tuple_list_list_given)):
            inter_qb_tuple_list = []
            for _ in range(inter_qb_tuples):
                if seed_for_remaining_qb_subtasks is not None:
                    random.seed(seed_for_remaining_qb_subtasks)     # Set random seed
                b = random.randint(1, len(qb_tuple_cand_list))
                inter_qb_tuple_list.append(qb_tuple_cand_list[b-1])
                if seed_for_remaining_qb_subtasks is not None:
                    seed_for_remaining_qb_subtasks += 100             # Increase random seed
            qb_subtask_list.append(inter_qb_tuple_list)

        # Reset random seed
        if seed_for_remaining_qb_subtasks is not None:
            random.seed()

    assert len(qb_subtask_list) == qb_subtasks

    for qbst, inter_qb_tuple_list in enumerate(qb_subtask_list, start=1):
        f2.write(f'-------------- Quantization subtask {qbst} --------------\n')

        qb_w_first_temp, qb_a_first_temp = qb_w_first, qb_a_first

        if qb_w_first == 'same':
            qb_w_first_temp = inter_qb_tuple_list[0][0]
        if qb_a_first == 'same':
            qb_a_first_temp = inter_qb_tuple_list[0][1]

        full_precision = True
        for (qb_w, qb_a) in inter_qb_tuple_list:
            if (qb_w < 32) or (qb_a < 32):
                full_precision = False
                break

        if full_precision:
            qb_w_first_temp, qb_a_first_temp = 32, 32

        f2.write(f'[({qb_w_first_temp}, {qb_a_first_temp}), {", ".join(map(str, inter_qb_tuple_list))}]\n\n')            
    f2.close()
    ## Prepare quantization bitwidth subtasks, end

    # Main test
    for t in range(1, test_tasks + 1):
        # Select classification subtask
        remainder = int((t - 1) % cl_subtasks) 
        imgs_sup, imgs_qry = cl_subtask_list[remainder]

        x = torch.cat(
            [imgs_sup, imgs_qry], 
            0).to(device)

        # Select quantization subtask
        quotient = int((t - 1) // cl_subtasks) 
        inter_qb_tuple_list = qb_subtask_list[quotient]

        if net_arch == 'proto-conv-net':
            net = ProtoConvNet(
                quant_scheme,
                quant_scheme_kwargs,
                qb_w_first, qb_a_first,
                inter_qb_tuple_list, 
                in_channels,
                track_running_stats=track_running_stats).to(device)

        if last_epoch != 0:
            base_state_dict = torch.load(f'./{phase1_scheme}/checkpoints/net_base_e{last_epoch}.pth')
            if quant_scheme == 'lsq':
                for key, param in base_state_dict.items():
                    if 'scale_w' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    elif 'scale_a' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                net.load_state_dict(base_state_dict, strict=False)
            elif quant_scheme == 'dorefa':
                net.load_state_dict(base_state_dict, strict=False)
        net.quant_w()
        net.eval()

        with torch.no_grad():
            outputs = net(x)
            _, test_acc_qry = loss_func(outputs, n_way, k_shot_sup, k_shot_qry)
        test_acc_qry *= 100

        if t == 1:
            log(f'\n  (QA-)Prototype calculation & inference (phase 2~3)')
        else:
            log('')

        log(f'  Inference | {t}-th test task | accuracy: {test_acc_qry:.2f}%')

        tb_writer.add_scalar('test_acc_qry', test_acc_qry, t)

        csv_writer.writerow([test_acc_qry.item()])

    log('\n*****************************************************************************************')
    print(f'{now}\n')   # Print to check
    tb_writer.close()
    f.close()

def main():
    if PHASE2AND3_CONFIG == 'qa-ft-and-ifr':
        qa_finetune_and_infer(
            WORKERS,
            DATASET,
            DATASET_PATH,
            NET_ARCH,
            QUANT_SCHEME,
            QUANT_SCHEME_KWARGS,
            QB_W_FIRST, QB_A_FIRST,
            QB_W_LAST, QB_A_LAST,
            PHASE1_SCHEME,
            LAST_EPOCH,
            CL_SUBTASKS,
            SEED_FOR_CL_SUBTASKS,
            QB_SUBTASKS,
            INTER_QB_TUPLE_LIST_LIST_GIVEN,
            SEED_FOR_REMAINING_QB_SUBTASKS,
            N_WAY,
            K_SHOT_FT,
            K_SHOT_IFR,
            FT_OPTIM,
            FT_OPTIM_KWARGS,
            FT_UPDATES)
    elif PHASE2AND3_CONFIG == 'qa-prototype-and-infer':
        qa_prototype_and_infer(
            WORKERS,
            DATASET,
            DATASET_PATH,
            NET_ARCH,
            QUANT_SCHEME,
            QUANT_SCHEME_KWARGS,
            QB_W_FIRST, QB_A_FIRST,
            PHASE1_SCHEME,
            LAST_EPOCH,
            CL_SUBTASKS,
            SEED_FOR_CL_SUBTASKS,
            QB_SUBTASKS,
            INTER_QB_TUPLE_LIST_LIST_GIVEN,
            SEED_FOR_REMAINING_QB_SUBTASKS,
            N_WAY,
            K_SHOT_SUP,
            K_SHOT_QRY,
            TRACK_RUNNING_STATS)
    else:
        raise Exception(f'Not supported phase2~3 configuration called {PHASE2AND3_CONFIG}')

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

    # Check if PHASE1_SCHEME is compatible to LAST_EPOCH
    assert (PHASE1_SCHEME is None) == (LAST_EPOCH == 0)

    main()
