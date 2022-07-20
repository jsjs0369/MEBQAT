import datetime
import math
import numpy as np
import os
import pprint
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torchmeta
import torchvision
from torch.utils.data import DataLoader
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from collections import OrderedDict
from tensorboardX import SummaryWriter
# From own code(s)
from arch import BatchNorm2d, BatchNorm2dOnlyBeta, MAMLConvNet, ProtoConvNet, resnet, mobilenet_v2
from arg_phase1 import *
from util import create_log_func, plot_prediction, rgetattr, rdelattr, rsetparam
from loss import prototypical_loss

def qat_fomaml(
    workers,
    dataset,  
    dataset_path,
    net_arch,
    quant_scheme,
    quant_scheme_kwargs,
    qb_w_first, qb_a_first,
    qb_w_last, qb_a_last,
    last_epoch,
    last_best_avg_val_acc_ifr,
    epochs,
    inter_qb_tuple_list_given,
    n_way,
    k_shot_sup,
    k_shot_qry,
    outer_optim,
    outer_optim_kwargs,
    inner_cl_subtasks,
    inner_optim,
    inner_optim_kwargs, 
    inner_updates,
    val_updates,
    save_period):
    ## Prepare to report
    middle_dir = 'qat-fomaml'

    # Make directories if not exist
    if not os.path.exists(f'./{middle_dir}/reports/phase1/'):
        os.makedirs(f'./{middle_dir}/reports/phase1/')
    if not os.path.exists(f'./{middle_dir}/checkpoints/'):
        os.makedirs(f'./{middle_dir}/checkpoints/')

    # Add logger
    log_path = f'./{middle_dir}/reports/phase1/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    # Add Tensorboard summary writer
    tb_writer = SummaryWriter(f'./{middle_dir}/reports/phase1/{now}_tb')
    ## Prepare to report, end

    ## Clarify fixed quantization bitwidth setting for intermediate conv/FC layers
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

    if net_arch == 'maml-conv-net':
        inter_qb_tuples = MAMLConvNet.inter_qb_tuples
    else:
        raise Exception(f'Not supported model architecture called {net_arch}')

    if inter_qb_tuple_list_given is not None:
        inter_qb_tuple_list = inter_qb_tuple_list_given
    else:
        inter_qb_tuple_list = []
        for _ in range(inter_qb_tuples):
            b = random.randint(1, len(qb_tuple_cand_list))
            inter_qb_tuple_list.append(qb_tuple_cand_list[b-1])
    ## Clarify fixed quantization bitwidth setting for intermediate conv/FC layers, end

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

    log('Phase 1: QAT + FOMAML')
    log(f'Last phase1 epoch: {last_epoch}')
    log(f'Last best validation accuracy: {last_best_avg_val_acc_ifr}')
    log(f'Phase1 epochs: {epochs}')
    log(f'For intermediate layers, use {inter_qb_tuple_list}')
    log(f'# of classes: {n_way}')
    log(f'{k_shot_sup} for support mini-batch, {k_shot_qry} for query mini-batch')
    log(f'Outer-loop optimizer: {outer_optim}, {outer_optim_kwargs}')
    log(f'# of training classification subtasks (i.e., meta batch size): {inner_cl_subtasks}')  
    log(f'Inner-loop optimizer: {inner_optim}, {inner_optim_kwargs}')
    log(f'# of updates per inner-loop: {inner_updates}')
    log(f'# of updates in validation: {val_updates}')
    log(f'Save period: {save_period} epoch(s)')

    log('\n******************************           Phase 1           ******************************')
    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    ## Prepare dataset
    # Numbers
    steps_per_epoch = 800
    log(f'# of outer-loop (step)s per epoch: {steps_per_epoch}\n')
    val_tasks_per_epoch = 200
    log(f'# of val tasks per epoch: {val_tasks_per_epoch}\n')

    # Data augmentation & sets
    if dataset == 'omniglot':
        # Train dataset
        train_set = omniglot(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            seed=None,
            meta_split='train',
            download=True)
        # Validation dataset
        val_set = omniglot(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            seed=None,
            meta_split='val',
            download=True)
    elif dataset == 'miniimagenet':
        # Train dataset
        train_set = miniimagenet(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            seed=None, 
            meta_split='train',
            download=True)
        # Validation dataset
        val_set = miniimagenet(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            seed=None, 
            meta_split='val',
            download=True)
    else:
        raise Exception(f'Not supported dataset called {dataset}')

    # Data loaders
    train_loader = BatchMetaDataLoader(
        train_set, 
        batch_size=inner_cl_subtasks, 
        shuffle=True, 
        num_workers=workers,
        pin_memory=True, 
        drop_last=True)
    if quant_scheme == 'lsq':
        temp_loader = BatchMetaDataLoader(
            train_set, 
            batch_size=1,
            shuffle=True)
    val_loader = BatchMetaDataLoader(
        val_set, 
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        pin_memory=True, 
        drop_last=True)
    ## Prepare dataset, end

    # Configure model architecture, referring to dataset when needed
    if net_arch == 'maml-conv-net':
        if dataset == 'omniglot':
            in_channels = 1
            hidden_channels = 64
            classifier_in_features = 64
        elif dataset == 'miniimagenet':
            in_channels = 3
            hidden_channels = 32
            classifier_in_features = 32*5*5

    ## Prepare outer-loop (base model, base optimizer) 
    if net_arch == 'maml-conv-net':
        net_base = MAMLConvNet(
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

    optimizer_base = outer_optim(net_base.parameters(), **outer_optim_kwargs)

    if last_epoch == 0:
        log('Starting from scratch.\n')
        if quant_scheme == 'lsq':
            with torch.no_grad():
                meta_batch = next(iter(temp_loader))
                x_temp, _ = meta_batch['train']
                imgs_temp = x_temp[0].to(device)
                _ = net_base(imgs_temp)
    else:
        base_state_dict = torch.load(f'./{middle_dir}/checkpoints/net_base_e{last_epoch}.pth')
        if quant_scheme == 'lsq':
            for key, param in base_state_dict.items():
                if 'scale_w' in key:
                    rdelattr(net_base, key)
                    rsetparam(net_base, key, nn.Parameter(param.clone().detach()))
                elif 'scale_a' in key:
                    rdelattr(net_base, key)
                    rsetparam(net_base, key, nn.Parameter(param.clone().detach()))
            net_base.load_state_dict(base_state_dict, strict=False)
        elif quant_scheme == 'dorefa':
            net_base.load_state_dict(base_state_dict, strict=True)

        optimizer_base.load_state_dict(
            torch.load(f'./{middle_dir}/checkpoints/optimizer_base_e{last_epoch}.pth')) 

        log(f'Successfully loaded base-model and base-optimizer with {last_epoch}-th phase1 epoch.\n')
    ## Prepare outer-loop, end

    # Prepare to freeze BN layers in all inner-loops
    bn_param_key_list = []
    for key, m in net_base.named_modules():
        if type(m) in [BatchNorm2d, BatchNorm2dOnlyBeta]:
            bn_param_key_list.append(key + '.bias')

    # Prepare to track best validation accuracy
    best_avg_val_acc_ifr = last_best_avg_val_acc_ifr

    # Main training & validation
    for epoch in range(last_epoch + 1, last_epoch + 1 + epochs):
        # Outer-loop
        for step, meta_batch in enumerate(train_loader, start=1):
            x_sup, y_sup = meta_batch['train']
            x_qry, y_qry = meta_batch['test']

            sum_base_grad_dict = {}
            optimizer_base.zero_grad()  

            # Inner-loop
            for clst in range(1, inner_cl_subtasks + 1):
                # Prepare data
                try:
                    imgs_sup, labels_sup = x_sup[clst-1].to(device), y_sup[clst-1].to(device)
                    imgs_qry, labels_qry = x_qry[clst-1].to(device), y_qry[clst-1].to(device)  
                except:
                    log(f'Skipped {epoch}-th epoch -> {step}-th step -> {clst}-th classification subtask')
                    continue

                ## Prepare inner-loop (model, optimizer)
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

                scope_dict = OrderedDict(net.named_parameters())
                for key in list(scope_dict.keys()):
                    if key in bn_param_key_list:
                        scope_dict.pop(key)
                optimizer = inner_optim(list(scope_dict.values()), **inner_optim_kwargs)

                if quant_scheme == 'lsq':
                    for key, param in net_base.named_parameters():
                        if 'scale_w' in key:
                            if key in dict(net.named_parameters()).keys():
                                rdelattr(net, key)
                                rsetparam(net, key, nn.Parameter(param.clone().detach()))
                        elif 'scale_a' in key:
                            if key in dict(net.named_parameters()).keys():
                                rdelattr(net, key)
                                rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    net.load_state_dict(net_base.state_dict(), strict=False)
                elif quant_scheme == 'dorefa':
                    net.load_state_dict(net_base.state_dict(), strict=True)                    
                ## Prepare inner-loop, end

                ## Support
                for u in range(1, inner_updates + 1):
                    optimizer.zero_grad()
                    outputs_sup = net(imgs_sup)
                    loss_sup = loss_func(outputs_sup, labels_sup)

                    if u == 1:
                        with torch.no_grad():
                            loss_sup_u1 = loss_sup.item()
                            _, preds_sup_u1 = torch.max(outputs_sup.data, 1)

                            total_examples = labels_sup.size(0)
                            correct_examples = (preds_sup_u1 == labels_sup).sum().item()
                            acc_sup_u1 = 100 * correct_examples / float(total_examples)

                    loss_sup.backward()
                    optimizer.step()

                with torch.no_grad():
                    outputs_sup = net(imgs_sup)
                    loss_sup = loss_func(outputs_sup, labels_sup)

                    loss_sup_last = loss_sup.item()
                    _, preds_sup_last = torch.max(outputs_sup.data, 1)

                    total_examples = labels_sup.size(0)
                    correct_examples = (preds_sup_last == labels_sup).sum().item()
                    acc_sup_last = 100 * correct_examples / float(total_examples)
                ## Support, end

                ## Query
                net.zero_grad() 
                outputs_qry = net(imgs_qry) 
                loss_qry = loss_func(outputs_qry, labels_qry)   

                with torch.no_grad():
                    _, preds_qry = torch.max(outputs_qry.data, 1)

                    total_examples = labels_qry.size(0)
                    correct_examples = (preds_qry == labels_qry).sum().item() 
                    acc_qry = 100 * correct_examples / float(total_examples)
                ## Query, end

                ## Sum up this inner-loop
                base_grad_group = autograd.grad(loss_qry, net.parameters())
                base_grad_dict = {}
                for i, (key, _) in enumerate(net.named_parameters()):
                    if base_grad_group[i] is not None:
                        base_grad_dict[key] = base_grad_group[i]

                if clst == 1:
                    sum_loss_sup_u1 = loss_sup_u1
                    sum_acc_sup_u1 = acc_sup_u1

                    sum_loss_sup_last = loss_sup_last
                    sum_acc_sup_last = acc_sup_last

                    sum_loss_qry = loss_qry.item()
                    sum_acc_qry = acc_qry

                    for key, param in net.named_parameters():
                        if key in base_grad_dict:
                            sum_base_grad_dict[key] = base_grad_dict[key]
                else:
                    sum_loss_sup_u1 += loss_sup_u1
                    sum_acc_sup_u1 += acc_sup_u1

                    sum_loss_sup_last += loss_sup_last
                    sum_acc_sup_last += acc_sup_last 

                    sum_loss_qry += loss_qry.item()
                    sum_acc_qry += acc_qry

                    for key, param in net.named_parameters():
                        if key in base_grad_dict:
                            if key in sum_base_grad_dict:
                                sum_base_grad_dict[key] += base_grad_dict[key]
                            else:
                                sum_base_grad_dict[key] = base_grad_dict[key]
                ## Sum up this inner-loop, end

            ## Sum up this outer-loop
            avg_loss_sup_u1 = sum_loss_sup_u1 / float(inner_cl_subtasks)
            avg_acc_sup_u1 = sum_acc_sup_u1 / float(inner_cl_subtasks)            

            avg_loss_sup_last = sum_loss_sup_last / float(inner_cl_subtasks)
            avg_acc_sup_last = sum_acc_sup_last / float(inner_cl_subtasks)          

            avg_loss_qry = sum_loss_qry / float(inner_cl_subtasks)
            avg_acc_qry = sum_acc_qry / float(inner_cl_subtasks) 

            for key, param in net_base.named_parameters():
                if key in sum_base_grad_dict:
                    param.grad = torch.clamp(
                        sum_base_grad_dict[key] / float(inner_cl_subtasks),
                        min=-10.0, max=10.0)

            optimizer_base.step()

            if step == steps_per_epoch:
                global_step = (epoch - 1) * steps_per_epoch + step

                log(f'\n  QAT + FOMAML (phase 1) | {epoch}-th epoch, {step}-th step')

                log(f'  Support, 1st iteration | Avg over tasks | loss: {avg_loss_sup_u1:.3f}, accuracy: {avg_acc_sup_u1:.2f}%')

                log(f'  Support, last iteration | Avg over tasks | loss: {avg_loss_sup_last:.3f}, accuracy: {avg_acc_sup_last:.2f}%')

                log(f'  Query | Avg over tasks | loss: {avg_loss_qry:.3f}, accuracy: {avg_acc_qry:.2f}%')

                tb_writer.add_scalar('avg_loss_sup_u1', avg_loss_sup_u1, global_step)
                tb_writer.add_scalar('avg_acc_sup_u1', avg_acc_sup_u1, global_step)

                tb_writer.add_scalar('avg_loss_sup_last', avg_loss_sup_last, global_step)
                tb_writer.add_scalar('avg_acc_sup_last', avg_acc_sup_last, global_step)

                tb_writer.add_scalar('avg_loss_qry', avg_loss_qry, global_step)
                tb_writer.add_scalar('avg_acc_qry', avg_acc_qry, global_step)

                """
                tb_writer.add_figure(
                    'prediction_vs_label/sup_u1_last-t',
                    plot_prediction(imgs_sup, preds_sup_u1, labels_sup, n_way, k_shot_sup),
                    global_step=global_step)

                tb_writer.add_figure(
                    'prediction_vs_label/sup_last_last-t',
                    plot_prediction(imgs_sup, preds_sup_last, labels_sup, n_way, k_shot_sup),
                    global_step=global_step)

                tb_writer.add_figure(
                    'prediction_vs_label/qry_last-t',
                    plot_prediction(imgs_qry, preds_qry, labels_qry, n_way, k_shot_qry),
                    global_step=global_step)
                """
            ## Sum up this outer-loop, end

            if step >= steps_per_epoch:
                break

        ## Validation on every epoch
        ft_optim, ft_optim_kwargs = inner_optim, inner_optim_kwargs
        ft_updates = val_updates
        val_acc_ft_u1_list, val_acc_ifr_list = [], []

        for t, meta_batch in enumerate(val_loader, start=1):
            x_ft, y_ft = meta_batch['train']
            x_ifr, y_ifr = meta_batch['test'] 

            imgs_ft, labels_ft = x_ft[0].to(device), y_ft[0].to(device)
            imgs_ifr, labels_ifr = x_ifr[0].to(device), y_ifr[0].to(device)

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

            scope_dict = OrderedDict(net.named_parameters())
            for key in list(scope_dict.keys()):
                if key in bn_param_key_list:
                    scope_dict.pop(key)
            optimizer = ft_optim(list(scope_dict.values()), **ft_optim_kwargs)

            if quant_scheme == 'lsq':
                for key, param in net_base.named_parameters():
                    if 'scale_w' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    elif 'scale_a' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                net.load_state_dict(net_base.state_dict(), strict=False)  
            elif quant_scheme == 'dorefa':
                net.load_state_dict(net_base.state_dict(), strict=True)
            ## Prepare fine-tuning, end

            ## Fine-tuning
            for u in range(1, ft_updates + 1):
                optimizer.zero_grad()
                outputs_ft = net(imgs_ft)
                loss_ft = loss_func(outputs_ft, labels_ft)

                if u == 1:
                    with torch.no_grad():
                        _, val_preds_ft_u1 = torch.max(outputs_ft.data, 1)

                        total_examples = labels_ft.size(0)
                        correct_examples = (val_preds_ft_u1 == labels_ft).sum().item()
                        val_acc_ft_u1 = 100 * correct_examples / float(total_examples)
                
                loss_ft.backward()
                optimizer.step()
            ## Fine-tuning, end

            ## Inference
            net.quant_w()
            net.eval()

            with torch.no_grad():
                outputs_ifr = net(imgs_ifr)

                _, val_preds_ifr = torch.max(outputs_ifr.data, 1)

                total_examples = labels_ifr.size(0)
                correct_examples = (val_preds_ifr == labels_ifr).sum().item()  
                val_acc_ifr = 100 * correct_examples / float(total_examples)

            val_acc_ft_u1_list.append(val_acc_ft_u1)
            val_acc_ifr_list.append(val_acc_ifr)
            ## Inference, end

            if t >= val_tasks_per_epoch:
                break

        ## Sum up this validation
        avg_val_acc_ft_u1 = np.mean(np.array(val_acc_ft_u1_list))
        avg_val_acc_ifr = np.mean(np.array(val_acc_ifr_list))

        log(f'\n  QA-fine-tuning & inference (phase 2~3) | {epoch}-th epoch')

        log(f'  QA-fine-tuning, 1st iteration | Avg over val tasks | accuracy: {avg_val_acc_ft_u1:.2f}%')

        log(f'  Inference after quantization | Avg over val tasks | accuracy: {avg_val_acc_ifr:.2f}%')
        if best_avg_val_acc_ifr < avg_val_acc_ifr:
            best_avg_val_acc_ifr = avg_val_acc_ifr
            log(f'  Achieved best validation accuracy.')
        ## Sum up this validation, end

        # Save base model and base optimizer
        if epoch % save_period == 0:
            torch.save(net_base.state_dict(), f'./{middle_dir}/checkpoints/net_base_e{epoch}.pth')

            torch.save(optimizer_base.state_dict(), f'./{middle_dir}/checkpoints/optimizer_base_e{epoch}.pth')
      
            log(f'\nSuccessfully saved base-model and base-optimizer with {epoch}-th phase1 epoch.\n') 

    log('\n*****************************************************************************************')
    print(f'{now}\n')   # Print to check
    tb_writer.close()

def mebqat_maml(
    workers,
    dataset,    
    dataset_path,
    net_arch,
    quant_scheme,
    quant_scheme_kwargs,
    qb_w_first, qb_a_first,
    qb_w_last, qb_a_last,
    last_epoch,
    last_best_avg_val_acc_ifr,
    epochs,
    inter_uniform,
    n_way,
    k_shot_sup,
    k_shot_qry,
    outer_optim,
    outer_optim_kwargs,
    inner_cl_subtasks,
    inner_optim,
    inner_optim_kwargs, 
    inner_updates,
    val_updates,
    save_period):
    ## Prepare to report
    middle_dir = 'mebqat-maml'

    # Make directories if not exist
    if not os.path.exists(f'./{middle_dir}/reports/phase1/'):
        os.makedirs(f'./{middle_dir}/reports/phase1/')
    if not os.path.exists(f'./{middle_dir}/checkpoints/'):
        os.makedirs(f'./{middle_dir}/checkpoints/')

    # Add logger
    log_path = f'./{middle_dir}/reports/phase1/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    # Add Tensorboard summary writer
    tb_writer = SummaryWriter(f'./{middle_dir}/reports/phase1/{now}_tb')
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

    log('Phase 1: MEBQAT-MAML')
    log(f'Last phase1 epoch: {last_epoch}')
    log(f'Last best validation accuracy: {last_best_avg_val_acc_ifr}')
    log(f'Phase1 epochs: {epochs}')
    log(f'Whether to use a uniform quantization bitwidth setting for intermediate layers in each model: {inter_uniform}')
    log(f'# of classes: {n_way}')
    log(f'{k_shot_sup} for support mini-batch, {k_shot_qry} for query mini-batch')
    log(f'Outer-loop optimizer: {outer_optim}, {outer_optim_kwargs}')
    log(f'# of training classification subtasks (i.e., meta batch size): {inner_cl_subtasks}') 
    log(f'Inner-loop optimizer: {inner_optim}, {inner_optim_kwargs}')
    log(f'# of updates per inner-loop: {inner_updates}')
    log(f'# of updates in validation: {val_updates}')
    log(f'Save period: {save_period} epoch(s)')

    log('\n******************************           Phase 1           ******************************')
    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    # Prepare quantization
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

    ## Prepare dataset
    # Numbers
    steps_per_epoch = 800
    log(f'# of outer-loop (step)s per epoch: {steps_per_epoch}\n')
    val_tasks_per_epoch = 200
    log(f'# of val tasks per epoch: {val_tasks_per_epoch}\n')

    # Data augmentation & sets
    if dataset == 'omniglot':
        # Train dataset
        train_set = omniglot(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            seed=None,
            meta_split='train',
            download=True)
        # Validation dataset
        val_set = omniglot(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            seed=None,
            meta_split='val',
            download=True)
    elif dataset == 'miniimagenet':
        # Train dataset
        train_set = miniimagenet(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            seed=None, 
            meta_split='train',
            download=True)
        # Validation dataset
        val_set = miniimagenet(
            dataset_path, 
            k_shot_sup, 
            n_way, 
            shuffle=True, 
            test_shots=k_shot_qry,
            seed=None, 
            meta_split='val',
            download=True)
    else:
        raise Exception(f'Not supported dataset called {dataset}')

    # Data loaders
    train_loader = BatchMetaDataLoader(
        train_set, 
        batch_size=inner_cl_subtasks,
        shuffle=True,
        num_workers=workers,
        pin_memory=True, 
        drop_last=True)
    if quant_scheme == 'lsq':
        temp_loader = BatchMetaDataLoader(
            train_set, 
            batch_size=1,
            shuffle=True)
    val_loader = BatchMetaDataLoader(
        val_set, 
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

    ## Prepare outer-loop (base model, base optimizer) 
    if net_arch == 'maml-conv-net':
        net_base = MAMLConvNet(
            quant_scheme,
            quant_scheme_kwargs,
            qb_w_first, qb_a_first,
            [(2, 2)]*inter_qb_tuples,
            qb_w_last, qb_a_last,
            in_channels,
            hidden_channels,
            classifier_in_features,
            n_way,
            track_running_stats=False).to(device)

    optimizer_base = outer_optim(net_base.parameters(), **outer_optim_kwargs)

    if last_epoch == 0:
        log('Starting from scratch.\n')
        if quant_scheme == 'lsq':
            with torch.no_grad():
                meta_batch = next(iter(temp_loader))
                x_temp, _ = meta_batch['train']
                imgs_temp = x_temp[0].to(device)
                _ = net_base(imgs_temp)
    else:
        base_state_dict = torch.load(f'./{middle_dir}/checkpoints/net_base_e{last_epoch}.pth')
        if quant_scheme == 'lsq':
            for key, param in base_state_dict.items():
                if 'scale_w' in key:
                    rdelattr(net_base, key)
                    rsetparam(net_base, key, nn.Parameter(param.clone().detach()))
                elif 'scale_a' in key:
                    rdelattr(net_base, key)
                    rsetparam(net_base, key, nn.Parameter(param.clone().detach()))
            net_base.load_state_dict(base_state_dict, strict=False)
        elif quant_scheme == 'dorefa':
            net_base.load_state_dict(base_state_dict, strict=True)

        optimizer_base.load_state_dict(
            torch.load(f'./{middle_dir}/checkpoints/optimizer_base_e{last_epoch}.pth')) 

        log(f'Successfully loaded base-model and base-optimizer with {last_epoch}-th phase1 epoch.\n')
    ## Prepare outer-loop, end

    # Prepare to freeze BN layers in all inner-loops
    bn_param_key_list = []
    for key, m in net_base.named_modules():
        if type(m) in [BatchNorm2d, BatchNorm2dOnlyBeta]:
            bn_param_key_list.append(key + '.bias')

    # Prepare to track best validation accuracy
    best_avg_val_acc_ifr = last_best_avg_val_acc_ifr

    # Main training & validation
    for epoch in range(last_epoch + 1, last_epoch + 1 + epochs):
        # Outer-loop
        for step, meta_batch in enumerate(train_loader, start=1):
            x_sup, y_sup = meta_batch['train']
            x_qry, y_qry = meta_batch['test']

            sum_base_grad_dict = {}
            optimizer_base.zero_grad()  

            total_conv_fc_quant_cnt_list_w = [0] * (inter_qb_tuples + 2) 
            total_conv_fc_quant_cnt_list_a = [0] * (inter_qb_tuples + 2)

            # Inner-loop
            for clst in range(1, inner_cl_subtasks + 1):
                # Prepare data
                try:
                    imgs_sup, labels_sup = x_sup[clst-1].to(device), y_sup[clst-1].to(device)
                    imgs_qry, labels_qry = x_qry[clst-1].to(device), y_qry[clst-1].to(device)  
                except:
                    log(f'Skipped {epoch}-th epoch -> {step}-th step -> {clst}-th classification subtask')
                    continue

                # Randomly select quantization bitwidth setting
                if inter_uniform:
                    b = random.randint(1, len(qb_tuple_cand_list))
                    inter_qb_tuple_list = [qb_tuple_cand_list[b-1]] * inter_qb_tuples
                else:
                    inter_qb_tuple_list = []
                    for _ in range(inter_qb_tuples):
                        b = random.randint(1, len(qb_tuple_cand_list))
                        inter_qb_tuple_list.append(qb_tuple_cand_list[b-1])                

                ## Prepare inner-loop (model, optimizer)
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

                scope_dict = OrderedDict(net.named_parameters())
                for key in list(scope_dict.keys()):
                    if key in bn_param_key_list:
                        scope_dict.pop(key)
                optimizer = inner_optim(list(scope_dict.values()), **inner_optim_kwargs)

                if quant_scheme == 'lsq':
                    for key, param in net_base.named_parameters():
                        if 'scale_w' in key:
                            if key in dict(net.named_parameters()).keys():
                                rdelattr(net, key)
                                rsetparam(net, key, nn.Parameter(param.clone().detach()))
                        elif 'scale_a' in key:
                            if key in dict(net.named_parameters()).keys():
                                rdelattr(net, key)
                                rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    net.load_state_dict(net_base.state_dict(), strict=False)
                elif quant_scheme == 'dorefa':
                    net.load_state_dict(net_base.state_dict(), strict=False)
                ## Prepare inner-loop, end

                ## Support
                for u in range(1, inner_updates + 1):
                    optimizer.zero_grad()
                    outputs_sup = net(imgs_sup)
                    loss_sup = loss_func(outputs_sup, labels_sup)

                    if u == 1:
                        with torch.no_grad():
                            loss_sup_u1 = loss_sup.item()
                            _, preds_sup_u1 = torch.max(outputs_sup.data, 1)

                            total_examples = labels_sup.size(0)
                            correct_examples = (preds_sup_u1 == labels_sup).sum().item()
                            acc_sup_u1 = 100 * correct_examples / float(total_examples)

                    loss_sup.backward()
                    optimizer.step()

                with torch.no_grad():
                    outputs_sup = net(imgs_sup)
                    loss_sup = loss_func(outputs_sup, labels_sup)

                    loss_sup_last = loss_sup.item()
                    _, preds_sup_last = torch.max(outputs_sup.data, 1)

                    total_examples = labels_sup.size(0)
                    correct_examples = (preds_sup_last == labels_sup).sum().item()
                    acc_sup_last = 100 * correct_examples / float(total_examples)
                ## Support, end

                ## Query
                net.zero_grad() 
                outputs_qry = net(imgs_qry) 
                loss_qry = loss_func(outputs_qry, labels_qry)   

                with torch.no_grad():
                    _, preds_qry = torch.max(outputs_qry.data, 1)

                    total_examples = labels_qry.size(0)
                    correct_examples = (preds_qry == labels_qry).sum().item() 
                    acc_qry = 100 * correct_examples / float(total_examples)
                ## Query, end

                ## Sum up this inner-loop
                base_grad_group = autograd.grad(loss_qry, net.parameters())
                base_grad_dict = {}
                for i, (key, _) in enumerate(net.named_parameters()):
                    if base_grad_group[i] is not None:
                        base_grad_dict[key] = base_grad_group[i]

                if clst == 1:
                    sum_loss_sup_u1 = loss_sup_u1
                    sum_acc_sup_u1 = acc_sup_u1

                    sum_loss_sup_last = loss_sup_last
                    sum_acc_sup_last = acc_sup_last

                    sum_loss_qry = loss_qry.item()
                    sum_acc_qry = acc_qry

                    for key, param in net.named_parameters():
                        if key in base_grad_dict:
                            sum_base_grad_dict[key] = base_grad_dict[key]
                else:
                    sum_loss_sup_u1 += loss_sup_u1
                    sum_acc_sup_u1 += acc_sup_u1

                    sum_loss_sup_last += loss_sup_last
                    sum_acc_sup_last += acc_sup_last 

                    sum_loss_qry += loss_qry.item()
                    sum_acc_qry += acc_qry

                    for key, param in net.named_parameters():
                        if key in base_grad_dict:
                            if key in sum_base_grad_dict:
                                sum_base_grad_dict[key] += base_grad_dict[key]
                            else:
                                sum_base_grad_dict[key] = base_grad_dict[key]

                conv_fc_quant_cnt_list_w, conv_fc_quant_cnt_list_a = net.get_conv_fc_quant_cnt_lists()
                total_conv_fc_quant_cnt_list_w = [sum(i) for i in zip(total_conv_fc_quant_cnt_list_w, conv_fc_quant_cnt_list_w)]
                total_conv_fc_quant_cnt_list_a = [sum(i) for i in zip(total_conv_fc_quant_cnt_list_a, conv_fc_quant_cnt_list_a)] 
                ## Sum up this inner-loop, end

            ## Sum up this outer-loop
            avg_loss_sup_u1 = sum_loss_sup_u1 / float(inner_cl_subtasks)
            avg_acc_sup_u1 = sum_acc_sup_u1 / float(inner_cl_subtasks)            

            avg_loss_sup_last = sum_loss_sup_last / float(inner_cl_subtasks)
            avg_acc_sup_last = sum_acc_sup_last / float(inner_cl_subtasks)          

            avg_loss_qry = sum_loss_qry / float(inner_cl_subtasks)
            avg_acc_qry = sum_acc_qry / float(inner_cl_subtasks) 

            if quant_scheme == 'lsq':
                conv_fc_info_dict = net_base.get_conv_fc_info_dict()
                for key, param in net_base.named_parameters():
                    if key in sum_base_grad_dict:
                        if key.replace('.scale_w', '') in conv_fc_info_dict.keys():
                            divisor = total_conv_fc_quant_cnt_list_w[conv_fc_info_dict[key.replace('.scale_w', '')]]
                            assert divisor != 0
                        elif key.replace('.scale_a', '') in conv_fc_info_dict.keys():
                            divisor = total_conv_fc_quant_cnt_list_a[conv_fc_info_dict[key.replace('.scale_a', '')]]
                            assert divisor != 0
                        else:   # Not a quant param
                            divisor = inner_cl_subtasks

                        param.grad = torch.clamp(
                            sum_base_grad_dict[key] / float(divisor),
                            min=-10.0, max=10.0)
                    else:   # For checking validity
                        if key.replace('.scale_w', '') in conv_fc_info_dict.keys():
                            divisor = total_conv_fc_quant_cnt_list_w[conv_fc_info_dict[key.replace('.scale_w', '')]]
                            assert divisor == 0
                        elif key.replace('.scale_a', '') in conv_fc_info_dict.keys():
                            divisor = total_conv_fc_quant_cnt_list_a[conv_fc_info_dict[key.replace('.scale_a', '')]]
                            assert divisor == 0
            elif quant_scheme == 'dorefa':
                for key, param in net_base.named_parameters():
                    if key in sum_base_grad_dict:
                        param.grad = torch.clamp(
                            sum_base_grad_dict[key] / float(inner_cl_subtasks),
                            min=-10.0, max=10.0)

            optimizer_base.step()

            if step == steps_per_epoch:
                global_step = (epoch - 1) * steps_per_epoch + step

                log(f'\n  MEBQAT-MAML (phase 1) | {epoch}-th epoch, {step}-th step')

                log(f'  Support, 1st iteration | Avg over tasks | loss: {avg_loss_sup_u1:.3f}, accuracy: {avg_acc_sup_u1:.2f}%')

                log(f'  Support, last iteration | Avg over tasks | loss: {avg_loss_sup_last:.3f}, accuracy: {avg_acc_sup_last:.2f}%')

                log(f'  Query | Avg over tasks | loss: {avg_loss_qry:.3f}, accuracy: {avg_acc_qry:.2f}%')

                tb_writer.add_scalar('avg_loss_sup_u1', avg_loss_sup_u1, global_step)
                tb_writer.add_scalar('avg_acc_sup_u1', avg_acc_sup_u1, global_step)

                tb_writer.add_scalar('avg_loss_sup_last', avg_loss_sup_last, global_step)
                tb_writer.add_scalar('avg_acc_sup_last', avg_acc_sup_last, global_step)

                tb_writer.add_scalar('avg_loss_qry', avg_loss_qry, global_step)
                tb_writer.add_scalar('avg_acc_qry', avg_acc_qry, global_step)

                """
                tb_writer.add_figure(
                    'prediction_vs_label/sup_u1_last-t',
                    plot_prediction(imgs_sup, preds_sup_u1, labels_sup, n_way, k_shot_sup),
                    global_step=global_step)

                tb_writer.add_figure(
                    'prediction_vs_label/sup_last_last-t',
                    plot_prediction(imgs_sup, preds_sup_last, labels_sup, n_way, k_shot_sup),
                    global_step=global_step)

                tb_writer.add_figure(
                    'prediction_vs_label/qry_last-t',
                    plot_prediction(imgs_qry, preds_qry, labels_qry, n_way, k_shot_qry),
                    global_step=global_step)
                """
            ## Sum up this outer-loop, end
            
            if step >= steps_per_epoch:
                break

        ## Validation on every epoch
        ft_optim, ft_optim_kwargs = inner_optim, inner_optim_kwargs
        ft_updates = val_updates
        val_acc_ft_u1_list, val_acc_ifr_list = [], []

        for t, meta_batch in enumerate(val_loader, start=1):
            x_ft, y_ft = meta_batch['train']
            x_ifr, y_ifr = meta_batch['test'] 

            imgs_ft, labels_ft = x_ft[0].to(device), y_ft[0].to(device)
            imgs_ifr, labels_ifr = x_ifr[0].to(device), y_ifr[0].to(device)

            # Randomly select quantization bitwidth setting
            if inter_uniform:
                b = random.randint(1, len(qb_tuple_cand_list))
                inter_qb_tuple_list = [qb_tuple_cand_list[b-1]] * inter_qb_tuples
            else:
                inter_qb_tuple_list = []
                for _ in range(inter_qb_tuples):
                    b = random.randint(1, len(qb_tuple_cand_list))
                    inter_qb_tuple_list.append(qb_tuple_cand_list[b-1]) 

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

            scope_dict = OrderedDict(net.named_parameters())
            for key in list(scope_dict.keys()):
                if key in bn_param_key_list:
                    scope_dict.pop(key)
            optimizer = ft_optim(list(scope_dict.values()), **ft_optim_kwargs)
 
            if quant_scheme == 'lsq':
                for key, param in net_base.named_parameters():
                    if 'scale_w' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    elif 'scale_a' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                net.load_state_dict(net_base.state_dict(), strict=False)  
            elif quant_scheme == 'dorefa':
                net.load_state_dict(net_base.state_dict(), strict=False)          
            ## Prepare fine-tuning, end

            ## Fine-tuning
            for u in range(1, ft_updates + 1):
                optimizer.zero_grad()
                outputs_ft = net(imgs_ft)
                loss_ft = loss_func(outputs_ft, labels_ft)

                if u == 1:
                    with torch.no_grad():
                        _, val_preds_ft_u1 = torch.max(outputs_ft.data, 1)

                        total_examples = labels_ft.size(0)
                        correct_examples = (val_preds_ft_u1 == labels_ft).sum().item()
                        val_acc_ft_u1 = 100 * correct_examples / float(total_examples)
                
                loss_ft.backward()
                optimizer.step()
            ## Fine-tuning, end

            ## Inference
            net.quant_w()
            net.eval()

            with torch.no_grad():
                outputs_ifr = net(imgs_ifr)

                _, val_preds_ifr = torch.max(outputs_ifr.data, 1)

                total_examples = labels_ifr.size(0)
                correct_examples = (val_preds_ifr == labels_ifr).sum().item()  
                val_acc_ifr = 100 * correct_examples / float(total_examples)

            val_acc_ft_u1_list.append(val_acc_ft_u1)
            val_acc_ifr_list.append(val_acc_ifr)
            ## Inference, end

            if t >= val_tasks_per_epoch:
                break

        ## Sum up this validation
        avg_val_acc_ft_u1 = np.mean(np.array(val_acc_ft_u1_list))
        avg_val_acc_ifr = np.mean(np.array(val_acc_ifr_list))

        log(f'\n  (QA-)Fine-tuning & inference (phase 2~3) | {epoch}-th epoch')

        log(f'  (QA-)Fine-tuning, 1st iteration | Avg over val tasks | accuracy: {avg_val_acc_ft_u1:.2f}%')

        log(f'  Inference (after quantization) | Avg over val tasks | accuracy: {avg_val_acc_ifr:.2f}%')
        if best_avg_val_acc_ifr < avg_val_acc_ifr:
            best_avg_val_acc_ifr = avg_val_acc_ifr
            log(f'  Achieved best validation accuracy.')
        ## Sum up this validation, end

        # Save base model and base optimizer
        if epoch % save_period == 0:
            torch.save(net_base.state_dict(), f'./{middle_dir}/checkpoints/net_base_e{epoch}.pth')

            torch.save(optimizer_base.state_dict(), f'./{middle_dir}/checkpoints/optimizer_base_e{epoch}.pth')
      
            log(f'\nSuccessfully saved base-model and base-optimizer with {epoch}-th phase1 epoch.\n') 

    log('\n*****************************************************************************************')
    print(f'{now}\n')   # Print to check
    tb_writer.close()

def qat_pn(
    workers,
    dataset,
    dataset_path,
    net_arch,
    quant_scheme,
    quant_scheme_kwargs,
    qb_w_first, qb_a_first,
    last_epoch,
    last_best_avg_val_acc_qry,
    epochs,
    inter_qb_tuple_list_given,
    n_way_val,
    k_shot_sup,
    k_shot_qry_val,
    optim,
    optim_kwargs,
    lr_sch,
    lr_sch_kwargs,
    save_period):
    ## Prepare to report
    middle_dir = 'qat-pn'

    # Make directories if not exist
    if not os.path.exists(f'./{middle_dir}/reports/phase1/'):
        os.makedirs(f'./{middle_dir}/reports/phase1/')
    if not os.path.exists(f'./{middle_dir}/checkpoints/'):
        os.makedirs(f'./{middle_dir}/checkpoints/')

    # Add logger
    log_path = f'./{middle_dir}/reports/phase1/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    # Add Tensorboard summary writer
    tb_writer = SummaryWriter(f'./{middle_dir}/reports/phase1/{now}_tb')
    ## Prepare to report, end

    ## Clarify fixed quantization bitwidth setting for intermediate conv/FC layers
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

    if net_arch == 'proto-conv-net':
        inter_qb_tuples = ProtoConvNet.inter_qb_tuples
    else:
        raise Exception(f'Not supported model architecture called {net_arch}')

    if inter_qb_tuple_list_given is not None:
        inter_qb_tuple_list = inter_qb_tuple_list_given
    else:
        inter_qb_tuple_list = []
        for _ in range(inter_qb_tuples):
            b = random.randint(1, len(qb_tuple_cand_list))
            inter_qb_tuple_list.append(qb_tuple_cand_list[b-1])
    ## Clarify fixed quantization bitwidth setting for intermediate conv/FC layers, end

    # Set n_way_train, referring to the paper
    if dataset == 'omniglot':
        n_way_train = 60
    elif dataset == 'miniimagenet':
        if k_shot_sup == 1:
            n_way_train = 30
        elif k_shot_sup == 5:
            n_way_train = 20

    # Set k_shot_qry_train, referring to the paper
    if dataset == 'omniglot':
        k_shot_qry_train = 5
    elif dataset == 'miniimagenet':
        k_shot_qry_train = 15

    log('******************************          Arguments          ******************************')
    log(f'Time: {now}')

    log(f'# of workers: {workers}')
    log(f'Dataset: {dataset}')
    log(f'Path to dataset: {dataset_path}')

    log(f'Network model architecture: {net_arch}')
    log('Inductive setting (i.e., tracking running statistics in BN layer)')

    log(f'QAT/quantization scheme: {quant_scheme}')
    log(f'QAT/quantization scheme keyword arguments: {quant_scheme_kwargs}')
    log(f'For the first layer, use {qb_w_first}-bit weight and {qb_a_first}-bit activation')

    log('Phase 1: QAT + Prototypical Networks')
    log(f'Last phase1 epoch: {last_epoch}')
    log(f'Last best validation accuracy: {last_best_avg_val_acc_qry}')
    log(f'Phase1 epochs: {epochs}')
    log(f'For intermediate layers, use {inter_qb_tuple_list}')
    log(f'# of classes: {n_way_train} in training, {n_way_val} in validation')  
    log(f'{k_shot_sup} for support mini-batch')
    log(f'For query mini-batch, {k_shot_qry_train} in training and {k_shot_qry_val} in validation') 
    log(f'Optimizer: {optim}, {optim_kwargs}')
    log(f'LR scheduler: {lr_sch}, {lr_sch_kwargs}')    
    log(f'Save period: {save_period} epoch(s)')

    log('\n******************************           Phase 1           ******************************')
    # Define loss function
    loss_func = prototypical_loss

    ## Prepare dataset
    # Numbers
    steps_per_epoch = 100
    log(f'# of steps per epoch: {steps_per_epoch}\n')
    val_tasks_per_epoch = 200
    log(f'# of val tasks per epoch: {val_tasks_per_epoch}\n')

    # Data augmentation & sets
    if dataset == 'omniglot':
        # Train dataset
        train_set = omniglot(
            dataset_path, 
            k_shot_sup, 
            n_way_train, 
            shuffle=True, 
            test_shots=k_shot_qry_train,
            seed=None,
            meta_split='train',
            download=True)
        # Validation dataset
        val_set = omniglot(
            dataset_path, 
            k_shot_sup, 
            n_way_val, 
            shuffle=True, 
            test_shots=k_shot_qry_val,
            seed=None,
            meta_split='val',
            download=True)
    elif dataset == 'miniimagenet':
        # Train dataset
        train_set = miniimagenet(
            dataset_path, 
            k_shot_sup, 
            n_way_train, 
            shuffle=True, 
            test_shots=k_shot_qry_train,
            seed=None, 
            meta_split='train',
            download=True)
        # Validation dataset
        val_set = miniimagenet(
            dataset_path, 
            k_shot_sup, 
            n_way_val, 
            shuffle=True, 
            test_shots=k_shot_qry_val,
            seed=None, 
            meta_split='val',
            download=True)
    else:
        raise Exception(f'Not supported dataset called {dataset}')

    # Data loaders
    train_loader = BatchMetaDataLoader(
        train_set, 
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        pin_memory=True, 
        drop_last=True)
    if quant_scheme == 'lsq':
        temp_loader = BatchMetaDataLoader(
            train_set, 
            batch_size=1,
            shuffle=True)
    val_loader = BatchMetaDataLoader(
        val_set, 
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

    ## Prepare loop (base model, base optimizer, base LR scheduler) 
    if net_arch == 'proto-conv-net':
        net_base = ProtoConvNet(
            quant_scheme,
            quant_scheme_kwargs,
            qb_w_first, qb_a_first,
            inter_qb_tuple_list,
            in_channels,
            track_running_stats=True).to(device)

    optimizer_base = optim(net_base.parameters(), **optim_kwargs)
    lr_scheduler_base = lr_sch(optimizer_base, **lr_sch_kwargs)

    if last_epoch == 0:
        log('Starting from scratch.\n')
        if quant_scheme == 'lsq':
            with torch.no_grad():
                meta_batch = next(iter(temp_loader))
                x_temp, _ = meta_batch['train']
                imgs_temp = x_temp[0].to(device)
                _ = net_base(imgs_temp)
    else:
        base_state_dict = torch.load(f'./{middle_dir}/checkpoints/net_base_e{last_epoch}.pth')
        if quant_scheme == 'lsq':
            for key, param in base_state_dict.items():
                if 'scale_w' in key:
                    rdelattr(net_base, key)
                    rsetparam(net_base, key, nn.Parameter(param.clone().detach()))
                elif 'scale_a' in key:
                    rdelattr(net_base, key)
                    rsetparam(net_base, key, nn.Parameter(param.clone().detach()))
            net_base.load_state_dict(base_state_dict, strict=False)
        elif quant_scheme == 'dorefa':
            net_base.load_state_dict(base_state_dict, strict=True)

        optimizer_base.load_state_dict(
            torch.load(f'./{middle_dir}/checkpoints/optimizer_base_e{last_epoch}.pth')) 
        for _ in range(last_epoch):
            lr_scheduler_base.step()
        log(f'Successfully loaded base model, base optimizer, and base LR scheduler with {last_epoch}-th phase1 epoch.\n')
    ## Prepare loop (base model, base optimizer, base LR scheduler), end

    # Prepare to track best validation accuracy
    best_avg_val_acc_qry = last_best_avg_val_acc_qry

    # Main training & validation
    for epoch in range(last_epoch + 1, last_epoch + 1 + epochs):
        for step, meta_batch in enumerate(train_loader, start=1):
            x_sup, _ = meta_batch['train']
            x_qry, _ = meta_batch['test']

            optimizer_base.zero_grad()

            x = torch.cat(
                [x_sup[0], x_qry[0]], 
                0).to(device)

            if net_arch == 'proto-conv-net':
                net = ProtoConvNet(
                    quant_scheme,
                    quant_scheme_kwargs,
                    qb_w_first, qb_a_first,
                    inter_qb_tuple_list, 
                    in_channels,
                    track_running_stats=True).to(device)

            if quant_scheme == 'lsq':
                for key, param in net_base.named_parameters():
                    if 'scale_w' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    elif 'scale_a' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                net.load_state_dict(net_base.state_dict(), strict=False)
            elif quant_scheme == 'dorefa':
                net.load_state_dict(net_base.state_dict(), strict=True)

            outputs = net(x)
            loss_qry, acc_qry = loss_func(outputs, n_way_train, k_shot_sup, k_shot_qry_train)
            acc_qry *= 100

            base_grad_group = autograd.grad(loss_qry, net.parameters())
            base_grad_dict = {}
            for i, (key, _) in enumerate(net.named_parameters()):
                if base_grad_group[i] is not None:
                    base_grad_dict[key] = base_grad_group[i]

            for key, m in net_base.named_modules():
                if type(m) == BatchNorm2d:
                    net_bn_layer = rgetattr(net, key)
                    m.running_mean = net_bn_layer.running_mean
                    m.running_var = net_bn_layer.running_var
                    m.num_batches_tracked = net_bn_layer.num_batches_tracked

            for key, param in net_base.named_parameters():
                if key in base_grad_dict:
                    param.grad = torch.clamp(
                        base_grad_dict[key],
                        min=-10.0, max=10.0)

            optimizer_base.step()

            if step == steps_per_epoch:
                global_step = (epoch - 1) * steps_per_epoch + step

                log(f'\n  QAT + Prototypical Networks (phase 1) | {epoch}-th epoch, {step}-th step')

                log(f'  Query | specific-precision task | loss: {loss_qry.item():.3f}, accuracy: {acc_qry:.2f}%')

                tb_writer.add_scalar('loss_qry', loss_qry.item(), global_step)
                tb_writer.add_scalar('acc_qry', acc_qry, global_step)

            if step >= steps_per_epoch:
                break

        lr_scheduler_base.step()

        ## Validation on every epoch
        avg_val_loss_qry = 0.0
        avg_val_acc_qry = 0.0

        for t, meta_batch in enumerate(val_loader, start=1):
            x_sup, _ = meta_batch['train']
            x_qry, _ = meta_batch['test']

            x = torch.cat(
                [x_sup[0], x_qry[0]], 
                0).to(device)

            if net_arch == 'proto-conv-net':
                net = ProtoConvNet(
                    quant_scheme,
                    quant_scheme_kwargs,
                    qb_w_first, qb_a_first,
                    inter_qb_tuple_list, 
                    in_channels,
                    track_running_stats=True).to(device)

            if quant_scheme == 'lsq':
                for key, param in net_base.named_parameters():
                    if 'scale_w' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    elif 'scale_a' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                net.load_state_dict(net_base.state_dict(), strict=False)
            elif quant_scheme == 'dorefa':
                net.load_state_dict(net_base.state_dict(), strict=True)
            net.quant_w()
            net.eval()

            with torch.no_grad():
                outputs = net(x)
                loss_qry, acc_qry = loss_func(outputs, n_way_val, k_shot_sup, k_shot_qry_val)
            acc_qry *= 100

            avg_val_loss_qry += loss_qry.item() / val_tasks_per_epoch
            avg_val_acc_qry += acc_qry / val_tasks_per_epoch

            if t >= val_tasks_per_epoch:
                break

        log(f'\n  (QA-)Prototype calculation & inference (phase 2~3) | {epoch}-th epoch')

        log(f'  Inference | Avg over val tasks | loss: {avg_val_loss_qry:.3f}, accuracy: {avg_val_acc_qry:.2f}%')
        if best_avg_val_acc_qry < avg_val_acc_qry:
            best_avg_val_acc_qry = avg_val_acc_qry
            log(f'  Achieved best validation accuracy.')
        ## Validation on every epoch, end

        # Save base model and base optimizer
        if epoch % save_period == 0:
            torch.save(net_base.state_dict(), f'./{middle_dir}/checkpoints/net_base_e{epoch}.pth')
            torch.save(optimizer_base.state_dict(), f'./{middle_dir}/checkpoints/optimizer_base_e{epoch}.pth')
            log(f'\nSuccessfully saved base model and base optimizer with {epoch}-th phase1 epoch.\n') 

    log('\n*****************************************************************************************')
    print(f'{now}\n')   # Print to check
    tb_writer.close()

def mebqat_pn(
    workers,
    dataset,
    dataset_path,
    net_arch,
    quant_scheme,
    quant_scheme_kwargs,
    qb_w_first, qb_a_first,
    last_epoch,
    last_best_avg_val_acc_qry,
    epochs,
    inter_uniform,
    n_way_val,
    k_shot_sup,
    k_shot_qry_val,
    outer_optim,
    outer_optim_kwargs,
    outer_lr_sch,
    outer_lr_sch_kwargs,
    inner_qb_subtasks,
    save_period,
    track_running_stats):
    ## Prepare to report
    middle_dir = 'mebqat-pn'

    # Make directories if not exist
    if not os.path.exists(f'./{middle_dir}/reports/phase1/'):
        os.makedirs(f'./{middle_dir}/reports/phase1/')
    if not os.path.exists(f'./{middle_dir}/checkpoints/'):
        os.makedirs(f'./{middle_dir}/checkpoints/')

    # Add logger
    log_path = f'./{middle_dir}/reports/phase1/{now}.log'
    log, log_close = create_log_func(log_path)
    with open(log_path, 'w+') as f:
        pprint.pprint(f)

    # Add Tensorboard summary writer
    tb_writer = SummaryWriter(f'./{middle_dir}/reports/phase1/{now}_tb')
    ## Prepare to report, end

    # Set n_way_train, referring to the paper
    if dataset == 'omniglot':
        n_way_train = 60
    elif dataset == 'miniimagenet':
        if k_shot_sup == 1:
            n_way_train = 30
        elif k_shot_sup == 5:
            n_way_train = 20

    # Set k_shot_qry_train, referring to the paper
    if dataset == 'omniglot':
        k_shot_qry_train = 5
    elif dataset == 'miniimagenet':
        k_shot_qry_train = 15

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

    log('Phase 1: MEBQAT-PN')
    log(f'Last phase1 epoch: {last_epoch}')
    log(f'Last best validation accuracy: {last_best_avg_val_acc_qry}')
    log(f'Phase1 epochs: {epochs}')
    log(f'Whether to use a uniform quantization bitwidth setting for intermediate layers in each model: {inter_uniform}')
    log(f'# of classes: {n_way_train} in training, {n_way_val} in validation')  
    log(f'{k_shot_sup} for support mini-batch')
    log(f'For query mini-batch, {k_shot_qry_train} in training and {k_shot_qry_val} in validation') 
    log(f'Outer-loop optimizer: {outer_optim}, {outer_optim_kwargs}')
    log(f'Outer-loop LR scheduler: {outer_lr_sch}, {outer_lr_sch_kwargs}')    
    log(f'# of training quantization bitwidth subtasks (i.e., meta batch size): {inner_qb_subtasks}') 
    log(f'Save period: {save_period} epoch(s)')

    log('\n******************************           Phase 1           ******************************')
    # Define loss function
    loss_func = prototypical_loss

    # Prepare quantization
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

    ## Prepare dataset
    # Numbers
    steps_per_epoch = 100
    log(f'# of steps per epoch: {steps_per_epoch}\n')
    val_tasks_per_epoch = 200
    log(f'# of val tasks per epoch: {val_tasks_per_epoch}\n')

    # Data augmentation & sets
    if dataset == 'omniglot':
        # Train dataset
        train_set = omniglot(
            dataset_path, 
            k_shot_sup, 
            n_way_train, 
            shuffle=True, 
            test_shots=k_shot_qry_train,
            seed=None,
            meta_split='train',
            download=True)
        # Validation dataset
        val_set = omniglot(
            dataset_path, 
            k_shot_sup, 
            n_way_val, 
            shuffle=True, 
            test_shots=k_shot_qry_val,
            seed=None,
            meta_split='val',
            download=True)
    elif dataset == 'miniimagenet':
        # Train dataset
        train_set = miniimagenet(
            dataset_path, 
            k_shot_sup, 
            n_way_train, 
            shuffle=True, 
            test_shots=k_shot_qry_train,
            seed=None, 
            meta_split='train',
            download=True)
        # Validation dataset
        val_set = miniimagenet(
            dataset_path, 
            k_shot_sup, 
            n_way_val, 
            shuffle=True, 
            test_shots=k_shot_qry_val,
            seed=None, 
            meta_split='val',
            download=True)
    else:
        raise Exception(f'Not supported dataset called {dataset}')

    # Data loaders
    train_loader = BatchMetaDataLoader(
        train_set, 
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        pin_memory=True, 
        drop_last=True)
    if quant_scheme == 'lsq':
        temp_loader = BatchMetaDataLoader(
            train_set, 
            batch_size=1,
            shuffle=True)
    val_loader = BatchMetaDataLoader(
        val_set, 
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

    ## Prepare loop (base model, base optimizer, base LR scheduler) 
    if net_arch == 'proto-conv-net':
        net_base = ProtoConvNet(
            quant_scheme,
            quant_scheme_kwargs,
            qb_w_first, qb_a_first,
            [(2, 2)]*inter_qb_tuples,
            in_channels,
            track_running_stats=track_running_stats).to(device)

    optimizer_base = outer_optim(net_base.parameters(), **outer_optim_kwargs)
    lr_scheduler_base = outer_lr_sch(optimizer_base, **outer_lr_sch_kwargs)

    if last_epoch == 0:
        log('Starting from scratch.\n')
        if quant_scheme == 'lsq':
            with torch.no_grad():
                meta_batch = next(iter(temp_loader))
                x_temp, _ = meta_batch['train']
                imgs_temp = x_temp[0].to(device)
                _ = net_base(imgs_temp)
    else:
        base_state_dict = torch.load(f'./{middle_dir}/checkpoints/net_base_e{last_epoch}.pth')
        if quant_scheme == 'lsq':
            for key, param in base_state_dict.items():
                if 'scale_w' in key:
                    rdelattr(net_base, key)
                    rsetparam(net_base, key, nn.Parameter(param.clone().detach()))
                elif 'scale_a' in key:
                    rdelattr(net_base, key)
                    rsetparam(net_base, key, nn.Parameter(param.clone().detach()))
            net_base.load_state_dict(base_state_dict, strict=False)
        elif quant_scheme == 'dorefa':
            net_base.load_state_dict(base_state_dict, strict=True)

        optimizer_base.load_state_dict(
            torch.load(f'./{middle_dir}/checkpoints/optimizer_base_e{last_epoch}.pth')) 
        for _ in range(last_epoch):
            lr_scheduler_base.step()
        log(f'Successfully loaded base model, base optimizer, and base LR scheduler with {last_epoch}-th phase1 epoch.\n')
    ## Prepare loop (base model, base optimizer, base LR scheduler), end

    # Prepare to track best validation accuracy
    best_avg_val_acc_qry = last_best_avg_val_acc_qry

    # Main training & validation
    for epoch in range(last_epoch + 1, last_epoch + 1 + epochs):
        # Outer-loop
        for step, meta_batch in enumerate(train_loader, start=1):
            x_sup, _ = meta_batch['train']
            x_qry, _ = meta_batch['test']

            sum_base_grad_dict = {}
            optimizer_base.zero_grad()

            total_conv_quant_cnt_list_w = [0] * (inter_qb_tuples + 1) 
            total_conv_quant_cnt_list_a = [0] * (inter_qb_tuples + 1)

            # Prepare data
            try:
                x = torch.cat(
                    [x_sup[0], x_qry[0]], 
                    0).to(device)
            except:
                log(f'Skipped {epoch}-th epoch -> {step}-th step')
                continue

            # Inner-loop
            for qbst in range(1, inner_qb_subtasks + 1):
                # Randomly select quantization bitwidth setting
                if inter_uniform:
                    b = random.randint(1, len(qb_tuple_cand_list))
                    inter_qb_tuple_list = [qb_tuple_cand_list[b-1]] * inter_qb_tuples
                else:
                    inter_qb_tuple_list = []
                    for _ in range(inter_qb_tuples):
                        b = random.randint(1, len(qb_tuple_cand_list))
                        inter_qb_tuple_list.append(qb_tuple_cand_list[b-1]) 

                ## Prepare inner-loop (model, optimizer)
                if net_arch == 'proto-conv-net':
                    net = ProtoConvNet(
                        quant_scheme,
                        quant_scheme_kwargs,
                        qb_w_first, qb_a_first,
                        inter_qb_tuple_list, 
                        in_channels,
                        track_running_stats=track_running_stats).to(device)

                if quant_scheme == 'lsq':
                    for key, param in net_base.named_parameters():
                        if 'scale_w' in key:
                            if key in dict(net.named_parameters()).keys():
                                rdelattr(net, key)
                                rsetparam(net, key, nn.Parameter(param.clone().detach()))
                        elif 'scale_a' in key:
                            if key in dict(net.named_parameters()).keys():
                                rdelattr(net, key)
                                rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    net.load_state_dict(net_base.state_dict(), strict=False)
                elif quant_scheme == 'dorefa':
                    net.load_state_dict(net_base.state_dict(), strict=False)
                ## Prepare inner-loop, end

                outputs = net(x)
                loss_qry, acc_qry = loss_func(outputs, n_way_train, k_shot_sup, k_shot_qry_train)
                acc_qry *= 100

                ## Sum up this inner-loop
                base_grad_group = autograd.grad(loss_qry, net.parameters())
                base_grad_dict = {}
                for i, (key, _) in enumerate(net.named_parameters()):
                    if base_grad_group[i] is not None:
                        base_grad_dict[key] = base_grad_group[i]

                if track_running_stats:
                    for key, m in net_base.named_modules():
                        if type(m) == BatchNorm2d:
                            net_bn_layer = rgetattr(net, key)
                            m.running_mean = net_bn_layer.running_mean
                            m.running_var = net_bn_layer.running_var
                            m.num_batches_tracked = net_bn_layer.num_batches_tracked

                if qbst == 1:
                    sum_loss_qry = loss_qry.item()
                    sum_acc_qry = acc_qry

                    for key, param in net.named_parameters():
                        if key in base_grad_dict:
                            sum_base_grad_dict[key] = base_grad_dict[key]
                else:
                    sum_loss_qry += loss_qry.item()
                    sum_acc_qry += acc_qry

                    for key, param in net.named_parameters():
                        if key in base_grad_dict:
                            if key in sum_base_grad_dict:
                                sum_base_grad_dict[key] += base_grad_dict[key]
                            else:
                                sum_base_grad_dict[key] = base_grad_dict[key]

                conv_quant_cnt_list_w, conv_quant_cnt_list_a = net.get_conv_quant_cnt_lists()
                total_conv_quant_cnt_list_w = [sum(i) for i in zip(total_conv_quant_cnt_list_w, conv_quant_cnt_list_w)]
                total_conv_quant_cnt_list_a = [sum(i) for i in zip(total_conv_quant_cnt_list_a, conv_quant_cnt_list_a)] 
                ## Sum up this inner-loop, end

            ## Sum up this outer-loop
            avg_loss_qry = sum_loss_qry / float(inner_qb_subtasks)
            avg_acc_qry = sum_acc_qry / float(inner_qb_subtasks) 

            if quant_scheme == 'lsq':
                conv_info_dict = net_base.get_conv_info_dict()
                for key, param in net_base.named_parameters():
                    if key in sum_base_grad_dict:
                        if key.replace('.scale_w', '') in conv_info_dict.keys():
                            divisor = total_conv_quant_cnt_list_w[conv_info_dict[key.replace('.scale_w', '')]]
                            assert divisor != 0
                        elif key.replace('.scale_a', '') in conv_info_dict.keys():
                            divisor = total_conv_quant_cnt_list_a[conv_info_dict[key.replace('.scale_a', '')]]
                            assert divisor != 0
                        else:   # Not a quant param
                            divisor = inner_qb_subtasks

                        param.grad = torch.clamp(
                            sum_base_grad_dict[key] / float(divisor),
                            min=-10.0, max=10.0)
                    else:   # For checking validity
                        if key.replace('.scale_w', '') in conv_info_dict.keys():
                            divisor = total_conv_quant_cnt_list_w[conv_info_dict[key.replace('.scale_w', '')]]
                            assert divisor == 0
                        elif key.replace('.scale_a', '') in conv_info_dict.keys():
                            divisor = total_conv_quant_cnt_list_a[conv_info_dict[key.replace('.scale_a', '')]]
                            assert divisor == 0
            elif quant_scheme == 'dorefa':
                for key, param in net_base.named_parameters():
                    if key in sum_base_grad_dict:
                        param.grad = torch.clamp(
                            sum_base_grad_dict[key] / float(inner_qb_subtasks),
                            min=-10.0, max=10.0)

            optimizer_base.step()

            if step == steps_per_epoch:
                global_step = (epoch - 1) * steps_per_epoch + step

                log(f'\n  MEBQAT-PN (phase 1) | {epoch}-th epoch, {step}-th step')

                log(f'  Query | Avg over tasks | loss: {avg_loss_qry:.3f}, accuracy: {avg_acc_qry:.2f}%')

                tb_writer.add_scalar('avg_loss_qry', avg_loss_qry, global_step)
                tb_writer.add_scalar('avg_acc_qry', avg_acc_qry, global_step)
            ## Sum up this outer-loop, end

            if step >= steps_per_epoch:
                break

        lr_scheduler_base.step()

        ## Validation on every epoch
        avg_val_loss_qry = 0.0
        avg_val_acc_qry = 0.0

        for t, meta_batch in enumerate(val_loader, start=1):
            x_sup, _ = meta_batch['train']
            x_qry, _ = meta_batch['test']

            x = torch.cat(
                [x_sup[0], x_qry[0]], 
                0).to(device)

            # Randomly select quantization bitwidth setting
            if inter_uniform:
                b = random.randint(1, len(qb_tuple_cand_list))
                inter_qb_tuple_list = [qb_tuple_cand_list[b-1]] * inter_qb_tuples
            else:
                inter_qb_tuple_list = []
                for _ in range(inter_qb_tuples):
                    b = random.randint(1, len(qb_tuple_cand_list))
                    inter_qb_tuple_list.append(qb_tuple_cand_list[b-1]) 

            if net_arch == 'proto-conv-net':
                net = ProtoConvNet(
                    quant_scheme,
                    quant_scheme_kwargs,
                    qb_w_first, qb_a_first,
                    inter_qb_tuple_list, 
                    in_channels,
                    track_running_stats=track_running_stats).to(device)

            if quant_scheme == 'lsq':
                for key, param in net_base.named_parameters():
                    if 'scale_w' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                    elif 'scale_a' in key:
                        if key in dict(net.named_parameters()).keys():
                            rdelattr(net, key)
                            rsetparam(net, key, nn.Parameter(param.clone().detach()))
                net.load_state_dict(net_base.state_dict(), strict=False)
            elif quant_scheme == 'dorefa':
                net.load_state_dict(net_base.state_dict(), strict=False)
            net.quant_w()
            net.eval()

            with torch.no_grad():
                outputs = net(x)
                loss_qry, acc_qry = loss_func(outputs, n_way_val, k_shot_sup, k_shot_qry_val)
            acc_qry *= 100

            avg_val_loss_qry += loss_qry.item() / val_tasks_per_epoch
            avg_val_acc_qry += acc_qry / val_tasks_per_epoch

            if t >= val_tasks_per_epoch:
                break

        log(f'\n  (QA-)Prototype calculation & inference (phase 2~3) | {epoch}-th epoch')

        log(f'  Inference | Avg over val tasks | loss: {avg_val_loss_qry:.3f}, accuracy: {avg_val_acc_qry:.2f}%')
        if best_avg_val_acc_qry < avg_val_acc_qry:
            best_avg_val_acc_qry = avg_val_acc_qry
            log(f'  Achieved best validation accuracy.')
        ## Validation on every epoch, end

        # Save base model and base optimizer
        if epoch % save_period == 0:
            torch.save(net_base.state_dict(), f'./{middle_dir}/checkpoints/net_base_e{epoch}.pth')
            torch.save(optimizer_base.state_dict(), f'./{middle_dir}/checkpoints/optimizer_base_e{epoch}.pth')
            log(f'\nSuccessfully saved base model and base optimizer with {epoch}-th phase1 epoch.\n') 

    log('\n*****************************************************************************************')
    print(f'{now}\n')   # Print to check
    tb_writer.close()

def main():
    if PHASE1_SCHEME == 'qat-fomaml':
        qat_fomaml(
            WORKERS,
            DATASET,    
            DATASET_PATH,
            NET_ARCH,
            QUANT_SCHEME,
            QUANT_SCHEME_KWARGS,
            QB_W_FIRST, QB_A_FIRST,
            QB_W_LAST, QB_A_LAST,
            LAST_EPOCH,
            LAST_BEST_AVG_VAL_ACC_IFR,
            EPOCHS,
            INTER_QB_TUPLE_LIST_GIVEN,
            N_WAY,
            K_SHOT_SUP,
            K_SHOT_QRY,
            OUTER_OPTIM,
            OUTER_OPTIM_KWARGS,
            INNER_CL_SUBTASKS,
            INNER_OPTIM,
            INNER_OPTIM_KWARGS, 
            INNER_UPDATES,
            VAL_UPDATES,
            SAVE_PERIOD)
    elif PHASE1_SCHEME == 'mebqat-maml':
        mebqat_maml(
            WORKERS,
            DATASET,    
            DATASET_PATH,
            NET_ARCH,
            QUANT_SCHEME,
            QUANT_SCHEME_KWARGS,
            QB_W_FIRST, QB_A_FIRST,
            QB_W_LAST, QB_A_LAST,
            LAST_EPOCH,
            LAST_BEST_AVG_VAL_ACC_IFR,
            EPOCHS,
            INTER_UNIFORM,
            N_WAY,
            K_SHOT_SUP,
            K_SHOT_QRY,
            OUTER_OPTIM,
            OUTER_OPTIM_KWARGS,
            INNER_CL_SUBTASKS,
            INNER_OPTIM,
            INNER_OPTIM_KWARGS, 
            INNER_UPDATES,
            VAL_UPDATES,
            SAVE_PERIOD)
    elif PHASE1_SCHEME == 'qat-pn':
        qat_pn(
            WORKERS,
            DATASET,
            DATASET_PATH,
            NET_ARCH,
            QUANT_SCHEME,
            QUANT_SCHEME_KWARGS,
            QB_W_FIRST, QB_A_FIRST,
            LAST_EPOCH,
            LAST_BEST_AVG_VAL_ACC_QRY,
            EPOCHS,
            INTER_QB_TUPLE_LIST_GIVEN,
            N_WAY_VAL,
            K_SHOT_SUP,
            K_SHOT_QRY_VAL,
            OPTIM,
            OPTIM_KWARGS,
            LR_SCH,
            LR_SCH_KWARGS,
            SAVE_PERIOD)
    elif PHASE1_SCHEME == 'mebqat-pn':
        mebqat_pn(
            WORKERS,
            DATASET,
            DATASET_PATH,
            NET_ARCH,
            QUANT_SCHEME,
            QUANT_SCHEME_KWARGS,
            QB_W_FIRST, QB_A_FIRST,
            LAST_EPOCH,
            LAST_BEST_AVG_VAL_ACC_QRY,
            EPOCHS,
            INTER_UNIFORM,
            N_WAY_VAL,
            K_SHOT_SUP,
            K_SHOT_QRY_VAL,
            OUTER_OPTIM,
            OUTER_OPTIM_KWARGS,
            OUTER_LR_SCH,
            OUTER_LR_SCH_KWARGS,
            INNER_QB_SUBTASKS,
            SAVE_PERIOD,
            TRACK_RUNNING_STATS)
    else:
        raise Exception(f'Not supported phase1 scheme called {PHASE1_SCHEME}')

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
