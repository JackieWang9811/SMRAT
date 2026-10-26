import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets_factory import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders
from config_smrat import Config as SHDTConfig

from models.smrat import SMRAT

import utils
import numpy as np
import torch
from utils import init_logger, build_optimizer
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from spikingjelly.activation_based import functional
from datetime import datetime
from uuid import uuid4
import os
import matplotlib.pyplot as plt
import json
import statistics

eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())


def calc_loss(config, output, y):
    if config.loss == 'mean':
        m = torch.mean(output, 0)
    elif config.loss == 'max':
        m, _ = torch.max(output, 0)
    elif config.loss == 'spike_count':
        m = torch.sum(output, 0)
    elif config.loss == 'sum':
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)
    elif config.loss == 'none':
        m = output
    # probably better to add it in init, or in general do it one time only
    if config.loss_fn == 'CEloss':
        # compare using this to directly using nn.CrossEntropyLoss

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(m, y)

        return loss


def calc_loss_nonspike(config, output, y):
    # probably better to add it in init, or in general do it one time only
    if config.loss_fn == 'CEloss':
        # compare using this to directly using nn.CrossEntropyLoss

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(output, y)

        return loss


def calc_metric(config, output, y):
    # mean accuracy over batch
    if config.loss == 'mean':
        m = torch.mean(output, 0)

    elif config.loss == 'max':
        m, _ = torch.max(output, 0)

    elif config.loss == 'spike_count':
        m = torch.sum(output, 0)

    elif config.loss == 'sum':
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)

    elif config.loss == 'none':
        m = output

    return np.mean((torch.max(y, 1)[1] == torch.max(m, 1)[1]).detach().cpu().numpy())


def train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, num_epochs):
    ##################################    Train Loop    ##############################

    loss_epochs = {'train': [], 'valid': [], 'test': []}
    metric_epochs = {'train': [], 'valid': [], 'test': []}
    best_metric_val = 0  # 1e6
    best_metric_test = 0  # 1e6
    best_loss_val = 1e6
    # calc_loss_std = SoftTargetCrossEntropy()

    for epoch in range(num_epochs):

        ##################################    Train Loop    ##############################
        model.train()
        # last element in the tuple corresponds to the collate_fn return
        loss_batch, metric_batch = [], []

        for i, (x, y, x_len) in enumerate(tqdm(train_loader)):
            y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)  # (batch, time, neurons)
            x = x.permute(0, 2, 1)
            y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(x)
            loss = calc_loss(config, output, y)

            loss.backward()
            optimizer.step()

            metric = calc_metric(config, output, y)

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)

        loss_epochs['train'].append(np.mean(loss_batch))
        metric_epochs['train'].append(np.mean(metric_batch))

        scheduler.step()

        ##################################    Eval Loop    #########################
        model.eval()
        with torch.no_grad():
            loss_batch, metric_batch = [], []
            for i, (x, y, x_len) in enumerate(tqdm(valid_loader)):
                y = F.one_hot(y, config.n_outputs).float()
                x = x.float().to(device)
                x = x.permute(0, 2, 1)
                y = y.to(device)

                output = model(x)
                loss = calc_loss(config, output, y)
                metric = calc_metric(config, output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)

                functional.reset_net(model)

        loss_valid = np.mean(loss_batch)
        metric_valid = np.mean(metric_batch)

        loss_epochs['valid'].append(loss_valid)
        metric_epochs['valid'].append(metric_valid)

        ########################## Logging and Plotting  ##########################

        logger.info(
            f"=====> Epoch {epoch} : Loss Train = {loss_epochs['train'][-1]:.3f}  |  Acc Train = {100 * metric_epochs['train'][-1]:.2f}%")
        logger.info(
            f"Loss Valid = {loss_epochs['valid'][-1]:.3f}  |  Acc Valid = {100 * metric_epochs['valid'][-1]:.2f}%  |  Best Acc Valid = {100 * max(metric_epochs['valid'][-1], best_metric_val):.2f}%")
        if test_loader:
            logger.info(
                f"Loss Test = {loss_epochs['test'][-1]:.3f}  |  Acc Test = {100 * metric_epochs['test'][-1]:.2f}%  |  Best Acc Test = {100 * max(metric_epochs['test'][-1], best_metric_test):.2f}%")

        checkpoint_dir = os.path.join('checkpoints', config.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        ave_model_path = os.path.join(checkpoint_dir, config.save_model_path)

        if metric_valid > best_metric_val:
            print("# Saving best Metric model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_ACC'))
            best_metric_val = metric_valid

        if loss_valid < best_loss_val:
            print("# Saving best Loss model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_Loss'))
            best_loss_val = loss_valid

    return loss_epochs, metric_epochs


def train_model_with_seeds(config, train_loader, valid_loader, test_loader, device, model_class, optimizer_class,
                           scheduler_class, num_epochs, seeds):
    all_metrics = []

    for seed in seeds:
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize model, optimizer, and scheduler
        model = model_class(
            config,
            d_output=config.n_outputs,
            d_model=config.n_hidden_neurons,
            n_layers=config.layers,
            encoder="speech",
            n_tokens=None,
            max_length=config.max_length,
            depth=config.depths,
        ).to(device)

        optimizer = optimizer_class(config, model)
        scheduler = scheduler_class(optimizer, T_max=config.t_max)

        log_and_dump_run_artifacts(logger, config, seed, model)

        # Train the model
        loss_epochs, metric_epochs = train_model(config, train_loader, valid_loader, test_loader, device, model,
                                                 optimizer, scheduler, num_epochs)

        # Store the best validation metric
        all_metrics.append(max(metric_epochs['valid']))

    # Calculate mean and standard deviation of metrics
    mean_metric = statistics.mean(all_metrics)
    std_metric = statistics.stdev(all_metrics)

    # 
    mean_percent = mean_metric * 100
    std_percent = std_metric * 100

    # 
    print(f"All Metrics from different seeds: {all_metrics}")
    print(f"Mean Validation Accuracy: {mean_percent:.2f} +/- {std_percent:.2f}")

    # 
    return mean_percent, std_percent


def _extract_config_dict(cfg_obj):
    """Extract serializable config fields for logging."""
    cfg_dict = {}
    for k in dir(cfg_obj):
        if k.startswith('_'):
            continue
        v = getattr(cfg_obj, k)
        if callable(v):
            continue
        try:
            json.dumps(v)
            cfg_dict[k] = v
        except TypeError:
            cfg_dict[k] = str(v)
    return dict(sorted(cfg_dict.items(), key=lambda x: x[0]))


def log_and_dump_run_artifacts(logger, config, seed_val, model):
    # 
    logger.info("##############################################\n")
    logger.info("Seed :{}".format(seed_val))
    epochs = config.epochs
    logger.info("The epoch is: {}".format(epochs))
    logger.info("The batch size is : {}".format(config.batch_size))
    logger.info("The backend is: {}".format(config.backend))

    if getattr(config, "use_aug", False):
        logger.info("The TN_mask_aug_proba is: {}".format(config.TN_mask_aug_proba))
        logger.info("The time_mask_proportion is :{}".format(config.time_mask_proportion))
        logger.info("The neuron_mask_size is: {}".format(config.neuron_mask_size))
    else:
        logger.info("The augs is None")

    logger.info("The time_step is: {}".format(config.time_step))
    logger.info("The n_bins is: {}".format(config.n_bins))
    logger.info("The hidden_dim is: {}".format(config.n_hidden_neurons))
    logger.info("The spike_mode is: {}".format(config.spike_mode))
    logger.info("The block_mode is :{}".format(config.block_type))
    logger.info("The kernel_size is: {}".format(config.kernel_sizes[0]))
    logger.info("The compartments is: {}".format(config.compartments))
    logger.info("The layers of SMRAT is: {}".format(config.layers))
    logger.info("The depths in SMRAT is: {}".format(config.depths))

    # model size
    logger.info("Model size:{}".format(utils.count_parameters(model)))

    # 
    logger.info("lr_w: {}".format(config.lr_w))
    logger.info("weight_decay:{}".format(config.weight_decay))
    logger.info("T:{}".format(config.t_max))

    # 
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    dataset_info = config.dataset

    # 
    folder_path = os.path.join('model_structure', dataset_info)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f'model_structure_{dataset_info}_seed{seed_val}_{formatted_time}.txt')
    with open(filename, 'w') as f:
        print(model, file=f)

    cfg_dir = os.path.join('config_recording', dataset_info)
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_txt = os.path.join(cfg_dir, f'config_{dataset_info}_seed{seed_val}_{formatted_time}.txt')

    _cfg = _extract_config_dict(config)
    with open(cfg_txt, 'w', encoding='utf-8') as f:
        f.write("# ===== Config Snapshot =====\n")
        for k, v in _cfg.items():
            f.write(f"{k}: {v}\n")


if __name__ == '__main__':

    config = SHDTConfig()

    logger = init_logger(config, "training")
    logger.info("Logger is properly initialized and ready to use.")
    logger.info("The GPU is {}".format(config.gpu))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        dev = f"cuda:{config.gpu}"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    if config.dataset == 'shd':
        train_loader, valid_loader = SHD_dataloaders(config)
        test_loader = None
    elif config.dataset == 'ssc':
        train_loader, valid_loader, test_loader = SSC_dataloaders(config)
    elif config.dataset == 'gsc':
        train_loader, valid_loader, test_loader = GSC_dataloaders(config)
    else:
        raise Exception(f'dataset {config.dataset} not implemented')

    seeds = config.seeds

    mean_metric, std_metric = train_model_with_seeds(
        config,
        train_loader,
        valid_loader,
        test_loader,  # 
        device,
        SMRAT,
        build_optimizer,
        lr_scheduler.CosineAnnealingLR,
        config.epochs,
        seeds
    )

    logger.info(f"Final Result: Mean +/- Std = {mean_metric:.2f} +/- {std_metric:.2f}")
