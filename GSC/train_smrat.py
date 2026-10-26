import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets_factory import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders
from config_smrat import Config as GSCTConfig

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
from datasets_factory import SpecAugment
import json
import statistics

eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())


def comp_kernel_diversity(comp_kernels):
    # list of [C,1,K]
    Ks = [k.squeeze(1) for k in comp_kernels]   # [C,K]
    Kmat = torch.stack(Ks, dim=0)               # [M,C,K]
    Kmat = F.normalize(Kmat, dim=-1)
    sims = torch.einsum("mck,nck->mn", Kmat, Kmat)
    mask = torch.triu(torch.ones_like(sims), diagonal=1)
    return (sims[mask == 1] ** 2).mean()


def attach_all_blocks_comp_firing_hooks(model):
    """Cleaned documentation for the anonymous supplementary release."""
    stats = {}   # (blk, comp, level) -> dict(sum_rate, n_calls)
    hooks = []

    for blk, mix in enumerate(model.mixing_layers):
        mr = mix.multires_gate  # MultiresLayer
        for m in range(mr.n_comp):
            for level in range(mr.depth):
                key = (blk, m, level)
                stats[key] = {"sum_rate": 0.0, "n_calls": 0}

                lif = mr.comp_lif_layers[m][level]

                def _make_hook(_key):
                    def _hook(module, inp, out):
                        if torch.is_tensor(out):
                            rate = out.detach().float().mean().item()
                            stats[_key]["sum_rate"] += rate
                            stats[_key]["n_calls"] += 1
                    return _hook

                hooks.append(lif.register_forward_hook(_make_hook(key)))

    return stats, hooks


@torch.no_grad()
def print_all_blocks_stats(stats, n_blocks=4, depth=None, n_comp=None):
    if depth is None:
        depth = max(k[2] for k in stats.keys()) + 1
    if n_comp is None:
        n_comp = max(k[1] for k in stats.keys()) + 1

    print("\n[All blocks comp_lif firing rate]")
    for blk in range(n_blocks):
        print(f"Block {blk}:")
        for level in range(depth):
            line = [f"  level={level}:"]
            for m in range(n_comp):
                key = (blk, m, level)
                s = stats.get(key, None)
                if s is None:
                    line.append(f"comp={m}: N/A")
                    continue
                n = s["n_calls"]
                avg = s["sum_rate"] / max(n, 1)
                line.append(f"comp={m}: {avg:.6f}")
            print("    " + " | ".join(line))


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


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
    else:
        # 
        m = torch.mean(output, 0)

    if config.loss_fn == 'CEloss':
        CEloss = nn.CrossEntropyLoss()
        return CEloss(m, y)
    raise ValueError(f"loss_fn {config.loss_fn} not implemented")


def calc_metric(config, output, y):
    if config.loss == 'mean':
        m = torch.mean(output, 0)
    elif config.loss == 'max':
        m, _ = torch.max(output, 0)
    elif config.loss == 'spike_count':
        m = torch.sum(output, 0)
    elif config.loss == 'sum':
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)
    else:
        m = torch.mean(output, 0)

    return np.mean((torch.max(y, 1)[1] == torch.max(m, 1)[1]).detach().cpu().numpy())


def eval_model(config, model, loader, device):
    model.eval()
    with torch.no_grad():
        loss_batch, metric_batch = [], []
        for i, (x, y, x_len) in enumerate(tqdm(loader)):
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

    return np.mean(loss_batch), np.mean(metric_batch)


def train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, num_epochs):
    loss_epochs = {'train': [], 'valid': [], 'test': []}
    metric_epochs = {'train': [], 'valid': [], 'test': []}
    best_metric_val = 0
    best_metric_test = 0
    test_at_best_val = None
    best_loss_val = 1e6

    for epoch in range(num_epochs):
        # ---------------- Train ----------------
        model.train()
        loss_batch, metric_batch = [], []

        for i, (x, y, x_len) in enumerate(tqdm(train_loader)):

            if config.use_aug:
                x = augs(x, x_len)

            y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)  # (B,T,N)
            x = x.permute(0, 2, 1)    # (B,N,T)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = calc_loss(config, output, y)

            # 
            # loss_div = 0
            # lambda_div = 1e-3
            # for blk in model.mixing_layers:
            #     loss_div += comp_kernel_diversity(blk.multires_gate.comp_kernels)
            # loss = loss + lambda_div * loss_div

            loss.backward()
            optimizer.step()

            metric = calc_metric(config, output, y)
            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)

        loss_epochs['train'].append(np.mean(loss_batch))
        metric_epochs['train'].append(np.mean(metric_batch))

        scheduler.step()

        # ---------------- Valid ----------------
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

        # ---------------- Test ----------------
        if test_loader:
            loss_test, metric_test = eval_model(config, model, test_loader, device)
        else:
            loss_test, metric_test = 100, 0
        loss_epochs['test'].append(loss_test)
        metric_epochs['test'].append(metric_test)

        # ---------------- Logging & Save ----------------
        logger.info(
            f"=====> Epoch {epoch} : Loss Train = {loss_epochs['train'][-1]:.3f}  |  Acc Train = {100 * metric_epochs['train'][-1]:.2f}%")
        logger.info(
            f"Loss Valid = {loss_epochs['valid'][-1]:.3f}  |  Acc Valid = {100 * metric_epochs['valid'][-1]:.2f}%  |  Best Acc Valid = {100 * max(metric_epochs['valid'][-1], best_metric_val):.2f}%")
        if test_loader:
            logger.info(
                f"Loss Test = {loss_epochs['test'][-1]:.3f}  |  Acc Test = {100 * metric_epochs['test'][-1]:.2f}%  |  Best Acc Test = {100 * max(metric_epochs['test'][-1], best_metric_test):.2f}%")

        checkpoint_dir = os.path.join('checkpoints', config.dataset)
        os.makedirs(checkpoint_dir, exist_ok=True)
        ave_model_path = os.path.join(checkpoint_dir, config.save_model_path)

        if metric_valid > best_metric_val:
            print("# Saving best Metric model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_ACC'))
            best_metric_val = metric_valid
            test_at_best_val = metric_test if test_loader else None

        if loss_valid < best_loss_val:
            print("# Saving best Loss model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_Loss'))
            best_loss_val = loss_valid

        # Test scores are never used for checkpoint selection.

    metric_epochs['test_at_best_val'] = test_at_best_val
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
        all_metrics.append(metric_epochs['test_at_best_val'])

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
    logger.info("##############################################\n")
    logger.info("Seed :{}".format(seed_val))
    epochs = config.epochs
    logger.info("The epoch is: {}".format(epochs))
    logger.info("The batch size is : {}".format(config.batch_size))
    logger.info("The backend is: {}".format(config.backend))

    if getattr(config, "use_aug", False):
        logger.info("The mF is: {}".format(config.mF))
        logger.info("The F is: {}".format(config.F))
        logger.info("The mT is: {}".format(config.mT))
        logger.info("The pS is: {}".format(config.pS))
    else:
        logger.info("The augs is None")

    logger.info("The window_size is {}".format(config.window_size))
    logger.info("The hop_length is {}".format(config.hop_length))
    logger.info("The n_mels is {}".format(config.n_mels))
    logger.info("The hidden_dim is: {}".format(config.n_hidden_neurons))
    logger.info("The spike_mode is: {}".format(config.spike_mode))
    logger.info("The block_mode is :{}".format(config.block_type))
    logger.info("The kernel_size is: {}".format(config.kernel_sizes[0]))
    logger.info("The compartments is: {}".format(config.compartments))
    logger.info("The layers of SMRAT is: {}".format(config.layers))
    logger.info("The depths in SMRAT is: {}".format(config.depths))

    logger.info("Model size:{}".format(utils.count_parameters(model)))
    logger.info("lr_w: {}".format(config.lr_w))
    logger.info("weight_decay: {}".format(config.weight_decay))
    logger.info("T: {}".format(config.t_max))

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

    config = GSCTConfig()
    logger = init_logger(config, "training")
    logger.info("Logger is properly initialized and ready to use.")
    logger.info("The GPU is {}".format(config.gpu))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''' set up device '''
    # use cuda
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

    augs = SpecAugment(config)

    mean_metric, std_metric = train_model_with_seeds(
        config,
        train_loader,
        valid_loader,
        test_loader,
        device,
        SMRAT,
        build_optimizer,
        lr_scheduler.CosineAnnealingLR,
        config.epochs,
        seeds
    )

    logger.info(f"Final Result: Mean +/- Std = {mean_metric:.2f} +/- {std_metric:.2f}")
