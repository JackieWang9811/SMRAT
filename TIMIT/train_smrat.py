from config_smrat import Config as TIMITConfig
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.smrat import SMRAT
from datasets_factory import build_timit_dataloaders

from torch.utils import data
import utils
import numpy as np
import torch
from utils import init_logger,build_optimizer
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from datetime import datetime
from uuid import uuid4
import os
import matplotlib.pyplot as plt
import json
import statistics

eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

def calc_loss_seq_ce(output_logits, y_idx):
    """Cleaned documentation for the anonymous supplementary release."""
    B, T, C = output_logits.shape
    return F.cross_entropy(output_logits.reshape(B*T, C), y_idx.reshape(B*T))

def calc_metric_seq_acc(output_logits, y_idx):
    """Cleaned documentation for the anonymous supplementary release."""
    pred = output_logits.argmax(dim=-1)  # [B, T]
    correct = (pred == y_idx).float().mean().item()
    return correct

def model_with_softmax(model, x, dim=2):
    output = model(x)
    return F.softmax(output, dim=dim)


def eval_model(config, model, loader, device):
    ##################################    Eval Loop    #########################
    model.eval()

    with torch.no_grad():
        loss_batch, metric_batch = [], []
        for i, (x, y) in enumerate(tqdm(loader)):
            y = y.view((-1, seq_length)).long().to(device)
            # y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)
            x = x.transpose(1, 2)  # -> (batch, neurons, time)
            y = y.to(device)

            output = model(x)
            output = output.transpose(0, 1)  # -> [B, T, C]
            loss = calc_loss_seq_ce(output, y)
            metric = calc_metric_seq_acc(output, y)

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)


    return np.mean(loss_batch), np.mean(metric_batch)


def train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, num_epochs):
    loss_epochs = {'train': [], 'valid': [], 'test': []}
    metric_epochs = {'train': [], 'valid': [], 'test': []}
    best_metric_val = 0  # 1e6
    best_metric_test = 0  # 1e6
    test_at_best_val = None
    best_loss_val = 1e6

    for epoch in range(num_epochs):

        ##################################    Train Loop    ##############################
        model.train()
        loss_batch, metric_batch = [], []

        for i, (x, y ) in enumerate(tqdm(train_loader)):

            y = y.view((-1, seq_length)).long().to(device)
            # y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)  # (batch, time, neurons) => (512,101,140)
            x = x.transpose(1, 2)  # -> (batch, neurons, time)
            y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output= model(x)
            output = output.transpose(0, 1)  # -> [B, T, C]
            loss = calc_loss_seq_ce(output, y)


            loss.backward()
            optimizer.step()

            metric = calc_metric_seq_acc(output, y)

            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(metric)

            functional.reset_net(model)

        loss_epochs['train'].append(np.mean(loss_batch))
        metric_epochs['train'].append(np.mean(metric_batch))

        scheduler.step()

        # best_model_wts = copy.deepcopy(model.state_dict())

        ##################################    Eval Loop    #########################
        model.eval()
        with torch.no_grad():
            loss_batch, metric_batch = [], []
            for i, (x, y) in enumerate(tqdm(valid_loader)):
                y = y.view((-1, seq_length)).long().to(device)
                # y = F.one_hot(y, config.n_outputs).float()
                x = x.float().to(device)
                x = x.transpose(1, 2)  # -> (batch, neurons, time)
                y = y.to(device)

                output = model(x)
                # output = model_with_softmax(model, x)
                output = output.transpose(0, 1)  # -> [B, T, C]
                loss = calc_loss_seq_ce(output, y)
                metric = calc_metric_seq_acc(output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)

                functional.reset_net(model)

        loss_valid = np.mean(loss_batch)
        metric_valid = np.mean(metric_batch)
        loss_epochs['valid'].append(loss_valid)
        metric_epochs['valid'].append(metric_valid)
        #
        if test_loader:
            loss_test, metric_test = eval_model(config, model, test_loader, device)
        else:
            # could be improved
            loss_test, metric_test = 100, 0
        #
        loss_epochs['test'].append(loss_test)
        metric_epochs['test'].append(metric_test)

        ########################## Logging and Plotting  ##########################


        logger.info(
            f"=====> Epoch {epoch} : Loss Train = {loss_epochs['train'][-1]:.3f}  |  Acc Train = {100 * metric_epochs['train'][-1]:.2f}%")
        logger.info(
            f"Loss Valid = {loss_epochs['valid'][-1]:.3f}  |  Acc Valid = {100 * metric_epochs['valid'][-1]:.2f}%  |  Best Acc Valid = {100 * max(metric_epochs['valid'][-1], best_metric_val):.2f}%")
        if test_loader:
            logger.info(
                f"Loss Test = {loss_epochs['test'][-1]:.3f}  |  Acc Test = {100 * metric_epochs['test'][-1]:.2f}%  |  Best Acc Test = {100 * max(metric_epochs['test'][-1], best_metric_test):.2f}%")

        checkpoint_dir = os.path.join('./checkpoints', config.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        ave_model_path = os.path.join(checkpoint_dir, config.save_model_path)

        if metric_valid > best_metric_val:
            print("# Saving best Metric model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_ACC'))
            best_metric_val = metric_valid
            test_at_best_val = metric_test if test_loader else None

        if loss_valid < best_loss_val:
            print("# Saving best Loss model...")
            torch.save(model.state_dict(),ave_model_path.replace('REPL', 'Best_Loss'))
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

def normalize(data_set,Vmax,Vmin):
    return (data_set-Vmin)/(Vmax-Vmin)

def get_DataLoader(train_x,train_y,batch_size=200):
    train_dataset = data.TensorDataset(torch.Tensor(train_x), torch.Tensor(np.argmax(train_y,axis=-1)))
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

if __name__ == '__main__':

    config = TIMITConfig()
    logger = init_logger(config,"training")
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

    """ dataset """

    train_loader, valid_loader, test_loader, data_meta = build_timit_dataloaders(
        dataset_root=config.datasets_path,
        time_step=config.time_step,
        batch_size=config.batch_size,
    )
    seq_length = data_meta["sequence_length"]
    input_dim = data_meta["input_dim"]
    output_dim = data_meta["output_dim"]


    seeds = config.seeds

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
