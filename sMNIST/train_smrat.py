from config_smrat import Config as SMNISTConfig
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models.smrat import SMRAT
from datasets_factory import build_mnist_sequence_dataloaders
import torch.nn as nn
import random
from torch.optim import lr_scheduler
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import os
import matplotlib.pyplot as plt
eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import utils
from utils import init_logger,build_optimizer
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.neuron import *
import json
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

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

def calc_metric(config, output, y):
    # mean accuracy over batch
    if config.loss == 'mean': m = torch.mean(output, 0)
    elif config.loss == 'max': m, _ = torch.max(output, 0)
    elif config.loss == 'spike_count': m = torch.sum(output, 0)
    elif config.loss == 'sum':
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)
    elif config.loss == 'none':
        m = output
    return np.mean((torch.max(y,1)[1]==torch.max(m,1)[1]).detach().cpu().numpy())


def setSeed(seed):
    """ Set all seeds to ensure reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def disp(img):
    """ Displays an image """
    if len(img.shape) == 3:
        img = img.squeeze(0)
    plt.imshow(img, cmap = "gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def dispSeq(seq, rows = 8):
    """ Displays a sequence of pixels """
    seq = seq.reshape(rows, -1) # divide the 1D sequence into `rows` rows for easy visualization
    disp(seq)

def countParameters(model):
    """ Counts and prints the number of trainable and non-trainable parameters of a model """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters"

class psMNIST(Dataset):
    """ Dataset that defines the psMNIST dataset, given the MNIST data and a fixed permutation """

    def __init__(self, mnist, perm):
        self.mnist = mnist # also a torch.data.Dataset object
        self.perm  = perm

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        unrolled = img.reshape(-1)
        permuted = unrolled[self.perm]
        permuted = permuted.reshape(-1, 1)
        return permuted, label



def eval_model(config, model, loader, device):
    model.eval()
    losses, metrics = [], []
    with torch.no_grad():
        for x, y in tqdm(loader):
            y = F.one_hot(y, config.n_outputs).float().to(device)
            x = x.float().to(device).transpose(1, 2)
            output = model(x)
            losses.append(calc_loss(config, output, y).detach().cpu().item())
            metrics.append(calc_metric(config, output, y))
            functional.reset_net(model)
    return float(np.mean(losses)), float(np.mean(metrics))


def train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, num_epochs):

    ##################################    Train Loop    ##############################

    loss_epochs = {'train': [], 'valid': [], 'test': []}
    metric_epochs = {'train': [], 'valid': [], 'test': []}
    best_metric_val = 0  # 1e6
    best_metric_test = 0  # 1e6
    best_loss_val = 1e6
    # calc_loss_std = SoftTargetCrossEntropy()

    scaler = GradScaler(enabled=True)

    for epoch in range(num_epochs):

        ##################################    Train Loop    ##############################
        model.train()
        # last element in the tuple corresponds to the collate_fn return
        loss_batch, metric_batch = [], []

        for i, (x, y) in enumerate(tqdm(train_loader)):

            y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)  # (batch, time, neurons)
            x = x.transpose(1, 2)
            y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # output= model(x)
            # loss = calc_loss(config, output, y)
            #
            # loss.backward()
            # optimizer.step()

            # --- AMP: autocast ---
            with autocast():
                output = model(x)
                loss = calc_loss(config, output, y)

            # --- AMP: scaled backward + step ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


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
            for i, (x, y) in enumerate(tqdm(valid_loader)):

                y = F.one_hot(y, config.n_outputs).float()
                x = x.float().to(device)
                x = x.transpose(1, 2)
                y = y.to(device)

                # output = model(x)
                # loss = calc_loss(config, output, y)
                # --- AMP: autocast ---
                with autocast():
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
        checkpoint_dir = os.path.join('./checkpoints', config.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        ave_model_path = os.path.join(checkpoint_dir, config.save_model_path)

        if metric_valid > best_metric_val:
            print("# Saving best Metric model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_ACC'))
            best_metric_val = metric_valid

        if loss_valid < best_loss_val:
            print("# Saving best Loss model...")
            torch.save(model.state_dict(),ave_model_path.replace('REPL', 'Best_Loss'))
            best_loss_val = loss_valid

    if test_loader:
        best_path = ave_model_path.replace('REPL', 'Best_ACC')
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        loss_test, metric_test = eval_model(config, model, test_loader, device)
        loss_epochs['test'].append(loss_test)
        metric_epochs['test'].append(metric_test)
        logger.info(
            f"Final test at best validation checkpoint: Loss={loss_test:.3f} | Acc={100 * metric_test:.2f}%"
        )

    ###### make_plot ######
    train_acc = [x * 100 for x in metric_epochs['train']]
    valid_acc = [x * 100 for x in metric_epochs['valid']]
    test_acc = [x * 100 for x in metric_epochs['test']]

    epochs = range(1, len(train_acc) + 1)
    if config.make_plot:

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_acc, '-o', label='Training Accuracy', color='g')
        plt.plot(epochs, valid_acc, '-^', label='Validation Accuracy', color='b')
        plt.plot(epochs, test_acc, '-s', label='Test Accuracy', color='y')

        plt.title('Training, Validation, and Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()


        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f%%'))

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 
        plt.savefig(f'Accuracy_{current_time}.png')
        plt.show()


if __name__ == '__main__':
    config = SMNISTConfig()

    logger = init_logger(config, "training")
    logger.info("Logger is properly initialized and ready to use.")
    logger.info("The GPU is {}".format(config.gpu))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = f"cuda:{config.gpu}"
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    print()
    print(f"\n=====> Device = {device} \n\n")

    ''' set random seeds '''
    seed_val = config.seed
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # config.attention_window = attention_window

    hidden_dim = config.n_hidden_neurons
    logger.info("##############################################\n")
    logger.info("Seed :{}".format(seed_val))
    epochs = config.epochs
    logger.info("The epoch is: {}".format(epochs))
    logger.info("The batch size is : {}".format(config.batch_size))
    logger.info("The backend is: {}".format(config.backend))

    dl_train, dl_val, dl_test = build_mnist_sequence_dataloaders(
        root=config.datasets_path,
        batch_size=config.batch_size,
        permuted=False,
        seed=seed_val,
        num_workers=4,
    )


    logger.info("The hidden_dim is: {}".format(config.n_hidden_neurons))
    logger.info("The spike_mode is: {}".format(config.spike_mode))
    logger.info("The block_mode is :{}".format(config.block_type))
    logger.info("The dropout is: {}".format(config.dropout_l))
    logger.info("The kernel_size is: {}".format(config.kernel_sizes[0]))
    logger.info("The compartments is: {}".format(config.compartments))
    logger.info("The layers of SMRAT is: {}".format(config.layers))
    logger.info("The depths in SMRAT is: {}".format(config.depths))


    model = SMRAT(
        config,
        d_output=config.n_outputs,
        d_model=config.n_hidden_neurons,
        n_layers=config.layers,
        encoder="sequential",
        n_tokens=None,
        max_length=config.max_length,
        depth=config.depths,
        is_sequential=True
    ).to(device)

    print(model)
    countParameters(model)

    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    dataset_info = config.dataset
    folder_path = os.path.join('model_structure', dataset_info)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f'model_structure_{dataset_info}_{formatted_time}.txt')

    with open(filename, 'w') as f:
        print(model, file=f)


    # 
    cfg_dir = os.path.join('config_recording', dataset_info)
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_txt = os.path.join(cfg_dir, f'config_{dataset_info}_{formatted_time}.txt')

    def _extract_config_dict(cfg_obj):
        """Extract serializable config fields for logging."""
        cfg_dict = {}
        for k in dir(cfg_obj):
            if k.startswith('_'):
                continue
            v = getattr(cfg_obj, k)
            if callable(v):
                continue
            # 
            try:
                json.dumps(v)
                cfg_dict[k] = v
            except TypeError:
                cfg_dict[k] = str(v)
        return dict(sorted(cfg_dict.items(), key=lambda x: x[0]))


    # txt
    _cfg = _extract_config_dict(config)
    with open(cfg_txt, 'w', encoding='utf-8') as f:
        f.write("# ===== Config Snapshot =====\n")
        for k, v in _cfg.items():
            f.write(f"{k}: {v}\n")


    print(f"===> Dataset    = {config.dataset}")
    print(f"===> Model type = {config.model_type}")
    print(f"===> Model size = {utils.count_parameters(model)}\n\n")
    logger.info("Model size:{}".format(utils.count_parameters(model)))
    lr_w = config.lr_w
    logger.info("lr_w: {}".format(lr_w))
    weight_decay = config.weight_decay
    logger.info("weight_decay: {}".format(weight_decay))
    optimizer = build_optimizer(config, model)
    T = config.t_max
    logger.info("T:{}".format(T))

    warmup_epochs = config.warmup_epochs
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs),
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )


    train_model(config, dl_train, dl_val, dl_test, device, model, optimizer, scheduler, epochs)

