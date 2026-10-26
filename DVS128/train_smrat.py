import datetime
import time
import random
import torch.utils.data
from spikingjelly.datasets import dvs128_gesture
from timm.data import Mixup
from timm.loss import  SoftTargetCrossEntropy
import autoaugment
from torchvision import transforms
from config_smrat import Config as DVS128Config
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.smrat_dvs import SMRAT
from datasets_factory import build_dvs128_dataloaders
import numpy as np
import torch
from utils import init_logger,build_optimizer, count_parameters
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from spikingjelly.activation_based import functional
from datetime import datetime
from uuid import uuid4
import os
import matplotlib.pyplot as plt
eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())



def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    dataset_train = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=True, data_type='frame', frames_number=T,
                                                 split_by='number')
    dataset_test = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=False, data_type='frame', frames_number=T,
                                                split_by='number')
    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler


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

    # probably better to add it in init, or in general do it one time only
    if config.loss_fn == 'CEloss':
        # compare using this to directly using nn.CrossEntropyLoss

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(m, y)

        return loss

def calc_loss_nonspike(config, output, y):
    # probably better to add it in init, or in general do it one time only
    if config.loss_fn == 'CEloss':

        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(output, y)

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
    # 
    if y.ndim > 1:                        # 
        target_idx = y.argmax(dim=-1)     # 
    else:
        target_idx = y                    # 

    # 
    pred_idx = m.argmax(dim=-1)                # (B,)
    acc = (pred_idx == target_idx).float().mean().item()
    return acc




def train_model(config, train_loader, valid_loader, test_loader, device, model, criterion_train, criterion_val, optimizer, scheduler, num_epochs, aug=None, trival_aug=None, mixup_fn=None):

    ##################################    Train Loop    ##############################

    loss_epochs = {'train': [], 'valid': [], 'test': []}
    metric_epochs = {'train': [], 'valid': [], 'test': []}
    best_metric_val = 0  # 1e6
    best_metric_test = 0  # 1e6
    best_loss_val = 1e6

    for epoch in range(num_epochs):

        ##################################    Train Loop    ##############################
        model.train()
        # last element in the tuple corresponds to the collate_fn return
        loss_batch, metric_batch = [], []

        # 
        if mixup_fn is not None and hasattr(mixup_fn, "mixup_enabled"):
            if epoch >= getattr(config, "mixup_off_epoch", 75):
                mixup_fn.mixup_enabled = False

        for i, (image, target) in enumerate(tqdm(train_loader)):

            image = image.float().to(device)
            N, T, C, H, W = image.shape  # C=2, H=128, W=128

            target = target.to(device)

            if aug != None:
                # 
                image = torch.stack([(aug(image[i])) for i in range(N)])
                # image = image.reshape(N,T,C,H,W)

            if trival_aug != None:
                # image = image.flatten(0,1).contiguous()
                image = torch.stack([(trival_aug(image[i])) for i in range(N)])
                # image = image.reshape(N,T,C,H,W).contiguous()

            if mixup_fn is not None:
                # 
                image, target = mixup_fn(image, target)


            # zero the parameter gradients
            optimizer.zero_grad()

            output= model(image)
            if mixup_fn is not None:
                # 
                loss = criterion_train(output, target)  # 
                target_for_acc = target.argmax(dim=-1)  # 
            else:
                loss = criterion_train(output, target)  # 
                target_for_acc = target

            loss.backward()
            optimizer.step()

            functional.reset_net(model)

            acc1 = (output.argmax(1) == target_for_acc).float().mean().item()


            loss_batch.append(loss.detach().cpu().item())
            metric_batch.append(acc1)


        loss_epochs['train'].append(np.mean(loss_batch))
        metric_epochs['train'].append(np.mean(metric_batch))

        scheduler.step()


        ##################################    Eval Loop    #########################
        model.eval()
        with torch.no_grad():
            loss_batch, metric_batch = [], []
            for i, (image, target) in enumerate(tqdm(valid_loader)):
                image = image.float().to(device)  # (batch, time, neurons) => (512,101,140)
                N, T, C, H, W = image.shape

                target = target.to(device)

                output = model(image)
                loss = criterion_val(output, target)
                # acc1 = calc_metric(config, output, target)
                acc1 = (output.argmax(1) == target).float().mean().item()

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(acc1)

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
        plt.savefig(f'Accuracy_{current_time}.png')
        plt.show()





if __name__ == '__main__':


    config = DVS128Config()

    logger = init_logger(config,"training")
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
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    hidden_dim = config.n_hidden_neurons
    logger.info("##############################################\n")
    logger.info("Seed :{}".format(seed_val))
    epochs = config.epochs
    logger.info("The epoch is: {}".format(epochs))
    logger.info("The batch size is : {}".format(config.batch_size))
    logger.info("The backend is: {}".format(config.backend))
    logger.info("The mixup_off_epoch is: {}".format(config.mixup_off_epoch))
    """ dataset """
    data_path = config.datasets_path
    train_loader, valid_loader, train_sampler, val_sampler = build_dvs128_dataloaders(
        dataset_dir=data_path,
        batch_size=config.batch_size,
        time_step=config.time_step,
        distributed=False,
        num_workers=4,
    )

    test_loader = None

    data_loader_test = None

    train_snn_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5)
                    ])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = config.mixup > 0 or config.cutmix > 0. or config.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=config.mixup, cutmix_alpha=config.cutmix, cutmix_minmax=config.cutmix_minmax,
            prob=config.mixup_prob, switch_prob=config.mixup_switch_prob, mode=config.mixup_mode,
            label_smoothing=config.smoothing, num_classes=config.n_outputs)
        mixup_fn = Mixup(**mixup_args)

    logger.info("The time_step is: {}".format(config.time_step))
    logger.info("The decay_input is: {}".format(config.decay_input))
    logger.info("The hidden_dim is: {}".format(config.n_hidden_neurons))
    logger.info("The spike_mode is: {}".format(config.spike_mode))
    logger.info("The block_mode is :{}".format(config.block_type))
    logger.info("The kernel_size is: {}".format(config.kernel_sizes[0]))
    logger.info("The compartments is: {}".format(config.compartments))
    logger.info("The layers of SMRAT is: {}".format(config.layers))
    logger.info("The depths in SMRAT is: {}".format(config.depths))

    model = SMRAT(
        config,
        d_output=config.n_outputs,
        d_model=config.n_hidden_neurons,
        n_layers=config.layers,
        encoder="dvs", # speech
        n_tokens=None,
        max_length=config.max_length,
        depth=config.depths
    ).to(device)

    for i, blk in enumerate(model.mixing_layers):
        gate = blk.multires_gate
        ds = [gate._dilation(l) for l in range(gate.depth)]
        print(i, ds)

    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    dataset_info = config.dataset
    folder_path = os.path.join('model_structure', dataset_info)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f'model_structure_{dataset_info}_{formatted_time}.txt')

    with open(filename, 'w') as f:
        print(model, file=f)

    print(f"===> Dataset    = {config.dataset}")
    print(f"===> Model type = {config.model_type}")
    print(f"===> Model size = {count_parameters(model)}\n\n")
    logger.info("Model size:{}".format(count_parameters(model)))
    lr_w = config.lr_w
    logger.info("lr_w: {}".format(lr_w))

    weight_decay = config.weight_decay
    logger.info("weight_decay: {}".format(weight_decay))
    optimizer = build_optimizer(config, model)
    T = config.t_max
    logger.info("T:{}".format(T))

    warmup_epochs = getattr(config, "warmup_epochs", 10)
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs),
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )

    if mixup_active:
        criterion_train = SoftTargetCrossEntropy()
    else:
        criterion_train = nn.CrossEntropyLoss()

    criterion_val = nn.CrossEntropyLoss()



    train_model(config, train_loader, valid_loader, test_loader, device, model, criterion_train, criterion_val, optimizer, scheduler, epochs, aug=train_snn_aug, trival_aug=train_trivalaug, mixup_fn=mixup_fn)
