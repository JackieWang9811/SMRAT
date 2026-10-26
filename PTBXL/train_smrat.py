from config_smrat import Config as PTBXLConfig
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.smrat import SMRAT

from torch.utils.data import Dataset
import utils
import numpy as np
import torch
from utils import init_logger,build_optimizer
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
from spikingjelly.activation_based import functional
from datetime import datetime
from uuid import uuid4
import os
import matplotlib.pyplot as plt

eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
from datasets_factory import build_ptbxl_dataloaders
from sklearn.metrics import roc_auc_score
import json
from spikingjelly.activation_based import surrogate

def calc_loss(config, output, y):

    m = temporal_reduce(config, output )  # [B, C] logits
    # m = output
    BCEloss = nn.BCEWithLogitsLoss()
    loss = BCEloss(m, y)

    return m, loss


def temporal_reduce(config, output):
    # mean accuracy over batch
    if config.loss == 'mean': m = torch.mean(output, 0)
    elif config.loss == 'max': m, _ = torch.max(output, 0)
    elif config.loss == 'spike_count': m = torch.sum(output, 0)
    elif config.loss == 'sum':
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)
    elif config.loss == 'none':
        m = output
    return m


def update_auc_buffers(config, output, y, true_buf, score_buf):
    # [B, C] logits
    true_buf.append(y.detach().to('cpu').numpy())
    score_buf.append(output.detach().to('cpu').numpy())   # 

def finalize_auc(true_buf, score_buf):
    y_true = np.concatenate(true_buf, axis=0)   # [N, C]
    y_score = np.concatenate(score_buf, axis=0) # [N, C]
    auroc = roc_auc_score(y_true, y_score, average="macro")

    return auroc

def eval_model(config, model, loader, device):
    ##################################    Eval Loop    #########################
    model.eval()
    true_buf_val, score_buf_val = [], []
    with torch.no_grad():
        loss_batch, metric_batch = [], []
        for i, (x, y) in enumerate(tqdm(loader)):

            x = x.float().to(device)
            y = y.to(device)
            output = model(x)

            output, loss = calc_loss(config, output, y)
            update_auc_buffers(config, output, y, true_buf_val, score_buf_val)

            loss_batch.append(loss.detach().cpu().item())

            functional.reset_net(model)


    loss_valid = np.mean(loss_batch)
    metric_valid = finalize_auc(true_buf_val, score_buf_val)
    return loss_valid, metric_valid



def train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, num_epochs):

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
        true_buf_train, score_buf_train = [], []
        for i, (x, y) in enumerate(tqdm(train_loader)):

            # y = y.view((-1, seq_length)).long().to(device)
            # y = F.one_hot(y, config.n_outputs).float()
            x = x.float().to(device)   # (batch, channel, patch, sampling_points)
            y = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            output= model(x)

            output, loss = calc_loss(config, output, y)
            update_auc_buffers(config, output, y, true_buf_train, score_buf_train)

            loss.backward()
            optimizer.step()



            loss_batch.append(loss.detach().cpu().item())

            functional.reset_net(model)

        loss_train = np.mean(loss_batch)
        metric_train = finalize_auc(true_buf_train, score_buf_train)

        loss_epochs['train'].append(loss_train)
        metric_epochs['train'].append(metric_train)

        scheduler.step()

        # best_model_wts = copy.deepcopy(model.state_dict())

        ##################################    Eval Loop    #########################
        model.eval()
        true_buf_val, score_buf_val = [], []
        with torch.no_grad():
            loss_batch, metric_batch = [], []
            for i, (x, y) in enumerate(tqdm(valid_loader)):
                # y = F.one_hot(y, config.n_outputs).float()
                x = x.float().to(device)
                y = y.to(device)
                output = model(x)

                output, loss = calc_loss(config, output, y)
                update_auc_buffers(config, output, y, true_buf_val, score_buf_val)

                loss_batch.append(loss.detach().cpu().item())

                functional.reset_net(model)

        loss_valid = np.mean(loss_batch)
        metric_valid = finalize_auc(true_buf_val, score_buf_val)

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
            best_metric_test = metric_test if test_loader else 0

        if loss_valid < best_loss_val:
            print("# Saving best Loss model...")
            torch.save(model.state_dict(),ave_model_path.replace('REPL', 'Best_Loss'))
            best_loss_val = loss_valid

        # Test scores are never used for checkpoint selection.

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

class EEGDataset(Dataset):
    def __init__(self, data_path, label_path, scale_factor=100.0):
        self.data = np.load(data_path, allow_pickle=True).astype(np.float32)
        self.labels = np.load(label_path, allow_pickle=True).astype(np.int64)
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx] / self.scale_factor
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y)




if __name__ == '__main__':


    # config = SHDTConfig()
    # config = GSCTConfig()
    config = PTBXLConfig()

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
    logger.info("The task is: {}".format(config.task))
    logger.info("Data augmentation enabled: {}".format(config.use_aug))


    """ dataset """
    train_loader, valid_loader, test_loader, meta = build_ptbxl_dataloaders(
        data_dir=config.datasets_path,
        task=config.task,  # e.g., "superdiagnostic"
        sampling_rate=100,  # 
        min_samples=0,
        train_fold=8, val_fold=9, test_fold=10,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    # 
    config.n_outputs = meta["d_output"]
    config.n_inputs = meta["d_input"]
    config.seq_len = meta["L"]
    NEED_REARRANGE = meta["needs_rearrange"]  # False

    

    logger.info("The time_step is: {}".format(config.time_step))
    logger.info("The n_bins is: {}".format(config.n_bins))
    logger.info("The hidden_dim is: {}".format(config.n_hidden_neurons))
    logger.info("The spike_mode is: {}".format(config.spike_mode))
    logger.info("The block_mode is :{}".format(config.block_type))
    logger.info("The kernel_size is: {}".format(config.kernel_sizes[0]))
    logger.info("The compartments is: {}".format(config.compartments))
    logger.info("The layers of SMRAT is: {}".format(config.layers))
    logger.info("The depths in SMRAT is: {}".format(config.depths))

    # model = SpikingMultiresNet(
    #     config,
    #     d_input=config.n_inputs,
    #     d_output=config.n_outputs,
    #     d_model=config.n_hidden_neurons,
    #     n_layers=config.layers,
    #     dropout=config.dropout_l,
    #     encoder="linear",
    #     n_tokens=None,
    #     layer_type='multires',
    #     max_length=config.max_length,
    #     hinit=None,
    #     depth=config.depths,
    #     tree_select="fading",
    #     d_mem=None,
    #     kernel_size=config.kernel_size,
    #     indep_res_init=False,
    # ).to(device)

    model = SMRAT(
        config,
        d_output=config.n_outputs,
        d_model=config.n_hidden_neurons,
        n_layers=config.layers,
        encoder="speech",
        # encoder="sequential",
        n_tokens=None,
        max_length=config.max_length,
        depth=config.depths,
    ).to(device)


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

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

    warmup_epochs = config.warmup_epochs
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs),
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )


    train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, epochs)
