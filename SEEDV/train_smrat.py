from config_smrat import Config as SEEDVConfig
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.smrat import SMRAT
from datasets_factory import build_seedv_dataloaders
from torch.utils.data import Dataset, DataLoader
import utils
import numpy as np
import torch
from utils import init_logger,build_optimizer
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from spikingjelly.activation_based import functional
from datetime import datetime
from uuid import uuid4
import os
import matplotlib.pyplot as plt
from einops import rearrange
eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score,\
    precision_recall_curve, auc, r2_score, mean_squared_error
import json

def reduce_to_batch_logits(config, output):
    """Cleaned documentation for the anonymous supplementary release."""
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
    else:
        raise ValueError(f"Unknown reduction mode for config.loss = {config.loss}")
    return m  # [B, C]


def calc_loss(output, y):

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

    return np.mean((torch.max(y,1)[1]==torch.max(m,1)[1]).detach().cpu().numpy())


@torch.no_grad()
def get_metrics_for_multiclass(config, model, loader, device):
    model.eval()
    truths, preds = [], []

    for x, y in loader:
        # 
        x = x.float().to(device)                 # (B, C, S, T)
        x = rearrange(x, 'b c s t -> b (c s) t') # 
        output = model(x)                        # 
        m = reduce_to_batch_logits(config, output)  # [B, C]
        pred_y = torch.argmax(m, dim=-1)

        truths.extend(y.cpu().numpy().tolist())
        preds.extend(pred_y.cpu().numpy().tolist())

        functional.reset_net(model)

    truths = np.array(truths)
    preds = np.array(preds)

    metrics = {
        "acc": accuracy_score(truths, preds),                      # 
        "bacc": balanced_accuracy_score(truths, preds),            # 
        "f1_weighted": f1_score(truths, preds, average='weighted'),
        "kappa": cohen_kappa_score(truths, preds),
        "cm": confusion_matrix(truths, preds),                     # 
    }
    return metrics

def eval_model(config, model, loader, device):
    ##################################    Eval Loop    #########################
    model.eval()
    truths, preds = [], []
    with torch.no_grad():
        loss_batch, metric_batch = [], []
        for i, (x, y) in enumerate(tqdm(loader)):
            # y_oh = F.one_hot(y, config.n_outputs).float().to(device)
            y = y.to(device)  # 
            x = x.float().to(device)
            x = rearrange(x, 'b c s t -> b (c s) t')

            output = model(x)
            m = reduce_to_batch_logits(config, output)  # [B, C]
            pred_y = torch.argmax(m, dim=-1)

            truths.extend(y.cpu().numpy().tolist())
            preds.extend(pred_y.cpu().numpy().tolist())

            loss = calc_loss(m, y)

            loss_batch.append(loss.detach().cpu().item())

            functional.reset_net(model)

        truths = np.array(truths)
        preds = np.array(preds)

        detail = {
            "acc": accuracy_score(truths, preds),  # 
            "bacc": balanced_accuracy_score(truths, preds),  # 
            "f1_weighted": f1_score(truths, preds, average='weighted'),
            "kappa": cohen_kappa_score(truths, preds),
            "cm": confusion_matrix(truths, preds),  # 
        }

        loss_valid = np.mean(loss_batch)
        return loss_valid, detail


def train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, num_epochs):

    ##################################    Train Loop    ##############################

    loss_epochs = {'train': [], 'valid': [], 'test': []}
    best_metric_val = 0  # 1e6
    best_loss_val = 1e6

    best_test_acc = 0
    best_test_detail = None
    best_test_detail_str = None   # 


    for epoch in range(num_epochs):
        ################################## Train ##################################
        model.train()
        # last element in the tuple corresponds to the collate_fn return
        loss_batch, metric_batch = [], []
        truths, preds = [], []
        for i, (x, y) in enumerate(tqdm(train_loader)):
            y = y.to(device)
            x = x.float().to(device)
            x = rearrange(x, 'b c s t -> b (c s) t')

            optimizer.zero_grad()
            output = model(x)

            m = reduce_to_batch_logits(config, output)  # [B, C]
            pred_y = torch.argmax(m, dim=-1)

            truths.extend(y.cpu().numpy().tolist())
            preds.extend(pred_y.cpu().numpy().tolist())

            loss = calc_loss(m, y)

            loss.backward()
            optimizer.step()

            loss_batch.append(loss.detach().cpu().item())

            functional.reset_net(model)

        train_loss = float(np.mean(loss_batch))
        loss_epochs['train'].append(train_loss)

        truths = np.array(truths)
        preds = np.array(preds)

        train_detail = {
            "acc": accuracy_score(truths, preds),  # 
            "bacc": balanced_accuracy_score(truths, preds),  # 
            "f1_weighted": f1_score(truths, preds, average='weighted'),
            "kappa": cohen_kappa_score(truths, preds),
            "cm": confusion_matrix(truths, preds),  # 
        }

        scheduler.step()

        ################################## Valid ##################################
        model.eval()

        loss_valid, val_detail = eval_model(config, model, valid_loader, device)

        ################################## Test ##################################
        if test_loader:
            loss_test, test_detail = eval_model(config, model, test_loader, device)
        else:
            loss_test, metric_test = 100.0, 0.0
            test_detail = None

        loss_epochs['test'].append(loss_test)

        ################################## Logging ##################################

        logger.info(f"=====> Epoch {epoch} : Loss Train = {train_loss:.3f}")
        logger.info("[Train Metrics] Acc={:.2f}% | BAcc={:.2f}% | F1(w)={:.4f} | Kappa={:.4f}".format(
            100 * train_detail["acc"], 100 * train_detail["bacc"],
            train_detail["f1_weighted"], train_detail["kappa"]
        ))
        # logger.info("[Train ConfMat]\n{}".format(train_detail["cm"]))

        logger.info(f"Loss Valid = {loss_valid:.3f}")
        logger.info("[Valid Metrics] Acc={:.2f}% | BAcc={:.2f}% | F1(w)={:.4f} | Kappa={:.4f}".format(
            100 * val_detail["acc"], 100 * val_detail["bacc"],
            val_detail["f1_weighted"], val_detail["kappa"]
        ))
        # logger.info("[Valid ConfMat]\n{}".format(val_detail["cm"]))

        if test_loader:
            logger.info(f"Loss Test = {loss_test:.3f}")
            logger.info("[Test  Metrics] Acc={:.2f}% | BAcc={:.2f}% | F1(w)={:.4f} | Kappa={:.4f}".format(
                100 * test_detail["acc"], 100 * test_detail["bacc"],
                test_detail["f1_weighted"], test_detail["kappa"]
            ))
            # logger.info("[Test  ConfMat]\n{}".format(test_detail["cm"]))

            if val_detail["acc"] > best_metric_val:
                best_test_acc = test_detail["acc"]
                best_test_detail = test_detail
                best_test_detail_str = (
                    "[Test  Metrics] Acc={:.2f}% | BAcc={:.2f}% | F1(w)={:.4f} | Kappa={:.4f}".format(
                        100 * test_detail["acc"], 100 * test_detail["bacc"],
                        test_detail["f1_weighted"], test_detail["kappa"]
                    )
                )
        # 
        if best_test_detail_str is not None:
            logger.info("[Test at Best Validation] " + best_test_detail_str)


        ################################## Checkpoints ##################################
        checkpoint_dir = os.path.join('./checkpoints', config.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ave_model_path = os.path.join(checkpoint_dir, config.save_model_path)

        if val_detail["acc"] > best_metric_val:
            print("# Saving best Metric model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_ACC'))
            best_metric_val = val_detail["acc"]

        if loss_valid < best_loss_val:
            print("# Saving best Loss model...")
            torch.save(model.state_dict(), ave_model_path.replace('REPL', 'Best_Loss'))
            best_loss_val = loss_valid

    return {
        "best_test": best_test_detail,  # dict or None
        "best_test_acc": best_test_acc
    }



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


def get_dataloaders(dataset_root, batch_size=64, num_workers=4, scale_factor=100.0):
    # dataset_root = os.path.join(DATA_ROOT, "BCIC2020", "processed_npy")

    train_dataset = EEGDataset(
        os.path.join(dataset_root, "eeg_train.npy"),
        os.path.join(dataset_root, "labels_train.npy"),
        scale_factor
    )
    val_dataset = EEGDataset(
        os.path.join(dataset_root, "eeg_val.npy"),
        os.path.join(dataset_root, "labels_val.npy"),
        scale_factor
    )
    test_dataset = EEGDataset(
        os.path.join(dataset_root, "eeg_test.npy"),
        os.path.join(dataset_root, "labels_test.npy"),
        scale_factor
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    config = SEEDVConfig()

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
    seeds = config.seed
    if isinstance(seeds, int):
        seeds = [seeds]
    elif isinstance(seeds, (tuple, list)):
        seeds = list(seeds)
    else:
        raise TypeError(f"config.seed should be int or list/tuple, got {type(seeds)}")

    all_seed_results = []  # 

    for seed in seeds:

        seed_val = seed
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



        """ dataset """
        train_loader, valid_loader, test_loader = build_seedv_dataloaders(dataset_root=config.datasets_path, batch_size=config.batch_size)

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
            encoder="eeg",
            n_tokens=None,
            max_length=config.max_length,
            depth=config.depths,
            is_sequential=False
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

        # 
        warmup_epochs = getattr(config, "warmup_epochs", 10)
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs),
                lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
            ],
            milestones=[warmup_epochs]
        )

        out = train_model(config, train_loader, valid_loader, test_loader, device, model, optimizer, scheduler, epochs)
        best_test = out["best_test"]
        if best_test is None:
            logger.warning(f"Seed {seed_val}: best_test is None (no test_loader?)")
        else:
            all_seed_results.append({
                "seed": seed_val,
                "acc": best_test["acc"],
                "bacc": best_test["bacc"],
                "f1_weighted": best_test["f1_weighted"],
                "kappa": best_test["kappa"],
            })

    if len(all_seed_results) == 0:
        logger.warning("No test results collected.")
    elif len(all_seed_results) == 1:
        r = all_seed_results[0]
        logger.info(f"[Final Single-Seed] seed={r['seed']} "
                    f"Acc={100 * r['acc']:.2f}% | BAcc={100 * r['bacc']:.2f}% | "
                    f"F1(w)={r['f1_weighted']:.4f} | Kappa={r['kappa']:.4f}")
    else:
        def _mean_std(key):
            arr = np.array([x[key] for x in all_seed_results], dtype=np.float64)
            return arr.mean(), arr.std(ddof=1)  # 


        acc_m, acc_s = _mean_std("acc")
        bacc_m, bacc_s = _mean_std("bacc")
        f1_m, f1_s = _mean_std("f1_weighted")
        k_m, k_s = _mean_std("kappa")

        logger.info(f"[Final Multi-Seed] N={len(all_seed_results)} seeds={[x['seed'] for x in all_seed_results]}")
        logger.info("Acc   = {:.2f} +/- {:.2f} (%)".format(100 * acc_m, 100 * acc_s))
        logger.info("BAcc  = {:.2f} +/- {:.2f} (%)".format(100 * bacc_m, 100 * bacc_s))
        logger.info("F1(w) = {:.4f} +/- {:.4f}".format(f1_m, f1_s))
        logger.info("Kappa = {:.4f} +/- {:.4f}".format(k_m, k_s))
