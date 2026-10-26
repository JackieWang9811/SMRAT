from spikingjelly.activation_based import surrogate
import torch.nn as nn
import datetime
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Config:
    ################################################
    #            General configuration             #
    ################################################
    debug = False

    #  
    dataset = 'gsc'
    datasets_path = os.environ.get("SMRAT_DATA_ROOT", "./data/GSC")
    log_dir = './logs/logging/gsc/'
    

    seed = 312 #
    seeds = [312, 42, 3112, 0, 10086]

    gpu = 0
    model_type = 'smrat'
    block_type = 'smrat'

    dcls_share_weights =True
    distribute = False

    spike_mode = "lif" # lif  ssmlif_ptc rfn

    # Spectrogram
    window_size = 256
    hop_length = 100
    n_mels = 140
    max_length = 100

    layers = 4 # 
    depths = 7  # M = ceil(log2(100))
    block_stride = 2

    kernel_size = 2
    compartments = 2
    kernel_sizes = [2] * compartments
    dw_kernel_size = 31


    route_policy = "topk_sparsity"
    route_target_sparsity = 0.5
    route_topk_fixed_k = max(1, round(route_target_sparsity * compartments))

    bias = True
    batch_size = 256
    epochs = 300
    dropout_l = 0.1
    if compartments == 2:
        n_hidden_neurons = 256
    elif compartments == 4:
        n_hidden_neurons = 384


    ############################
    #        USE Module        #
    ############################
    use_bn = True
    use_aug = True
    use_dp = True
    use_dw_bias = False

    ############################
    #          Augment         #
    ############################
    #  SpecAugment #
    mF = 1
    F = 10
    mT = 1
    pS = 0.25

    backend = os.environ.get("SMRAT_BACKEND", "cupy")
    attn_mode = 'v2'
    bias = True

    n_warmup = 0
    t_max = 40
    lr_w = 2e-3 # 2e-3
    weight_decay = 5e-3 # 5e-3

    n_inputs = n_mels

    n_outputs = 20 if dataset == 'shd' else 35

    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum'
    loss_fn = 'CEloss' # 'SmoothCEloss', 'CEloss'

    init_tau = 2.0 
    v_threshold = 1.0  # LIF
    v_reset = 0.5
    alpha = 5.0 # Atan = 5.0 Rectangle=0.5
    surrogate_function = surrogate.ATan(alpha = alpha)
    detach_reset = True


    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adamw'
    optimizer_pos = 'adamw'


    ################################################
    #                    Save                      #
    ################################################
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = model_type
    run_info = f'||{dataset}||{depths}depths||{window_size}ms||bins={hop_length}||lr_w={lr_w}'
    checkpoint_name = run_name + f'||seed={seed}' + run_info
    # 
    save_model_path = f'{checkpoint_name}_REPL_{current_time}.pt'
    make_plot = False
