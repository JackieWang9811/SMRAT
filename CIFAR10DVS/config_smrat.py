from spikingjelly.activation_based import surrogate
import torch.nn as nn
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datetime

class Config:
    ################################################
    #            General configuration             #
    ################################################
    debug = False

    #  
    dataset = 'cifar10dvs'
    datasets_path = os.environ.get("SMRAT_DATA_ROOT", "./data/CIFAR10DVS")
    log_dir = './logs/logging/cifar10dvs/'
    

    seed = 312
    gpu = 0

    model_type = 'smrat'
    block_type = 'smrat'

    distribute = False

    spike_mode = "lif" # 
    layers = 4
    depths = 4  # M = ceil(log2(16))
    compartments = 2
    kernel_sizes = [2] * compartments
    max_length = None
    dw_kernel_size = 7
    block_stride = 2
    time_step = 16
    n_bins = 5
    warmup_epochs  = 10
    epochs = 300

    route_policy = "topk_sparsity"
    route_target_sparsity = 0.5
    route_topk_fixed_k = max(1, round(route_target_sparsity * compartments))

    if compartments == 2:
        batch_size = 16
    elif compartments == 4:
        batch_size = 14

    dropout_l = 0.1
    

    ############################
    #        USE Module        #
    ############################
    use_bn = True
    use_aug = False
    use_dp = True
    use_dw_bias = False
    


    ############################
    #          Augment         #
    ############################
    mixup = 0.5
    cutmix = 0.
    cutmix_minmax = None
    mixup_prob = 0.5
    mixup_switch_prob = 0.5
    mixup_mode = "batch"
    smoothing = 0.1


    backend = os.environ.get("SMRAT_BACKEND", "cupy")
    attn_mode = 'v2'
    bias = True

    t_max = 40
    lr_w = 0.002
    weight_decay = 0.01 # Default 1e-2 => 5e-4
    n_inputs = 192
    if compartments == 2:
        n_hidden_neurons = 256
    elif compartments == 4:
        n_hidden_neurons = 384

    n_outputs = 10
    pooling_stat = "0111"   # 0111 + 224 + 6 blocks => 23.9G

    num_heads = 16
    loss = 'none'           # 'mean', 'max', 'spike_count', 'sum'
    loss_fn = 'CEloss' # 'SmoothCEloss', 'CEloss'

    init_tau = 2.0 
    v_threshold = 1.0  # LIF
    v_reset = 0.5
    
    surrogate_function = surrogate.Sigmoid(alpha=4.0)
    detach_reset = True
    decay_input = False

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
    run_info = f'||{dataset}||{depths}depths||{time_step}ms||bins={n_bins}||lr_w={lr_w}'
    checkpoint_name = run_name + f'||seed={seed}' + run_info
    save_model_path = f'{checkpoint_name}_REPL_{current_time}.pt'
    make_plot = False
