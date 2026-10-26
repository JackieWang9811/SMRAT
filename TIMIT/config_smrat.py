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
    dataset = 'timit'
    datasets_path = os.environ.get("SMRAT_DATA_ROOT", "./data/TIMIT")
    log_dir = './logs/logging/timit/'


    seed = 312  #  
    seeds = [312, 42, 3112, 0, 10086]
    gpu = 0
    model_type = 'smrat'
    block_type = 'smrat'

    distribute = False

    spike_mode = "lif" #

    layers = 2  # 
    depths = 7  # M = ceil(log2(100))

    kernel_size = 2
    compartments = 4
    kernel_sizes = [2] * compartments
    dw_kernel_size = 31
    time_step = 20
    n_bins = 5
    n_warmup  = 10
    epochs = 300
    max_length = 100
    block_stride = 2

    route_policy = "topk_sparsity"
    route_target_sparsity = 0.5
    route_topk_fixed_k = max(1, round(route_target_sparsity * compartments))

    batch_size = 256
    dropout_l = 0.1
    if compartments == 2:
        n_hidden_neurons = 144 #
    elif compartments == 4:
        n_hidden_neurons = 256 # 

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
    #    TimeNeurons_mask Aug  #
    TN_mask_aug_proba = 0.5
    time_mask_proportion = 0.1
    neuron_mask_size= 10

    backend = os.environ.get("SMRAT_BACKEND", "cupy")
    attn_mode = 'v2'

    bias = True

    t_max = 40
    lr_w = 5e-3 # 5e-3 => 2e-3
    weight_decay = 1e-2 # Default 1e-2 =>5e-3
    n_inputs = 39

    n_outputs = 61

    num_heads = 16
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
    run_info = f'||{dataset}||{depths}depths||{time_step}ms||bins={n_bins}||lr_w={lr_w}'
    checkpoint_name = run_name + f'||seed={seed}' + run_info
    # 
    save_model_path = f'{checkpoint_name}_REPL_{current_time}.pt'
    make_plot = False
