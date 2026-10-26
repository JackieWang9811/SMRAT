from spikingjelly.activation_based import surrogate
import torch.nn as nn
import datetime
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Config:
    ################################################
    #            General configuration             #
    ################################################
    debug = False
    #  
    dataset = 'shd'
    datasets_path = os.environ.get("SMRAT_DATA_ROOT", "./data/SHD")
    log_dir = './logs/logging/shd/'
    

    seed = 312 #  
    seeds = [312, 42, 3112, 0, 10086]
    # seeds = [3112]
    gpu = 0
    model_type = 'smrat'
    block_type = 'smrat'
    distribute = False
    spike_mode = "lif"
    time_step = 10
    max_length = 1000 // time_step

    compartments = 4
    kernel_sizes = [2] * compartments
    dw_kernel_size = 31
    layers = 2
    depths = 7  # M = ceil(log2(100))
    bias = True
    block_stride = 2


    route_policy = "topk_sparsity"
    route_target_sparsity = 0.5
    route_topk_fixed_k = max(1, round(route_target_sparsity * compartments))


    n_bins = 5
    epochs = 500
    n_warmup = 10
    batch_size = 256 # 128 => 256 => 512
    dropout_l = 0.1
    ############################
    #        USE Module        #
    ############################

    use_bn = True
    use_aug = True
    use_dp = True
    use_dw_bias = False

    ############################
    #          Augment         #
    ############################s
    #    TimeNeurons_mask Aug  #
    TN_mask_aug_proba = 0.5
    time_mask_proportion = 0.2
    neuron_mask_size= 20

    backend = os.environ.get("SMRAT_BACKEND", "cupy")


    t_max = 40 # 40
    lr_w = 0.01
    weight_decay = 0.01
    n_inputs = 700//n_bins
    if compartments == 2:
        n_hidden_neurons = 96
    elif compartments == 4:
        n_hidden_neurons = 128
    n_outputs = 20 if dataset == 'shd' else 35

    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum'
    loss_fn = 'CEloss'

    init_tau = 2.0
    v_threshold = 1.0 # LIF
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
