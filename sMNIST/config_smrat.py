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
    dataset = 'sMNIST'
    datasets_path = os.environ.get("SMRAT_DATA_ROOT", "./data")
    log_dir = './logs/logging/smnist/'
    

    seed = 312  #  

    gpu = 0
    model_type = 'smrat'
    block_type = 'smrat'

    distribute = False

    spike_mode = "lif" # lif  ssmlif_ptc rfn
    warmup_epochs  = 10

    layers = 3
    depths = 10  # M = ceil(log2(784))
    block_stride = 2
    compartments = 2
    kernel_sizes = [2] * compartments
    dw_kernel_size = 31
    bias = True
    batch_size = 100
    epochs = 150
    dropout_l = 0.1
    n_inputs = 1
    if compartments == 2:
        n_hidden_neurons = 256
    elif compartments == 4:
        n_hidden_neurons = 400

    route_policy = "topk_sparsity"
    route_target_sparsity = 0.5
    route_topk_fixed_k = max(1, round(route_target_sparsity * compartments))

    n_outputs = 10
    max_length = 784

    backend = os.environ.get("SMRAT_BACKEND", "cupy")
    attn_mode = 'v2'
    bias = True

    time_step = 10
    t_max = 40
    lr_w = 5e-3
    weight_decay = 1e-2

    use_aug = False
    use_dw_bias = True
    use_dp = True
    use_bn = True

    loss = 'none'           # 'mean', 'max', 'spike_count', 'sum'
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
    run_info = f'||{dataset}||{depths}depths||lr_w={lr_w}'
    checkpoint_name = run_name + f'||seed={seed}' + run_info
    # 
    save_model_path = f'{checkpoint_name}_REPL_{current_time}.pt'
    make_plot = False
