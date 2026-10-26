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
    dataset = 'ptbxl'
    task = 'diagnostic' # "rhythm", "form", "diagnostic", "subdiagnostic", "superdiagnostic",
    datasets_path = os.environ.get("SMRAT_DATA_ROOT", "./data/PTBXL")
    log_dir = f'./logs/logging/ptbxl/{task}'
    

    seed = 312  #  
    gpu = 0
    model_type = 'smrat'
    block_type = 'smrat'
    distribute = False

    spike_mode = "lif" # 
    if task == "rhythm":
        layers = 6 # 
    else:
        layers = 2
    depths = 10  # M = ceil(log2(1000))
    max_length = 1000
    time_step = max_length
    n_bins = 5
    warmup_epochs = 15
    epochs = 200
    block_stride = 2
    compartments = 4
    kernel_sizes = [2]*compartments
    dw_kernel_size = 31
    ema_mix = 0.4

    route_policy = "topk_sparsity"
    route_target_sparsity = 0.5
    route_topk_fixed_k = max(1, round(route_target_sparsity * compartments))

    if task == "rhythm":
        if compartments == 2:
            batch_size = 50 # form 64 rhythm 50?
        elif compartments ==4:
            batch_size = 32
    else:
        batch_size = 64

    dropout_l = 0.2

    ############################
    #        USE Module        #
    ############################
    use_bn = True
    use_aug = False
    use_dp = True
    use_dw_bias = False


    backend = os.environ.get("SMRAT_BACKEND", "cupy")
    attn_mode = 'v2'

    bias = True

    t_max = 40
    lr_w = 0.0045  # 0.0045
    weight_decay = 0.06 # 0.06
    n_inputs = 12
    if compartments ==2:
        n_hidden_neurons = 128 # [128, 144, 160, 176, 192, 208, 224, 240, 256]
    elif compartments ==4:
        n_hidden_neurons = 256 # [128, 144, 160, 176, 192, 208, 224, 240, 256]
    n_outputs = 5

    num_heads = 16
    loss = 'mean'           # 'mean', 'max', 'spike_count', 'sum'
    # loss = 'none'           # 'mean', 'max', 'spike_count', 'sum'
    loss_fn = 'CEloss' # 'SmoothCEloss', 'CEloss'

    init_tau = 2.0 
    v_threshold = 1.0  # LIF
    v_reset = 0.5
    
    alpha = 5.0 # Atan = 5.0 Rectangle=1.0
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
