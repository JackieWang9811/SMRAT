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
    dataset = 'isruc'
    datasets_path = os.environ.get("SMRAT_DATA_ROOT", "./data/ISRUC/processed")
    log_dir = './logs/logging/isruc/smrat'
    

    # seed = 312  #  
    seed = [312, 42, 3112, 0, 10086]
    gpu = 0
    model_type = 'smrat'
    block_type = 'smrat'

    distribute = False

    spike_mode = "lif" # 
    layers = 2 # 
    depths = 8  # M = ceil(log2(200))
    block_stride = 2
    compartments = 4
    kernel_sizes = [2] * compartments  # * compartmens
    dw_kernel_size = 31
    backend = os.environ.get("SMRAT_BACKEND", "cupy")
    bias = True
    max_length = 200

    route_policy = "topk_sparsity"
    route_target_sparsity = 0.5
    route_topk_fixed_k = max(1, round(route_target_sparsity * compartments))
    dropout_l = 0.1

    time_step = 10
    n_bins = 5
    epochs = 50
    warmup_epochs = 5

    batch_size = 16
    gradient_accumulation_steps = 1  # 
    use_amp = True  # 
    num_workers = 4  # 



    ############################
    #        USE Module        #
    ############################
    use_bn = True
    use_aug = False
    use_dp = True
    use_dw_bias = False


    t_max = 40
    lr_w = 2e-3
    weight_decay = 0.05 # Default 1e-2 => 5e-4
    n_inputs = 180 # 6*30
    if compartments == 2:
        n_hidden_neurons = 16
    elif compartments == 4:
        n_hidden_neurons = 20
    n_outputs = 5

    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum'
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
    run_info = f'||{dataset}||{depths}depths||lr_w={lr_w}'
    checkpoint_name = run_name + f'||seed={seed}' + run_info
    # 
    save_model_path = f'{checkpoint_name}_REPL_{current_time}.pt'
    make_plot = False


