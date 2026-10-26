import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import layer
from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode
from module.conv import Transpose, PointwiseConv1d, DepthwiseConv1d


def init_comp_kernels_pure_random_uniform(d_model: int, kernel_sizes, zero_mean: bool = True) -> nn.ParameterList:
    plist = nn.ParameterList()
    for K in kernel_sizes:
        w = torch.empty(d_model, 1, K).uniform_(-1.0, 1.0)
        w = w * (1.0 / math.sqrt(float(K)))
        if zero_mean:
            w = w - w.mean(dim=2, keepdim=True)
        plist.append(nn.Parameter(w))
    return plist


class SpikingEmbeding(nn.Module):
    def __init__(self, config, kernel_size=7):
        super().__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True)
        self.dwconv = DepthwiseConv1d(
            config.n_hidden_neurons,
            config.n_hidden_neurons,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=config.use_dw_bias,
        )
        self.linear = layer.Linear(config.n_hidden_neurons, config.n_hidden_neurons, bias=False, step_mode='m')

        if self.config.use_bn:
            self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
            self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        if config.spike_mode == "lif":
            node_cls = LIFNode
            node_kwargs = dict(
                tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        elif config.spike_mode == "plif":
            node_cls = ParametricLIFNode
            node_kwargs = dict(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        else:
            raise ValueError(f"Unsupported spike_mode: {config.spike_mode}")

        self.lif1 = node_cls(**node_kwargs)
        self.lif2 = node_cls(**node_kwargs)

    def forward(self, x):
        x = self.pwconv(x)
        x = self.dwconv(x)
        x = self.trans2(x)
        if self.config.use_bn:
            x = self.bn1(x)
        x = self.lif1(x)
        if self.config.use_dp:
            x = self.dropout1(x)
        x_res = x
        x = self.linear(x)
        if self.config.use_bn:
            x = self.bn2(x)
        x = self.lif2(x)
        if self.config.use_dp:
            x = self.dropout2(x)
        x = x + x_res
        x = self.trans3(x)
        return x


def div_ceil(a, b):
    return (a + b - 1) // b


class SPS(nn.Module):
    def __init__(self, config, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        self.patch_size = (patch_size, patch_size)
        self.H = self.image_size[0] // self.patch_size[0]
        self.W = self.image_size[1] // self.patch_size[1]
        self.num_patches = self.H * self.W
        self.pooling_stat = config.pooling_stat

        self.proj_conv = layer.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.proj_bn = layer.BatchNorm2d(embed_dims // 8, step_mode='m')
        self.proj_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, step_mode='m')

        self.proj_conv1 = layer.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.proj_bn1 = layer.BatchNorm2d(embed_dims // 4, step_mode='m')
        self.proj_lif1 = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )
        self.maxpool1 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, step_mode='m')

        self.proj_conv2 = layer.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.proj_bn2 = layer.BatchNorm2d(embed_dims // 2, step_mode='m')
        self.proj_lif2 = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )
        self.maxpool2 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, step_mode='m')

        self.proj_conv3 = layer.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.proj_bn3 = layer.BatchNorm2d(embed_dims, step_mode='m')
        self.proj_lif3 = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )
        self.maxpool3 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, step_mode='m')

        self.rpe_conv = layer.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1, padding=0, bias=False, step_mode='m')
        self.rpe_bn = layer.BatchNorm2d(embed_dims, step_mode='m')
        self.rpe_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        ratio = 1

        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        h = div_ceil(H, ratio)
        w = div_ceil(W, ratio)
        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, h, w).contiguous()
        x = self.proj_lif1(x).contiguous()
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        h = div_ceil(H, ratio)
        w = div_ceil(W, ratio)
        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, h, w).contiguous()
        x = self.proj_lif2(x).contiguous()
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2

        h = div_ceil(H, ratio)
        w = div_ceil(W, ratio)
        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, h, w).contiguous()
        x = self.proj_lif3(x).contiguous()
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        h = div_ceil(H, ratio)
        w = div_ceil(W, ratio)
        x_feat = x.reshape(T, B, -1, h, w).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, h, w).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        return x.flatten(-2)


class SCATR(nn.Module):
    def __init__(self, config, d_model, kernel_sizes=None, depth=None, seq_len=None, block_idx: int = 0, block_stride: int = 0, max_dilation: int = None):
        super().__init__()

        self.d_model = int(d_model)
        self.block_idx = int(block_idx)
        self.block_stride = int(block_stride)
        self.max_dilation = max_dilation

        if depth is not None:
            self.depth = int(depth)
        elif seq_len is not None:
            if kernel_sizes is None:
                raise ValueError("kernel_sizes must be provided when depth is None and seq_len is set.")
            K_ref = max(kernel_sizes) if isinstance(kernel_sizes, (list, tuple)) else int(kernel_sizes)
            self.depth = self.max_depth(seq_len, K_ref)
        else:
            raise ValueError("Either depth or seq_len must be provided.")
        self.S = self.depth

        if kernel_sizes is None:
            K = getattr(config, "kernel_size", None)
            if K is None:
                raise ValueError("kernel_sizes is None and config.kernel_size is not set.")
            kernel_sizes = [K, K]
        self.kernel_sizes = list(kernel_sizes)
        self.n_comp = len(self.kernel_sizes)
        if self.n_comp < 1:
            raise ValueError("n_comp must be >= 1.")

        self.route_policy = config.route_policy
        self.route_topk_fixed_k = config.route_topk_fixed_k
        self.route_target_sparsity = config.route_target_sparsity
        self.activate_k = max(1, int(round(self.n_comp * self.route_target_sparsity)))
        self.route_score = "innovation_l1"
        self.dilation_mode = getattr(config, "dilation_mode", "2p1")

        self.comp_kernels = init_comp_kernels_pure_random_uniform(
            d_model=self.d_model,
            kernel_sizes=self.kernel_sizes,
            zero_mean=True,
        )

        self.scale_dropout = layer.Dropout(config.dropout_l, step_mode="m")
        self.compartment_dropout = layer.Dropout(config.dropout_l, step_mode="m")

        self.comp_lif_layers = nn.ModuleList([
            nn.ModuleList([
                LIFNode(
                    tau=config.init_tau,
                    v_threshold=config.v_threshold,
                    v_reset=config.v_reset,
                    surrogate_function=config.surrogate_function,
                    detach_reset=config.detach_reset,
                    step_mode='m',
                    decay_input=False,
                    store_v_seq=False,
                    backend=config.backend,
                )
                for _ in range(self.depth)
            ])
            for _ in range(self.n_comp)
        ])

        self.dendritic_layers = nn.ModuleList([
            LIFNode(
                tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
            for _ in range(self.depth)
        ])

        self.compartment_mixers = nn.ModuleList([
            layer.Conv1d(
                in_channels=self.n_comp * self.d_model,
                out_channels=self.d_model,
                kernel_size=1,
                padding=0,
                groups=self.d_model,
                bias=False,
                step_mode='m',
            )
            for _ in range(self.depth)
        ])
        with torch.no_grad():
            for cm in self.compartment_mixers:
                cm.weight.zero_()
                cm.weight[:, :, 0] = 0.5 if self.activate_k > 1 else 1.0

        self.scale_mixer = layer.Conv1d(
            in_channels=self.d_model * self.S,
            out_channels=self.d_model,
            kernel_size=1,
            padding=0,
            groups=self.d_model,
            bias=False,
            step_mode='m',
        )
        with torch.no_grad():
            s = torch.arange(self.S, device=self.scale_mixer.weight.device, dtype=torch.float32)
            decay = -math.log(0.5) / max(self.S - 1, 1)
            base = torch.exp(-decay * s)
            base = base / (base.sum() + 1e-12)
            scale = math.sqrt(1.0 / self.S)
            w = base[None, :].expand(self.d_model, self.S)
            self.scale_mixer.weight.copy_((w * scale).view(self.d_model, self.S, 1))

        self.scale_bn = layer.BatchNorm1d(self.d_model, step_mode='m')
        self.lif3 = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )

        self.enable_route_log = bool(getattr(config, "enable_route_log", False))
        if self.enable_route_log:
            os.makedirs("./route_logs", exist_ok=True)
            self.route_log_path = os.path.join(
                "./route_logs",
                f"route_block{block_idx}_ncomp{self.n_comp}_{config.current_time}.log",
            )

    @staticmethod
    def max_depth(L, K_ref):
        return math.ceil(math.log2((L - 1) / (K_ref - 1) + 1))

    def _dilation(self, level: int):
        base = 2 ** level if self.dilation_mode == "pow2" else (2 ** level) + 1
        d = base + self.block_idx * self.block_stride
        if self.max_dilation is not None:
            d = min(d, int(self.max_dilation))
        return d

    @staticmethod
    def _pool_vec(z: torch.Tensor) -> torch.Tensor:
        return z.detach().mean(dim=(0, 1, 3))

    @staticmethod
    def _cosine_abs(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = a / (a.norm(p=2) + eps)
        b = b / (b.norm(p=2) + eps)
        return (a * b).sum().abs()

    def _score_comp(self, z_or_y: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        return (z_or_y.detach() - x_ref.detach()).abs().mean()

    def _select_active_idx(self, scores: torch.Tensor, zs: list) -> list:
        if self.route_policy == "neighbor":
            m_star = torch.argmax(scores).item()
            active_idx = [m_star]
            if self.activate_k >= 2 and self.n_comp >= 2:
                cand = []
                if m_star - 1 >= 0:
                    cand.append(m_star - 1)
                if m_star + 1 < self.n_comp:
                    cand.append(m_star + 1)
                v_star = self._pool_vec(zs[m_star])
                best_m, best_val = None, -1e9
                for m in cand:
                    v_m = self._pool_vec(zs[m])
                    co = self._cosine_abs(v_m, v_star)
                    val = 0.7 * co + 0.3 * scores[m].item()
                    if val > best_val:
                        best_val, best_m = val, m
                if best_m is not None:
                    active_idx.append(best_m)
            return active_idx

        if self.route_policy == "topk_fixed":
            k = max(1, min(int(self.route_topk_fixed_k), self.n_comp))
            return torch.topk(scores, k=k, largest=True).indices.detach().cpu().tolist()

        if self.route_policy == "topk_sparsity":
            k = max(1, int(round(self.n_comp * float(self.route_target_sparsity))))
            k = min(k, self.n_comp)
            return torch.topk(scores, k=k, largest=True).indices.detach().cpu().tolist()

        raise ValueError(f"Unknown route_policy={self.route_policy}")

    def _maybe_log_route(self, level: int, scores: torch.Tensor, active_idx: list):
        # if not (self.enable_route_log and self.training):
        #     return
        if not self.enable_route_log:
            return
        if not hasattr(self, "_log_step"):
            self._log_step = 0
        self._log_step += 1
        if self._log_step % 18 != 0:
            return
        score_list = [f"{s:.4f}" for s in scores.detach().cpu().tolist()]
        m_star = int(active_idx[0]) if active_idx else int(torch.argmax(scores).item())
        assist_idx = active_idx[1:]
        assist_str = ",".join(str(i) for i in assist_idx) if assist_idx else "-"
        with open(self.route_log_path, "a") as f:
            f.write(
                f"{self._log_step:06d} | {level:02d} | {self.route_policy} | "
                f"{m_star} | {assist_str} | {score_list}\n"
            )

    def forward(self, x):
        """Interface-level pseudocode for the review-time release.

        Expected input/output shape: ``[time, batch, channels, tokens]``.

        Pseudocode:
            1. Validate the input channel dimension.
            2. CRTM: generate compartment--resolution responses with
               compartment-specific depth-wise filtering and multi-step LIF
               activation over the embedded event representation.
            3. SCATR: at each resolution, score temporal innovation, retain the
               configured top compartment responses, and mix the activated
               compartments.
            4. SMRM: stack all routed resolution representations and perform
               spike-driven cross-resolution mixing.
            5. Add the resulting representation to the residual input.

        The exact tensor operations in this method are intentionally omitted
        from the anonymous review package. Consequently, this package exposes
        the architecture, initialization, configuration, and training
        interfaces, but it is not sufficient to reproduce the reported
        numerical results. The authors intend to release the complete
        implementation with the camera-ready version if the paper is accepted.
        """
        raise NotImplementedError(
            "SCATR.forward is intentionally omitted from the anonymous "
            "review-time release; see its docstring for the disclosed scope."
        )


class SGRM(nn.Module):
    def __init__(self, config, kernel_size=7, mr_kernel_size=7, mr_depth=3, mr_seq_len=None, block_idx: int = 0, block_stride=3, max_dilation=None):
        super().__init__()
        C = config.n_hidden_neurons

        self.pwconv = layer.Conv1d(C, 2 * C, kernel_size=1, stride=1, padding=0, bias=True, step_mode='m')
        self.dwconv = layer.Conv1d(2 * C, 2 * C, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=2 * C, bias=config.use_dw_bias, step_mode='m')
        self.bn1 = layer.BatchNorm1d(2 * C, step_mode='m')
        self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')

        if config.spike_mode == 'lif':
            self.lif1 = LIFNode(
                tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        elif config.spike_mode == 'plif':
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        else:
            raise ValueError(f"Unsupported spike_mode: {config.spike_mode}")

        self.multires_gate = SCATR(
            config=config,
            d_model=C,
            kernel_sizes=mr_kernel_size,
            depth=mr_depth,
            seq_len=mr_seq_len,
            block_stride=block_stride,
            max_dilation=max_dilation,
            block_idx=block_idx,
        )
        self.n_hidden_neurons = config.n_hidden_neurons

    def forward(self, x):
        x = self.pwconv(x)
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.dropout1(x)
        x_res, x_gate_in = torch.split(x, self.n_hidden_neurons, dim=-2)
        x_gate = self.multires_gate(x_gate_in)
        return x_res * x_gate


class SMRAT(nn.Module):
    def __init__(self, config, d_output=10, d_model=256, n_layers=4, encoder="dvs", n_tokens=None, max_length=None, depth=None, is_sequential=False):
        super().__init__()
        self.config = config
        self.max_length = max_length
        self.depth = depth
        self.is_sequential = is_sequential

        if encoder == "speech":
            self.encoder = SpikingEmbeding(config, kernel_size=7)
        elif encoder == "dvs":
            self.encoder = SPS(config, embed_dims=config.n_hidden_neurons)
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        self.mixing_layers = nn.ModuleList([
            SGRM(
                config,
                kernel_size=config.dw_kernel_size,
                mr_kernel_size=config.kernel_sizes,
                mr_depth=config.depths,
                mr_seq_len=max_length,
                block_idx=i,
                block_stride=config.block_stride,
                max_dilation=None,
            )
            for i in range(n_layers)
        ])

        self.output_mapping = nn.Linear(d_model, d_output, bias=config.bias)
        self.final_dp = nn.Dropout(p=config.dropout_l)

    def forward(self, x, **kwargs):
        x = x.permute(1, 0, 2, 3, 4)
        x = self.encoder(x)

        if isinstance(self.encoder, nn.Embedding):
            x = x.transpose(-1, -2)

        for mixing_layer in self.mixing_layers:
            x = mixing_layer(x)

        x = x.mean(3).mean(0)
        out = self.output_mapping(x)
        out = self.final_dp(out)
        return out