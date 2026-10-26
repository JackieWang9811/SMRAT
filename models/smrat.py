import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode
from module.conv import Transpose, PointwiseConv1d, DepthwiseConv1d


def init_comp_kernels_pure_random_uniform(
    d_model: int,
    kernel_sizes,
    zero_mean: bool = True,
) -> nn.ParameterList:
    """Per-compartment random temporal kernels used by SCATR.

    This is the only kernel initializer used in the current model path.
    """
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

        self.pwconv = PointwiseConv1d(
            config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True
        )
        self.dwconv = DepthwiseConv1d(
            config.n_hidden_neurons,
            config.n_hidden_neurons,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=config.use_dw_bias,
        )
        self.linear = layer.Linear(
            config.n_hidden_neurons,
            config.n_hidden_neurons,
            bias=False,
            step_mode="m",
        )

        self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode="m")
        self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode="m")
        self.dropout1 = layer.Dropout(config.dropout_l, step_mode="m")
        self.dropout2 = layer.Dropout(config.dropout_l, step_mode="m")

        self.trans_bt_to_tbc = Transpose(2, 0, 1)  # [B, C, T] -> [T, B, C]
        self.trans_tbc_to_bct = Transpose(1, 2, 0)  # [T, B, C] -> [B, C, T]

        if config.spike_mode == "lif":
            neuron_cls = LIFNode
            neuron_kwargs = dict(
                tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode="m",
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        elif config.spike_mode == "plif":
            neuron_cls = ParametricLIFNode
            neuron_kwargs = dict(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode="m",
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        else:
            raise ValueError(f"Unsupported spike_mode for SpikingEmbeding: {config.spike_mode}")

        self.lif1 = neuron_cls(**neuron_kwargs)
        self.lif2 = neuron_cls(**neuron_kwargs)

    def forward(self, x):
        x = self.pwconv(x)
        x = self.dwconv(x)
        x = self.trans_bt_to_tbc(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.dropout1(x)

        x_res = x
        x = self.linear(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.dropout2(x)
        x = x + x_res
        x = self.trans_tbc_to_bct(x)
        return x


class SpikingEmbedingSequentail(nn.Module):
    def __init__(self, config, kernel_size=7):
        super().__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(
            config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True
        )
        self.dwconv = DepthwiseConv1d(
            config.n_hidden_neurons,
            config.n_hidden_neurons,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=config.use_dw_bias,
        )
        self.linear = layer.Linear(
            config.n_hidden_neurons,
            config.n_hidden_neurons,
            bias=False,
            step_mode="m",
        )

        self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode="m")
        self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode="m")
        self.dropout1 = layer.Dropout(config.dropout_l, step_mode="m")
        self.dropout2 = layer.Dropout(config.dropout_l, step_mode="m")

        self.trans_bt_to_tbc = Transpose(2, 0, 1)
        self.trans_tbc_to_bct = Transpose(1, 2, 0)
        self.mp = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1)

        if config.spike_mode == "lif":
            neuron_cls = LIFNode
            neuron_kwargs = dict(
                tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode="m",
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        elif config.spike_mode == "plif":
            neuron_cls = ParametricLIFNode
            neuron_kwargs = dict(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode="m",
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        else:
            raise ValueError(f"Unsupported spike_mode for SpikingEmbedingSequentail: {config.spike_mode}")

        self.lif1 = neuron_cls(**neuron_kwargs)
        self.lif2 = neuron_cls(**neuron_kwargs)

    def forward(self, x):
        x = self.pwconv(x)
        x = self.dwconv(x)
        x = self.trans_bt_to_tbc(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.trans_tbc_to_bct(x)
        x = self.mp(x)
        x = self.trans_bt_to_tbc(x)
        x = self.dropout1(x)

        x_res = x
        x = self.linear(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.dropout2(x)
        x = x + x_res
        x = self.trans_tbc_to_bct(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, config, c_in, d_model):
        super().__init__()
        self.config = config

        self.proj_conv = nn.Conv2d(1, d_model * 2, kernel_size=(1, 8), stride=1, padding="same", bias=True)
        self.proj_bn = nn.BatchNorm2d(d_model * 2)

        self.proj0_conv = nn.Conv2d(d_model * 2, d_model * 4, kernel_size=(1, 16), stride=1, padding="same", bias=True)
        self.proj0_bn = nn.BatchNorm2d(d_model * 4)

        self.proj1_conv = nn.Conv2d(d_model * 4, d_model, kernel_size=(c_in, 1), stride=1, padding=0, bias=config.bias)
        self.proj1_bn = nn.BatchNorm2d(d_model)

        self.proj_res_conv = nn.Conv2d(d_model * 4, d_model, kernel_size=(c_in, 1), stride=1, padding=0, bias=config.bias)
        self.proj_res_bn = nn.BatchNorm2d(d_model)

        self.trans_bct_to_tbcd = Transpose(3, 0, 1, 2)
        self.trans_tbcd_to_bctd = Transpose(1, 2, 3, 0)
        self.trans_bct_to_tbc = Transpose(2, 0, 1)
        self.trans_tbc_to_bct = Transpose(1, 2, 0)

        self.proj_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode="m",
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )
        self.proj0_lif = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode="m",
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )

        if config.spike_mode == "lif":
            neuron_cls = LIFNode
            neuron_kwargs = dict(tau=config.init_tau)
        elif config.spike_mode == "plif":
            neuron_cls = ParametricLIFNode
            neuron_kwargs = dict(init_tau=config.init_tau)
        else:
            raise ValueError(
                "The released paper configuration uses SpikingJelly LIFNode; "
                f"unsupported spike_mode for PatchEmbedding: {config.spike_mode}"
            )

        neuron_kwargs.update(
            v_threshold=config.v_threshold,
            v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode="m",
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )
        self.proj1_lif = neuron_cls(**neuron_kwargs)
        self.proj_res_lif = neuron_cls(**neuron_kwargs)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.trans_bct_to_tbcd(x)
        x = self.proj_lif(x)
        x = self.trans_tbcd_to_bctd(x)

        x = self.proj0_conv(x)
        x = self.proj0_bn(x)
        x = self.trans_bct_to_tbcd(x)
        x = self.proj0_lif(x)
        x = self.trans_tbcd_to_bctd(x)

        x_feat = x

        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = x.squeeze(2)
        x = self.trans_bct_to_tbc(x)
        x = self.proj1_lif(x)
        x = self.trans_tbc_to_bct(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat)
        x_feat = x_feat.squeeze(2)
        x_feat = self.trans_bct_to_tbc(x_feat)
        x_feat = self.proj_res_lif(x_feat)
        x_feat = self.trans_tbc_to_bct(x_feat)

        return x + x_feat


class SCATR(nn.Module):
    """Spiking Compartment-Aware Adaptive Routing Temporal module.
    """

    def __init__(
        self,
        config,
        d_model,
        kernel_sizes=None,
        depth=None,
        seq_len=None,
        block_idx: int = 0,
        block_stride: int = 0,
        max_dilation: int = None,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.seq_len = seq_len
        self.block_idx = int(block_idx)
        self.block_stride = int(block_stride)
        self.max_dilation = max_dilation

        if depth is not None:
            self.depth = int(depth)
        elif seq_len is not None:
            if kernel_sizes is None:
                raise ValueError("kernel_sizes must be provided when depth is None and seq_len is set.")
            k_ref = max(kernel_sizes) if isinstance(kernel_sizes, (list, tuple)) else int(kernel_sizes)
            self.depth = self.max_depth(seq_len, k_ref)
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
                    step_mode="m",
                    decay_input=False,
                    store_v_seq=False,
                    backend=config.backend,
                )
                for _ in range(self.depth)
            ])
            for _ in range(self.n_comp)
        ])

        self.trans_bct_to_tbc = Transpose(2, 0, 1)
        self.trans_tbc_to_bct = Transpose(1, 2, 0)

        self.dendritic_layers = nn.ModuleList([
            LIFNode(
                tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode="m",
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
            for _ in range(self.depth)
        ])

        self.compartment_mixers = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.n_comp * self.d_model,
                out_channels=self.d_model,
                kernel_size=1,
                padding=0,
                groups=self.d_model,
                bias=False,
            )
            for _ in range(self.depth)
        ])
        with torch.no_grad():
            for cm in self.compartment_mixers:
                cm.weight.zero_()
                cm.weight[:, :, 0] = 0.5 if self.activate_k > 1 else 1.0

        self.scale_mixer = nn.Conv1d(
            in_channels=self.d_model * self.S,
            out_channels=self.d_model,
            kernel_size=1,
            padding=0,
            groups=self.d_model,
            bias=False,
        )
        with torch.no_grad():
            s = torch.arange(self.S, device=self.scale_mixer.weight.device, dtype=torch.float32)
            denom = max(self.S - 1, 1)
            decay = -math.log(0.5) / denom
            base = torch.exp(-decay * s)
            base = base / (base.sum() + 1e-12)
            scale = math.sqrt(1.0 / self.S)
            w = base[None, :].expand(self.d_model, self.S)
            self.scale_mixer.weight.copy_((w * scale).view(self.d_model, self.S, 1))

        self.scale_bn = nn.BatchNorm1d(self.d_model)
        self.lif3 = LIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode="m",
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )

        self.enable_route_log = bool(getattr(config, "enable_route_log", False))
        if self.enable_route_log:
            log_dir = "./route_logs"
            os.makedirs(log_dir, exist_ok=True)
            self.route_log_path = os.path.join(
                log_dir,
                f"route_block{block_idx}_ncomp{self.n_comp}_{config.current_time}.log",
            )

    @staticmethod
    def max_depth(L, K_ref):
        return math.ceil(math.log2((L - 1) / (K_ref - 1) + 1))

    def _dilation(self, level: int):
        d = (2 ** level) + 1 + self.block_idx * self.block_stride
        if self.max_dilation is not None:
            d = min(d, int(self.max_dilation))
        return d

    @staticmethod
    def _pool_vec(z: torch.Tensor) -> torch.Tensor:
        return z.detach().mean(dim=(0, 2))

    @staticmethod
    def _cosine_abs(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = a / (a.norm(p=2) + eps)
        b = b / (b.norm(p=2) + eps)
        return (a * b).sum().abs()

    @staticmethod
    def _score_comp(z_or_y: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        return (z_or_y.detach() - x_ref.detach()).abs().mean()

    def _select_active_idx(self, scores: torch.Tensor, zs: list) -> list:
        policy = self.route_policy

        if policy == "neighbor":
            k = self.activate_k
            m_star = torch.argmax(scores).item()
            active_idx = [m_star]

            if k >= 2 and self.n_comp >= 2:
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

        if policy == "topk_fixed":
            k = max(1, min(int(self.route_topk_fixed_k), self.n_comp))
            idx = torch.topk(scores, k=k, largest=True).indices
            return idx.detach().cpu().tolist()

        if policy == "topk_sparsity":
            k = max(1, int(round(self.n_comp * float(self.route_target_sparsity))))
            k = min(k, self.n_comp)
            idx = torch.topk(scores, k=k, largest=True).indices
            return idx.detach().cpu().tolist()

        raise ValueError(f"Unknown route_policy={policy}")

    def _maybe_log_route(self, level: int, scores: torch.Tensor, active_idx: list):
        if not (self.enable_route_log):
            return
        if not hasattr(self, "_log_step"):
            self._log_step = 0
        self._log_step += 1
        if self._log_step % 1 != 0:
            return

        score_list = [f"{s:.4f}" for s in scores.detach().cpu().tolist()]
        m_star = int(active_idx[0]) if len(active_idx) > 0 else int(torch.argmax(scores).item())
        assist_idx = active_idx[1:]
        assist_str = ",".join(str(i) for i in assist_idx) if assist_idx else "-"

        with open(self.route_log_path, "a") as f:
            f.write(
                f"{self._log_step:06d} | {level:02d} | {self.route_policy} | "
                f"{m_star} | {assist_str} | {score_list}\n"
            )

    def forward(self, x):
        """Interface-level pseudocode for the review-time release.

        Expected input/output shape: ``[batch, channels, time]``.

        Pseudocode:
            1. Validate the input channel dimension.
            2. CRTM: generate compartment--resolution responses with causal,
               compartment-specific depth-wise temporal filtering and LIF
               activation.
            3. SCATR: at each resolution, score the temporal innovation of
               every compartment, retain the configured top responses, and
               mix the activated compartments.
            4. SMRM: stack the routed representations across resolutions and
               perform spike-driven cross-resolution mixing.
            5. Add the resulting temporal representation to the residual input.

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
    """Spiking Gated Multi-resolution Modeling module."""

    def __init__(
        self,
        config,
        kernel_size=7,
        mr_kernel_size=7,
        mr_depth=3,
        mr_seq_len=None,
        block_idx: int = 0,
        block_stride=3,
        max_dilation=None,
    ):
        super().__init__()
        self.config = config
        C = config.n_hidden_neurons

        self.pwconv = PointwiseConv1d(C, 2 * C, stride=1, padding=0, bias=True)
        self.dwconv = DepthwiseConv1d(
            2 * C,
            2 * C,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=config.use_dw_bias,
        )
        self.bn1 = layer.BatchNorm1d(2 * C, step_mode="m")
        self.dropout1 = layer.Dropout(config.dropout_l, step_mode="m")

        self.trans_bct_to_tbc = Transpose(2, 0, 1)
        self.trans_tbc_to_bct = Transpose(1, 2, 0)

        if config.spike_mode == "lif":
            self.lif1 = LIFNode(
                tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode="m",
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        elif config.spike_mode == "plif":
            self.lif1 = ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode="m",
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
        else:
            raise ValueError(f"Unsupported spike_mode for SGRM: {config.spike_mode}")

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

    def forward(self, x):
        x = self.pwconv(x)
        x = self.dwconv(x)
        x = self.trans_bct_to_tbc(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.dropout1(x)

        x_res, x_gate_in = torch.split(x, self.config.n_hidden_neurons, dim=-1)
        x_gate_in = self.trans_tbc_to_bct(x_gate_in)
        x_gate = self.multires_gate(x_gate_in)
        x_gate = self.trans_bct_to_tbc(x_gate)

        x_mix = x_res * x_gate
        return self.trans_tbc_to_bct(x_mix)


class SMRAT(nn.Module):
    def __init__(
        self,
        config,
        d_output=10,
        d_model=256,
        n_layers=4,
        encoder="speech",
        n_tokens=None,
        max_length=None,
        depth=None,
        is_sequential=False,
    ):
        super().__init__()

        self.config = config
        self.max_length = max_length
        self.depth = depth
        self.is_sequential = is_sequential

        if encoder == "speech":
            self.encoder = SpikingEmbeding(config, kernel_size=7)
        elif encoder == "sequential":
            self.encoder = SpikingEmbedingSequentail(config, kernel_size=7)
        elif encoder == "eeg":
            self.encoder = PatchEmbedding(config, c_in=config.n_inputs, d_model=config.n_hidden_neurons)
        else:
            raise ValueError(f"Unknown encoder={encoder}")

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

        if self.is_sequential:
            self.output_mapping = nn.Linear(d_model, d_output, bias=config.bias)
            self.final_dp = nn.Dropout(p=config.dropout_l)
        else:
            self.output_mapping = layer.Linear(d_model, d_output, step_mode="m", bias=config.bias)
            self.final_dp = layer.Dropout(config.dropout_l, step_mode="m")

        self.trans_bct_to_tbc = Transpose(2, 0, 1)

        if self.config.dataset == "isruc":
            self.proj_linear = nn.Linear(config.n_hidden_neurons * 200, 512)
            self.proj_lif = LIFNode(
                tau=config.init_tau,
                v_threshold=config.v_threshold,
                v_reset=config.v_reset,
                surrogate_function=config.surrogate_function,
                detach_reset=config.detach_reset,
                step_mode="m",
                decay_input=False,
                store_v_seq=False,
                backend=config.backend,
            )
            self.trans_seq = Transpose(1, 0, 2)
            self.output_mapping = layer.Linear(512, d_output, step_mode="m")
            self.final_dp = layer.Dropout(config.dropout_l, step_mode="m")

    def forward(self, x, **kwargs):
        if self.config.dataset == "isruc":
            bz, seq_len, ch_num, epoch_size = x.shape
            x = x.contiguous().view(bz * seq_len, ch_num * 30, 200)

        x = self.encoder(x)

        if isinstance(self.encoder, nn.Embedding):
            x = x.transpose(-1, -2)

        for mixing_layer in self.mixing_layers:
            x = mixing_layer(x)

        if self.config.dataset == "isruc":
            x = x.contiguous().view(bz, seq_len, self.config.n_hidden_neurons * 200)
            x = self.proj_linear(x)
            out = self.proj_lif(self.trans_seq(x))
        elif self.is_sequential:
            out = x.mean(2)
            out = self.output_mapping(out)
            out = self.final_dp(out)
            return out
        else:
            out = self.trans_bct_to_tbc(x)

        out = self.output_mapping(out)
        if self.config.dataset != "timit":
            out = self.final_dp(out)
        return out