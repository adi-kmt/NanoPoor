import torch
import torch.nn as nn
from torch.fft import fft, ifft
from torch.nn import functional as F
# from muon import Muon # Assuming Muon is a custom optimizer you have; commented out for now

# Your provided FilterBank PE code - This is where it should be
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal  # For FIR filter initialization

# Ensure signal is available for _design_fir, even if only for initialization
try:
    from scipy import signal
except ImportError:
    print("Warning: scipy.signal not found. FIR filter initialization might fail if not overridden.")

# --- Config ---
config = {
    "n_embd": 256,
    "n_head": 16,  # n_embd (256) must be divisible by n_head (16) for MHA.
    "n_layer": 4,
    "n_experts": 32,
    "dropout": 0.2,
    "vocab_size": 65,
    "ctx_len": 2048,
    "init_moe_scaling": 1.25,
    "type": ['mlp', 'moe', 'mlp', 'moe'],
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "pos_num_taps": 33,
    "pos_num_filters": 4,
    'pos_filter_size': 8,
    "pos_resolution_steps": 128,
    "pos_filter_fs": 1.0,
    "moe_top_k": 2,
    "rms_norm_eps": 1e-6,
    "block_size": 16,  # Retained from original config, though less directly used by SimpleMHA
    "window_size": 128,  # Retained from original config
    "tie_weights": True,
}
config["num_tokens_to_keep"] = config["ctx_len"] // 4  # Retained from original config


@torch.compiler.disable(recursive=True)
class LearnableFIRFFT(nn.Module):
    """FFT-optimized learnable FIR filter."""

    def __init__(self, init_cutoff=0.1, num_taps=101, fs=1.0):
        super().__init__()
        self.num_taps = num_taps
        self.fs = fs
        self.init_cutoff_val = init_cutoff

        if self.num_taps % 2 == 0:
            self.num_taps += 1
        initial_coeffs = self._design_fir(init_cutoff)
        self.coeffs = nn.Parameter(torch.from_numpy(initial_coeffs).float())

    def _design_fir(self, cutoff):
        nyq = self.fs / 2.0
        if isinstance(cutoff, list):
            processed_cutoff = [max(min(c, nyq * 0.999), nyq * 0.001) for c in cutoff]
            if len(processed_cutoff) < 2 or processed_cutoff[0] >= processed_cutoff[1]:
                processed_cutoff = [nyq * 0.25, nyq * 0.75]
            return signal.firwin(self.num_taps, processed_cutoff, fs=self.fs, pass_zero=False, window='hann')
        else:
            processed_cutoff = max(min(cutoff, nyq * 0.999), nyq * 0.001)
            is_highpass = processed_cutoff > 0.5 * nyq
            return signal.firwin(self.num_taps, processed_cutoff, fs=self.fs, pass_zero=not is_highpass, window='hann')

    def forward(self, x):
        original_ndim = x.ndim
        if original_ndim == 1: x = x.unsqueeze(0)  # (Len) -> (1, Len)

        # Handle multi-dim input for PE, e.g. (T,T) where filtering is on last dim
        original_shape = x.shape
        if original_ndim > 2:
            x = x.reshape(-1, original_shape[-1])  # (..., N) -> (M, N)

        batch_size, input_len = x.shape  # input_len is the dimension to be filtered
        fft_len = input_len + self.num_taps - 1
        x_pad = F.pad(x, (0, self.num_taps - 1))  # Pad last dimension

        coeffs_for_fft = torch.zeros(fft_len, device=self.coeffs.device, dtype=self.coeffs.dtype)
        coeffs_for_fft[:self.num_taps] = self.coeffs

        x_fft = fft(x_pad, n=fft_len, dim=-1)
        c_fft = fft(coeffs_for_fft, n=fft_len, dim=-1)  # Shape (fft_len)

        # c_fft needs to be broadcastable to x_fft's shape for element-wise product
        # x_fft shape (e.g. B, fft_len), c_fft shape (1, fft_len)
        out_fft = x_fft * c_fft.unsqueeze(0) if x_fft.ndim > 1 else x_fft * c_fft
        out_ifft_complex = ifft(out_fft, n=fft_len, dim=-1)

        out_ifft_real_cloned = out_ifft_complex.real.clone()
        result = out_ifft_real_cloned[..., :input_len]  # Truncate to original input_len

        if original_ndim > 2:  # Reshape back if necessary
            result = result.reshape(*original_shape[:-1], input_len)
        elif original_ndim == 1:  # Squeeze back if original was 1D
            result = result.squeeze(0)

        return result


class MultiScalePositionalEncodingOptimized(nn.Module):
    # MODIFIED: Added num_taps as an argument
    def __init__(self, num_filters=4, init_cutoffs_list=None, num_taps=101, resolution_steps=512, fs=1.0):
        super().__init__()
        self.num_filters = num_filters
        self.resolution_steps = resolution_steps

        if init_cutoffs_list is None:
            nyq = fs / 2.0
            default_cutoffs = [nyq * 0.2, [nyq * 0.3, nyq * 0.5], [nyq * 0.5, nyq * 0.75], nyq * 0.85]
            init_cutoffs_list = default_cutoffs[:num_filters]
            if len(init_cutoffs_list) < num_filters:
                init_cutoffs_list.extend([nyq * 0.5] * (num_filters - len(init_cutoffs_list)))

        self.filters = nn.ModuleList([
            LearnableFIRFFT(init_cutoff=init_cutoffs_list[i], num_taps=num_taps, fs=fs)  # Use passed num_taps
            for i in range(num_filters)
        ])
        self.register_buffer('proto_dist_base',
                             torch.linspace(0, 1, resolution_steps).unsqueeze(0))  # (1, resolution_steps)
        self.filtered_proto_features_cache = None
        self._cache_key_device = None

    def _get_filtered_prototypes(self, device):
        proto_dist = self.proto_dist_base.to(device)  # (1, resolution_steps)
        filtered_outputs = []
        for fil_module in self.filters:
            # fil_module(proto_dist) will return (1, resolution_steps)
            filtered_outputs.append(fil_module(proto_dist))

        # Each element in filtered_outputs is (1, resolution_steps)
        # torch.cat along dim=0 makes it (num_filters, resolution_steps)
        concatenated_filters = torch.cat(filtered_outputs, dim=0)
        # Transpose to get (resolution_steps, num_filters)
        return concatenated_filters.transpose(0, 1)

    def forward(self, d_normalized_indices):  # d_normalized_indices shape e.g. (T,T)
        current_filter_device = self.filters[0].coeffs.device
        if self.training or \
                self.filtered_proto_features_cache is None or \
                self._cache_key_device != current_filter_device or \
                self.filtered_proto_features_cache.shape[0] != self.resolution_steps:
            self.filtered_proto_features_cache = self._get_filtered_prototypes(current_filter_device)
            self._cache_key_device = current_filter_device

        # self.filtered_proto_features_cache has shape (resolution_steps, num_filters)
        # d_normalized_indices is e.g. (T,T) tensor of indices (float from round, convert to long for embedding)
        pos_features = F.embedding(d_normalized_indices.long(), self.filtered_proto_features_cache)
        # Output shape will be (*d_normalized_indices.shape, num_filters), e.g. (T,T, num_filters)
        return pos_features


# --- NEW SimpleMHA Attention Module ---
class SimpleMHA(nn.Module):
    def __init__(self, Cfg):  # Use Cfg to avoid conflict with global 'config'
        super().__init__()
        self.config = Cfg
        self.n_embd = self.config['n_embd']
        self.n_head = self.config['n_head']
        self.dropout_val = self.config['dropout']

        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}) for nn.MultiheadAttention.")

        self.mha = nn.MultiheadAttention(
            embed_dim=self.n_embd,
            num_heads=self.n_head,
            dropout=self.dropout_val,  # Dropout on attention weights
            batch_first=True
        )

        self.pos_encoder = MultiScalePositionalEncodingOptimized(
            num_filters=self.config['pos_num_filters'],
            num_taps=self.config['pos_num_taps'],
            resolution_steps=self.config['pos_resolution_steps'],
            fs=self.config['pos_filter_fs']
        )
        self.pos_proj = nn.Linear(self.config['pos_num_filters'], self.n_head, bias=False)
        self.register_buffer('_pos_bias_base_cache', None, persistent=False)
        self.cached_seq_len_pos_enc = -1

        self.res_dropout = nn.Dropout(p=self.dropout_val)

    def _get_full_pos_bias(self, T, device):
        # This check needs to ensure the cache is valid for (1, n_head, T, T)
        # The existing shape check self._pos_bias_base_cache.shape[-1] != T is correct for this.
        if self.training or \
                self.cached_seq_len_pos_enc != T or \
                self._pos_bias_base_cache is None or \
                self._pos_bias_base_cache.device != device or \
                self._pos_bias_base_cache.shape[-1] != T:  # Checks if cached T matches current T

            r = torch.arange(T, device=device)
            rel_dist_matrix = torch.abs(r.unsqueeze(0) - r.unsqueeze(1))  # (T, T)
            max_rel_dist = max(T - 1, 1.0)
            norm_rel_dist = rel_dist_matrix / max_rel_dist  # (T, T)

            current_pos_indices = (norm_rel_dist * (self.config['pos_resolution_steps'] - 1)).round()  # (T,T)
            clamped_indices = torch.clamp(current_pos_indices, 0, self.config['pos_resolution_steps'] - 1)  # (T,T)

            pos_features = self.pos_encoder(clamped_indices)  # (T, T, num_pos_filters)
            projected_pos_features = self.pos_proj(pos_features)  # (T, T, n_head)

            # Permute to (n_head, T, T) and add batch dim for caching: (1, n_head, T, T)
            self._pos_bias_base_cache = projected_pos_features.permute(2, 0, 1).unsqueeze(0)
            self.cached_seq_len_pos_enc = T
        return self._pos_bias_base_cache

    def forward(self, x):
        B, T, C = x.shape
        if T == 0:
            return torch.zeros(B, 0, C, device=x.device, dtype=x.dtype)

        # attn_mask_bias will be (1, n_head, T, T) from cache/computation
        attn_mask_bias = self._get_full_pos_bias(T, x.device)

        # Prepare the mask for nn.MultiheadAttention:
        # It expects a 3D mask of shape (B * num_heads, T, T)
        # or a 2D mask of shape (T,T)
        # Our bias is head-specific, so we need the (B * num_heads, T, T) format.
        # We expand our (1, n_head, T, T) mask to (B, n_head, T, T)
        # and then reshape to (B * n_head, T, T).
        attn_mask_for_mha = attn_mask_bias.expand(B, self.n_head, T, T).reshape(B * self.n_head, T, T)

        attn_output, _ = self.mha(
            query=x, key=x, value=x,
            attn_mask=attn_mask_for_mha,  # Additive bias, now correctly shaped
            is_causal=True,  # Applies causal mask in conjunction with attn_mask
            need_weights=False
        )
        return self.res_dropout(attn_output)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        n_embd = config['n_embd']
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class UnitCenteredNoise(nn.Module):
    def __init__(self, scaling=0.02):
        super(UnitCenteredNoise, self).__init__()
        self.scaling = scaling
        self.base = 1 - (scaling * 0.5)

    def forward(self, x):
        if self.training:
            noise = torch.rand(x.size(), device=x.device, dtype=x.dtype)
            noise_centered = (noise * self.scaling) + self.base
            return x * noise_centered
        else:
            return x


class DSMoE(nn.Module):
    def __init__(self, index, num_exp_routed_to=None):
        super().__init__()
        self.num_total_experts = config["n_experts"]
        if num_exp_routed_to is None:
            num_exp_routed_to = config['moe_top_k']
        self.num_experts_to_route_to = num_exp_routed_to

        self.experts = nn.ModuleList([MLP() for _ in range(self.num_total_experts)])
        self.gate = nn.Sequential(
            nn.Linear(config['n_embd'], self.num_total_experts, bias=False),
            UnitCenteredNoise(scaling=0.02),
        )

    def forward(self, x):
        b, t, c = x.shape
        x_flat = x.reshape(b * t, c)
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_probs, self.num_experts_to_route_to, dim=-1)
        top_k_weights_norm = top_k_weights / torch.clamp(top_k_weights.sum(dim=-1, keepdim=True), min=1e-6)

        output_flat = torch.zeros_like(x_flat)
        for i in range(self.num_experts_to_route_to):
            current_expert_indices = top_k_indices[:, i]
            current_router_weights = top_k_weights_norm[:, i]
            for expert_idx in range(self.num_total_experts):
                mask = (current_expert_indices == expert_idx)
                if mask.any():
                    tokens_for_this_expert = x_flat[mask]
                    expert_output = self.experts[expert_idx](tokens_for_this_expert)
                    output_flat[mask] += expert_output * current_router_weights[mask].unsqueeze(-1)

        router_weights_sparse = torch.zeros(x_flat.size(0), self.num_total_experts, device=x.device,
                                            dtype=top_k_weights_norm.dtype)
        router_weights_sparse.scatter_(1, top_k_indices, top_k_weights_norm)
        return output_flat.reshape(b, t, c), router_weights_sparse


class Block(nn.Module):
    def __init__(self, index):
        super().__init__()
        n_embd = config['n_embd']
        # MODIFICATION: Use SimpleMHA and pass the global config
        self.attn = SimpleMHA(config)
        self.ffn_type = config['type'][index]

        if self.ffn_type == "mlp":
            self.ffn = MLP()
        elif self.ffn_type == "moe":
            self.ffn = DSMoE(index, num_exp_routed_to=config['moe_top_k'])
        else:
            raise ValueError(f"Invalid layer type: {self.ffn_type}")

        self.rm1 = nn.RMSNorm(n_embd, eps=config['rms_norm_eps'])
        self.rm2 = nn.RMSNorm(n_embd, eps=config['rms_norm_eps'])

    def forward(self, x):
        x = x + self.attn(self.rm1(x))
        ffn_input = self.rm2(x)
        if self.ffn_type == "moe":
            x_ffn, router_weights = self.ffn(ffn_input)
            x = x + x_ffn
            return x, router_weights
        else:
            x_ffn = self.ffn(ffn_input)
            x = x + x_ffn
            return x, None


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(self.config['vocab_size'], self.config['n_embd'])
        self.blocks = nn.ModuleList([Block(i) for i in range(self.config['n_layer'])])
        self.rm_f = nn.RMSNorm(self.config['n_embd'], eps=self.config['rms_norm_eps'])
        self.lm_head = nn.Linear(self.config['n_embd'], self.config['vocab_size'], bias=False)
        if self.config.get('tie_weights', True):
            self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable model parameters: {self.total_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb
        all_router_weights = []
        for block_module in self.blocks:
            x, router_weights = block_module(x)
            if router_weights is not None:
                all_router_weights.append(router_weights)
        x = self.rm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits_for_loss = logits.view(-1, logits.size(-1))
            targets_for_loss = targets.view(-1)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)
        return logits, loss, all_router_weights

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, tiktoken_vocab_size=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['ctx_len']:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if tiktoken_vocab_size is not None and tiktoken_vocab_size < self.config['vocab_size']:
                    logits[:, tiktoken_vocab_size:] = float('-inf')
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx, 0.0  # kv_cache_gb is now a dummy value

    def configure_optimizers(self, weight_decay, learning_rate, device_type_str):
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.ndim >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.ndim < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        use_fused = False
        if device_type_str == 'cuda':
            if hasattr(torch.optim, 'AdamW'):
                # Try creating a dummy optimizer to check for fused support
                try:
                    dummy_param = nn.Parameter(torch.randn(1, device='cuda'))
                    torch.optim.AdamW([dummy_param], lr=1e-4, fused=True)
                    use_fused = True
                except RuntimeError:  # Catches "Torch not compiled with CUDA enabled" or "FusedAdamW isn't supported"
                    use_fused = False
                except Exception:  # Other potential errors
                    use_fused = False
            if not use_fused and hasattr(torch.optim, '_fused_adamw_supported'):  # Fallback for older PyTorch
                use_fused = torch.optim._fused_adamw_supported(torch.device('cuda'))

        optimizer_kwargs = {'lr': learning_rate, 'betas': (0.9, 0.95)}
        if use_fused:
            optimizer_kwargs['fused'] = True
        optimizer = torch.optim.AdamW(optim_groups, **optimizer_kwargs)
        print(f"Using Fused AdamW: {use_fused and optimizer_kwargs.get('fused', False)}")
        return [optimizer]

    def update_expert_biases(self, all_router_weights_list, update_rate):
        pass

    def estimate_mfu(self, num_params, fwdbwd_per_iter, dt):
        N = num_params
        L = self.config['ctx_len']
        flops_per_token = 6 * N  # Standard rough estimate for Transformers
        flops_per_fwd_iter = flops_per_token * L
        flops_per_iter_total = flops_per_fwd_iter * fwdbwd_per_iter * 3  # (fwd + bwd ~ 3x fwd)

        flops_achieved = flops_per_iter_total / dt
        flops_promised = 100e12  # Example: 100 TFLOPS for a modern GPU like A100 (BF16 peak is ~312 TFLOPs)
        mfu = flops_achieved / flops_promised
        return mfu


# --- Main Test Case ---
if __name__ == "__main__":
    device = config['device']
    print(f"Using device: {device}")

    model = Transformer().to(device)

    print("\nTesting forward pass...")
    test_idx = torch.randint(0, config['vocab_size'], (2, 32), device=device)
    test_targets = torch.randint(0, config['vocab_size'], (2, 32), device=device)
    logits, loss, router_weights_list = model(test_idx, test_targets)
    print(f"Logits shape: {logits.shape}")
    if loss is not None: print(f"Loss: {loss.item():.4f}")
    if router_weights_list and len(router_weights_list) > 0:
        print(f"Num MoE router weights sets: {len(router_weights_list)}")
        for i, r_w in enumerate(router_weights_list): print(f"  MoE {i} weights shape: {r_w.shape}")
    else:
        print("No MoE layers or MoE weights not returned/found in this configuration.")
    print("Forward pass test OK.")

    print("\nTesting generation...")
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    generated_output, _ = model.generate(prompt, max_new_tokens=5, temperature=0.0)
    print(f"Generated sequence shape: {generated_output.shape}")
    print(f"Generated sequence: {generated_output.tolist()}")
    print("Generation test OK.")

    print("\nTesting optimizer...")
    optimizers = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-4, device_type_str=str(device))
    print(f"Num optimizer groups: {len(optimizers[0].param_groups)}")
    print("Optimizer test OK.")

    print("\nTesting MFU...")
    mfu = model.estimate_mfu(model.total_params, fwdbwd_per_iter=4, dt=1.0)
    print(f"Estimated MFU: {mfu * 100:.2f}% (using 100 TFLOPs peak baseline)")
    print("MFU test OK.")

    first_block_attn = model.blocks[0].attn
    if isinstance(first_block_attn, SimpleMHA) and \
            hasattr(first_block_attn, 'pos_encoder') and \
            first_block_attn.pos_encoder.num_filters > 0:
        print("\nVisualizing initial PE filters (from SimpleMHA)...")
        try:
            num_filters_to_plot = first_block_attn.pos_encoder.num_filters
            cols = min(4, num_filters_to_plot)
            rows = (num_filters_to_plot + cols - 1) // cols

            plt.figure(figsize=(max(15, 3 * cols), 3 * rows))
            pos_encoder_module = first_block_attn.pos_encoder

            for i, filter_module in enumerate(pos_encoder_module.filters):
                coeffs = filter_module.coeffs.detach().cpu().numpy()
                plt.subplot(rows, cols, i + 1)
                plt.plot(coeffs)
                title_str = f"Filter {i + 1}"
                if hasattr(filter_module, 'init_cutoff_val'):
                    title_str += f"\n(Init: {filter_module.init_cutoff_val})"
                plt.title(title_str)
                plt.grid(True)
            plt.suptitle("Initial FIR Filter Coefficients (First SimpleMHA Block PE)", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig('simplemha_model_filters_initial.png')
            print("PE Filter visualization saved as simplemha_model_filters_initial.png")
        except Exception as e:
            print(f"Could not visualize PE filters: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Positional Encoding filters not found in the expected SimpleMHA structure for visualization.")