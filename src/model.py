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


class LearnableFIRFFT(nn.Module):
    """FFT-optimized learnable FIR filter."""

    def __init__(self, init_cutoff=0.1, num_taps=101, fs=1.0):
        super().__init__()
        self.num_taps = num_taps
        self.fs = fs
        self.init_cutoff_val = init_cutoff  # Store for potential visualization

        initial_coeffs = self._design_fir(init_cutoff)
        self.coeffs = nn.Parameter(
            torch.from_numpy(initial_coeffs).float()
        )

    def _design_fir(self, cutoff):
        """Design windowed-sinc FIR filter (lowpass/bandpass/highpass)."""
        nyq = self.fs / 2.0
        # Ensure num_taps is odd for symmetric filter (often desired for linear phase)
        if self.num_taps % 2 == 0:
            # print(f"Warning: num_taps ({self.num_taps}) is even. Incrementing to make it odd for symmetric FIR design.")
            self.num_taps += 1

        if isinstance(cutoff, list):  # Bandpass
            processed_cutoff = [max(min(c, nyq * 0.999), nyq * 0.001) for c in cutoff]
            if processed_cutoff[0] >= processed_cutoff[1]:
                processed_cutoff[0] = processed_cutoff[1] * 0.5
            return signal.firwin(self.num_taps, processed_cutoff, fs=self.fs, pass_zero=False, window='hann')
        else:  # Lowpass/Highpass
            processed_cutoff = max(min(cutoff, nyq * 0.999), nyq * 0.001)
            is_highpass = processed_cutoff > 0.5 * nyq
            return signal.firwin(self.num_taps, processed_cutoff, fs=self.fs, pass_zero=not is_highpass, window='hann')

    def forward(self, x):
        """FFT-based convolution. Input x shape: (batch, length)"""
        if x.ndim == 1:
            x = x.unsqueeze(0)

        batch_size, input_len = x.shape
        fft_len = input_len + self.num_taps - 1

        x_pad = torch.nn.functional.pad(x, (0, self.num_taps - 1))

        coeffs_for_fft = torch.zeros(fft_len, device=self.coeffs.device, dtype=self.coeffs.dtype)
        coeffs_for_fft[:self.num_taps] = self.coeffs

        x_fft = fft(x_pad, n=fft_len, dim=-1)
        c_fft = fft(coeffs_for_fft, n=fft_len, dim=-1)

        out_fft = x_fft * c_fft.unsqueeze(0)
        out_ifft = ifft(out_fft, n=fft_len, dim=-1).real

        return out_ifft[..., :input_len]


class MultiScalePositionalEncodingOptimized(nn.Module):
    """Learnable multi-scale positional encoding with FFT optimization on a prototype signal."""

    def __init__(self, num_filters=4, init_cutoffs_list=None, num_taps=101, resolution_steps=512, fs=1.0):
        super().__init__()
        self.num_filters = num_filters
        self.resolution_steps = resolution_steps
        self.num_taps = num_taps

        if init_cutoffs_list is None:
            nyq = fs / 2.0
            default_cutoffs = [0.2 * nyq, [0.3 * nyq, 0.5 * nyq], [0.5 * nyq, 0.75 * nyq], 0.85 * nyq]
            init_cutoffs_list = default_cutoffs[:num_filters]
            if len(init_cutoffs_list) < num_filters:
                init_cutoffs_list.extend([0.5 * nyq] * (num_filters - len(init_cutoffs_list)))

        self.filters = nn.ModuleList([
            LearnableFIRFFT(init_cutoff=init_cutoffs_list[i], num_taps=num_taps, fs=fs)
            for i in range(num_filters)
        ])

        self.register_buffer('proto_dist_base', torch.linspace(0, 1, resolution_steps).unsqueeze(0))
        self.filtered_proto_features_cache = None

    def _get_filtered_prototypes(self, device):
        proto_dist = self.proto_dist_base.to(device)
        filtered_outputs = []
        for fil_module in self.filters:
            filtered_outputs.append(fil_module(proto_dist))
        return torch.cat(filtered_outputs, dim=0).transpose(0, 1)

    def forward(self, d_normalized_indices):
        current_filter_device = self.filters[0].coeffs.device
        if self.training or self.filtered_proto_features_cache is None or \
                self.filtered_proto_features_cache.device != current_filter_device or \
                self.filtered_proto_features_cache.shape[0] != self.resolution_steps:
            self.filtered_proto_features_cache = self._get_filtered_prototypes(current_filter_device)

        pos_features = self.filtered_proto_features_cache[d_normalized_indices.reshape(-1)]
        return pos_features.view(*d_normalized_indices.shape, self.num_filters)


class FilterBankAttention(nn.Module):
    """Transformer attention with learnable multi-scale positional encoding (Optimized PE)."""

    def __init__(self, dim, num_heads=8, num_taps=101, num_pos_filters=4, pos_resolution_steps=512, pos_filter_fs=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.pos_encoder = MultiScalePositionalEncodingOptimized(
            num_filters=num_pos_filters,
            init_cutoffs_list=None,
            num_taps=num_taps,
            resolution_steps=pos_resolution_steps,
            fs=pos_filter_fs
        )
        self.pos_proj = nn.Linear(num_pos_filters, num_heads, bias=False)
        self.pos_resolution_steps = pos_resolution_steps

        self.register_buffer('pos_indices_cache', None)
        self.cached_seq_len = -1

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        if self.cached_seq_len != seq_len or self.pos_indices_cache is None or self.pos_indices_cache.device != x.device:
            r = torch.arange(seq_len, device=x.device)
            rel_dist_matrix = torch.abs(r.unsqueeze(0) - r.unsqueeze(1))

            # --- CORRECTED LINE HERE ---
            max_rel_dist = max(seq_len - 1, 1.0)  # Use Python's max for scalar, ensure float division
            # --- END OF CORRECTION ---

            norm_rel_dist = rel_dist_matrix / max_rel_dist  # Ensure max_rel_dist is float if rel_dist_matrix is float

            current_pos_indices = (norm_rel_dist * (self.pos_resolution_steps - 1)).round().long()
            self.pos_indices_cache = torch.clamp(current_pos_indices, 0, self.pos_resolution_steps - 1)
            self.cached_seq_len = seq_len

        pos_features = self.pos_encoder(self.pos_indices_cache)
        pos_bias = self.pos_proj(pos_features).permute(2, 0, 1).unsqueeze(0)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + pos_bias

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T) if mask was (T,T) originally
            elif mask.ndim == 3:
                mask = mask.unsqueeze(0)
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))

        attn_weights = attn_logits.softmax(dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.to_out(out)
# End of FilterBank PE code

# --- Config ---
config = {
    "n_embd": 256,
    "n_head": 16,
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
    "pos_resolution_steps": 128,
    "pos_filter_fs": 1.0,
    "moe_top_k": 2,  # Added for DSMoE
}


class Attn(FilterBankAttention):
    def __init__(self):
        super().__init__(
            dim=config['n_embd'],
            num_heads=config['n_head'],
            num_taps=config['pos_num_taps'],
            num_pos_filters=config['pos_num_filters'],
            pos_resolution_steps=config['pos_resolution_steps'],
            pos_filter_fs=config['pos_filter_fs']
        )

    def forward(self, x):
        B, T, C = x.shape
        # Create a causal mask for self-attention (True means keep)
        # Mask shape should be broadcastable to (B, H, T, T) for attn_logits
        # A (T,T) mask is fine, it will be broadcast.
        # Or (1,1,T,T)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        return super().forward(x, mask=causal_mask)


# Reg MLP
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
    def __init__(self, index, num_exp_routed_to=config['moe_top_k']):
        super().__init__()
        self.num_total_experts = config["n_experts"]
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
        top_k_weights_norm = top_k_weights / torch.clamp(top_k_weights.sum(dim=-1, keepdim=True), min=1e-6)  # Normalize

        router_weights_sparse = torch.zeros(x_flat.size(0), self.num_total_experts, device=x.device,
                                            dtype=top_k_weights_norm.dtype)
        router_weights_sparse.scatter_(1, top_k_indices, top_k_weights_norm)

        expert_outputs_all = torch.stack([expert(x_flat) for expert in self.experts], dim=0)
        weighted_expert_outputs = expert_outputs_all * router_weights_sparse.transpose(0, 1).unsqueeze(-1)
        output_flat = weighted_expert_outputs.sum(dim=0)

        return output_flat.reshape(b, t, c), router_weights_sparse


class Block(nn.Module):
    def __init__(self, index):
        super().__init__()
        n_embd = config['n_embd']
        self.attn = Attn()
        self.ffn_type = config['type'][index]

        if self.ffn_type == "mlp":
            self.ffn = MLP()
        elif self.ffn_type == "moe":
            self.ffn = DSMoE(index)  # num_exp_routed_to taken from config['moe_top_k']
        else:
            raise ValueError(f"Invalid layer type: {self.ffn_type}")

        self.rm1 = nn.RMSNorm(n_embd)
        self.rm2 = nn.RMSNorm(n_embd)

    def forward(self, x):
        x_attn = self.attn(self.rm1(x))
        x = x + x_attn

        ffn_input = self.rm2(x)
        if self.ffn_type == "moe":
            x_ffn, router_weights = self.ffn(ffn_input)
            return x + x_ffn, router_weights
        else:
            x_ffn = self.ffn(ffn_input)
            return x + x_ffn, None


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.blocks = nn.ModuleList([Block(i) for i in range(config['n_layer'])])
        self.rm_f = nn.RMSNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        if config.get('tie_weights', True):  # Add config to control tying
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

        for block in self.blocks:
            x, router_weights = block(x)
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
            idx_cond = idx[:, -self.config['ctx_len']:]  # Ensure context windowing
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0:  # Greedy sampling
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
        return idx, 0.0  # KV cache size is 0 for this PE

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.ndim >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.ndim < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = False
        if device_type == 'cuda':  # Check if CUDA is the device type string
            fused_available = hasattr(torch.optim, '_fused_adamw') or hasattr(torch.optim, 'AdamW_fused_available')
            if hasattr(torch.optim, 'AdamW_fused_available'): fused_available = torch.optim.AdamW_fused_available()

        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), fused=use_fused)
        print(f"Using Fused AdamW: {use_fused}")
        return [optimizer]

    def update_expert_biases(self, all_router_weights_list, update_rate):
        # This method would need careful alignment with how DSMoE handles biases for load balancing
        # For now, it's a placeholder as the current DSMoE uses softmax probabilities directly
        pass

    def estimate_mfu(self, num_params, fwdbwd_per_iter, dt):
        N = num_params
        T = self.config['ctx_len']
        flops_per_token = 6 * N
        flops_per_fwdbwd = flops_per_token * T * 3
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 65e12
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
    if router_weights_list:
        print(f"Num MoE router weights sets: {len(router_weights_list)}")
        for i, r_w in enumerate(router_weights_list): print(f"  MoE {i} weights shape: {r_w.shape}")
    print("Forward pass test OK.")

    print("\nTesting generation...")
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    generated_output, kv_cache_gb = model.generate(prompt, max_new_tokens=5, temperature=0.0)  # Greedy
    print(f"Generated sequence shape: {generated_output.shape}")
    print(f"Generated sequence: {generated_output.tolist()}")
    print("Generation test OK.")

    print("\nTesting optimizer...")
    optimizers = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-4, device_type=str(device))
    print(f"Num optimizer groups: {len(optimizers)}")
    print("Optimizer test OK.")

    print("\nTesting MFU...")
    mfu = model.estimate_mfu(model.total_params, fwdbwd_per_iter=4, dt=1.0)
    print(f"Estimated MFU: {mfu * 100:.2f}%")
    print("MFU test OK.")

    if hasattr(model.blocks[0].attn, 'pos_encoder'):
        print("\nVisualizing initial PE filters...")
        plt.figure(figsize=(15, 3 * (model.blocks[0].attn.pos_encoder.num_filters // 4 + 1)))  # Adjust height
        pos_encoder_module = model.blocks[0].attn.pos_encoder
        for i, filter_module in enumerate(pos_encoder_module.filters):
            coeffs = filter_module.coeffs.detach().cpu().numpy()
            plt.subplot((model.blocks[0].attn.pos_encoder.num_filters // 4 + 1), 4, i + 1)  # Dynamic subplot
            plt.plot(coeffs)
            title_str = f"Filter {i + 1}"
            if hasattr(filter_module, 'init_cutoff_val'):
                title_str += f"\n(Init: {filter_module.init_cutoff_val})"
            plt.title(title_str)
            plt.grid(True)
        plt.suptitle("Initial FIR Filter Coefficients (First Attn Block PE)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('integrated_model_filters_initial.png')
        print("PE Filter visualization saved.")
        plt.show()