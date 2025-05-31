import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import List, Optional, Dict, Tuple, Union

class BasisType(Enum):
    FOURIER = "fourier"
    CHEBYSHEV = "chebyshev"
    LEGENDRE = "legendre"
    HERMITE = "hermite"
    LAGUERRE = "laguerre"
    WAVELET = "wavelet"
    SINC = "sinc"
    RBF = "rbf"
    BESSEL = "bessel"


class OrthonormalBasisBank(nn.Module):
    """
    Orthonormal basis function bank for positional embeddings.
    Optimized with QR decomposition and interpolation.
    """

    def __init__(self,
                 basis_types: List[Union[str, BasisType]],
                 num_functions: int = 8,
                 domain_size: int = 256):
        super().__init__()

        # Validate domain size
        if domain_size < num_functions:
            raise ValueError(f"domain_size ({domain_size}) must be >= num_functions ({num_functions})")

        # Convert to BasisType objects
        self.basis_types = [BasisType(bt) if isinstance(bt, str) else bt for bt in basis_types]
        self.num_functions = num_functions
        self.domain_size = domain_size
        self.num_basis = len(self.basis_types)

        # Pre-compute basis function values
        self.register_buffer('domain_points', torch.linspace(0, 1, domain_size, dtype=torch.float32))
        basis_values = torch.zeros(self.num_basis, num_functions, domain_size)
        self.register_buffer('basis_values', basis_values)

        self._compute_basis_functions()
        self._orthonormalize()

    def _compute_basis_functions(self):
        x = self.domain_points

        for i, basis_type in enumerate(self.basis_types):
            if basis_type == BasisType.FOURIER:
                self.basis_values[i] = self._fourier_basis(x)
            elif basis_type == BasisType.CHEBYSHEV:
                self.basis_values[i] = self._chebyshev_basis(x)
            elif basis_type == BasisType.LEGENDRE:
                self.basis_values[i] = self._legendre_basis(x)
            elif basis_type == BasisType.HERMITE:
                self.basis_values[i] = self._hermite_basis(x)
            elif basis_type == BasisType.LAGUERRE:
                self.basis_values[i] = self._laguerre_basis(x)
            elif basis_type == BasisType.WAVELET:
                self.basis_values[i] = self._wavelet_basis(x)
            elif basis_type == BasisType.SINC:
                self.basis_values[i] = self._sinc_basis(x)
            elif basis_type == BasisType.RBF:
                self.basis_values[i] = self._rbf_basis(x)
            elif basis_type == BasisType.BESSEL:
                self.basis_values[i] = self._bessel_basis(x)
            else:
                raise ValueError(f"Unknown basis type: {basis_type}")

    # Improved basis implementations with normalization
    def _fourier_basis(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.zeros(self.num_functions, len(x))
        for n in range(self.num_functions):
            freq = (n // 2) + 1
            if n % 2 == 0:
                basis[n] = torch.cos(2 * math.pi * freq * x)
            else:
                basis[n] = torch.sin(2 * math.pi * freq * x)
        return basis / math.sqrt(len(x))

    def _chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        t = 2 * x - 1
        basis = torch.zeros(self.num_functions, len(x))
        if self.num_functions > 0: basis[0] = torch.ones_like(t)
        if self.num_functions > 1: basis[1] = t
        for n in range(2, self.num_functions):
            basis[n] = 2 * t * basis[n - 1] - basis[n - 2]
        return basis / math.sqrt(len(x))

    def _normalized_hermite(self, x, n):
        """Normalized Hermite polynomial with Gaussian envelope"""
        # Using physicist's Hermite polynomials
        coeff = 1 / math.sqrt(math.factorial(n) * math.sqrt(math.pi) * 2 ** n)
        return coeff * torch.exp(-x ** 2 / 2) * self._hermite_poly(x, n)

    def _hermite_poly(self, x, n):
        """Recursive computation of Hermite polynomials"""
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            return 2 * x * self._hermite_poly(x, n - 1) - 2 * (n - 1) * self._hermite_poly(x, n - 2)

    def _hermite_basis(self, x: torch.Tensor) -> torch.Tensor:
        t = 4 * (x - 0.5)  # Center and scale
        basis = torch.zeros(self.num_functions, len(x))
        for n in range(self.num_functions):
            basis[n] = self._normalized_hermite(t, n)
        return basis

    # Other basis functions with similar normalization improvements...
    # [Implementation details for other bases omitted for brevity]

    def _orthonormalize(self):
        """Apply QR decomposition for orthonormalization"""
        with torch.no_grad():
            for i in range(self.num_basis):
                basis_set = self.basis_values[i]  # (num_functions, domain_size)

                # Add small noise for stability
                basis_set += 1e-6 * torch.randn_like(basis_set)

                # Apply pivoted QR
                Q, R = torch.linalg.qr(
                    basis_set.T,
                    # pivoting=True,
                    mode='reduced'
                )
                self.basis_values[i] = Q.T

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Evaluate basis functions with linear interpolation"""
        # Flatten and clamp distances
        original_shape = distances.shape
        distances_flat = distances.reshape(-1)
        distances_flat = torch.clamp(distances_flat, 0.0, 1.0 - 1e-6)

        # Compute interpolation indices
        idx_float = distances_flat * (self.domain_size - 1)
        idx0 = idx_float.floor().long()
        idx1 = idx_float.ceil().long()
        alpha = idx_float - idx0.float()

        # Gather values
        basis_responses = torch.zeros(
            distances_flat.size(0),
            self.num_basis,
            self.num_functions,
            device=distances.device
        )

        for i in range(self.num_basis):
            vals = self.basis_values[i]  # (num_functions, domain_size)
            val0 = vals[:, idx0].permute(1, 0)  # (num_points, num_functions)
            val1 = vals[:, idx1].permute(1, 0)
            interpolated = val0 * (1 - alpha.unsqueeze(-1)) + val1 * alpha.unsqueeze(-1)
            basis_responses[:, i, :] = interpolated

        return basis_responses.reshape(*original_shape, self.num_basis, self.num_functions)


# Multi-Scale Orthonormal MHA
class OrthonormalMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with:
    - Orthonormal basis functions for positional encoding
    - Multi-scale distance awareness
    - Memory-efficient relative positioning
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 basis_types: Optional[List[str]] = None,
                 num_functions: int = 8,
                 head_basis_constraints: Optional[Dict[int, List[int]]] = None,
                 basis_domain_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        # Model dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Basis configuration
        if basis_types is None:
            basis_types = ['fourier', 'chebyshev', 'hermite']
        self.basis_bank = OrthonormalBasisBank(
            basis_types, num_functions, basis_domain_size
        )
        self.num_basis_types = len(basis_types)

        # Head constraints and scaling
        self.head_basis_constraints = self._create_default_constraints() if head_basis_constraints is None else head_basis_constraints
        self.distance_scalers = nn.Parameter(torch.linspace(-2, 2, num_heads))

        # Attention layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Positional projection heads
        self.pos_proj_heads = nn.ModuleList()
        for head_idx in range(num_heads):
            allowed_basis_indices = self.head_basis_constraints[head_idx]
            total_features = len(allowed_basis_indices) * num_functions

            if total_features > 0:
                self.pos_proj_heads.append(nn.Sequential(
                    nn.Linear(total_features, max(4, total_features // 2)),
                    nn.GELU(),
                    nn.Linear(max(4, total_features // 2), 1)
                ))
            else:
                # Dummy module for heads with no basis
                self.pos_proj_heads.append(nn.Identity())

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _create_default_constraints(self) -> Dict[int, List[int]]:
        constraints = {}
        for head_idx in range(self.num_heads):
            # Round-robin assignment of basis types
            start_idx = head_idx % self.num_basis_types
            end_idx = min(start_idx + max(1, self.num_basis_types // self.num_heads), self.num_basis_types)
            constraints[head_idx] = list(range(start_idx, end_idx))
        return constraints

    def _init_weights(self):
        # Initialize distance scalers with logarithmic distribution
        with torch.no_grad():
            scalers = torch.linspace(-2, 2, self.num_heads)
            self.distance_scalers.data.copy_(scalers)

        # Initialize attention projections
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Initialize positional heads
        for head in self.pos_proj_heads:
            if isinstance(head, nn.Sequential):
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

    def _compute_positional_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Memory-efficient positional bias computation"""
        # Compute unique distances [0 to seq_len-1]
        max_dist = max(1, seq_len - 1)
        unique_dists = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Head-specific bias vectors
        head_biases = []

        for head_idx in range(self.num_heads):
            # Apply head-specific distance scaling
            scale_factor = torch.sigmoid(self.distance_scalers[head_idx]) * 5 + 0.1
            scaled_dists = unique_dists * scale_factor
            norm_dists = scaled_dists / scaled_dists.max().clamp(min=1e-6)

            # Get basis responses: [seq_len, num_basis, num_func]
            basis_resp = self.basis_bank(norm_dists)

            # Filter by allowed basis indices
            allowed_basis_indices = self.head_basis_constraints[head_idx]
            if allowed_basis_indices:
                head_features = basis_resp[:, allowed_basis_indices, :]
                head_features = head_features.flatten(start_dim=1)  # [seq_len, total_features]
                bias_vector = self.pos_proj_heads[head_idx](head_features).squeeze(-1)
            else:
                bias_vector = torch.zeros(seq_len, device=device)

            head_biases.append(bias_vector)

        # Create full bias matrix using relative positions
        head_bias_tensor = torch.stack(head_biases)  # [num_heads, seq_len]
        rel_pos = torch.abs(torch.arange(seq_len, device=device)[:, None] -
                            torch.arange(seq_len, device=device)[None, :])
        return head_bias_tensor[:, rel_pos]  # [num_heads, seq_len, seq_len]

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add positional bias
        pos_bias = self._compute_positional_bias(seq_len, x.device).unsqueeze(0)
        scores = scores + pos_bias

        # Apply attention masks
        if is_causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask.unsqueeze(1), -1e9)
            else:
                scores = scores + attn_mask

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(context)
        return output, attn_weights


# Your existing config dictionary (example structure, provide your actual one)
config = {
    'n_embd': 512,
    'n_head': 8,
    'dropout': 0.1,
    'pos_num_filters': 64,  # This will be replaced by OrthonormalMHA params
    'pos_num_taps': 5,  # This will be replaced
    'pos_resolution_steps': 128,  # This will be replaced by basis_domain_size
    'pos_filter_fs': 1.0,  # This will be replaced

    # For OrthonormalMultiHeadAttention
    'ortho_basis_types': ['fourier'],  # Example
    'ortho_num_functions_per_basis': 8,  # Example
    'ortho_basis_domain_size': 256,  # Example

    'vocab_size': 50257,
    'n_layer': 12,
    'ctx_len': 1024,
    'rms_norm_eps': 1e-5,
    'type': ["mlp"] * 6 + ["moe"] * 6,  # Example layer types
    'n_experts': 4,  # For MoE
    'moe_top_k': 2,  # For MoE
    'tie_weights': True,
}


class SimpleMHA(nn.Module):
    def __init__(self, Cfg):
        super().__init__()
        self.config = Cfg  # Use the passed Cfg
        self.n_embd = self.config['n_embd']
        self.n_head = self.config['n_head']
        self.dropout_val = self.config['dropout']

        # Use OrthonormalMultiHeadAttention
        self.mha = OrthonormalMultiHeadAttention(
            d_model=self.n_embd,
            num_heads=self.n_head,
            basis_types=self.config.get('ortho_basis_types', ['fourier']),  # Provide defaults
            num_functions=self.config.get('ortho_num_functions_per_basis', 8),
            basis_domain_size=self.config.get('ortho_basis_domain_size', 256),
            dropout=self.dropout_val
        )
        # Positional encoding is now handled internally by OrthonormalMultiHeadAttention
        # No need for self.pos_encoder, self.pos_proj, or caching of pos_bias here
        # as OrthonormalMultiHeadAttention computes it on the fly.

        self.res_dropout = nn.Dropout(p=self.dropout_val)

    # _get_full_pos_bias is no longer needed here, as OrthonormalMHA handles it.

    def forward(self, x, is_causal=True):  # Added is_causal flag, common for decoders
        B, T, C = x.shape
        if T == 0:
            return torch.zeros(B, 0, C, device=x.device, dtype=x.dtype)

        # OrthonormalMultiHeadAttention handles its own positional bias.
        # It also handles the causal mask internally if is_causal is True.
        # No external attn_mask is passed from SimpleMHA by default, but could be.
        # The `attn_mask` parameter in `OrthonormalMultiHeadAttention.forward` can be used
        # for padding masks if needed. Here, we are primarily relying on `is_causal`.

        attn_output, _ = self.mha(
            x=x,
            attn_mask=None,  # Or pass a padding mask if you have one
            is_causal=is_causal  # Crucial for autoregressive models
        )
        return self.res_dropout(attn_output)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        n_embd = config['n_embd']  # Assuming global config for MLP/MoE/Block
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        # SwiGLU variant: x * F.silu(gate) or Squared ReLU
        x = F.gelu(x)  # As per your original
        # x = F.gelu(x) # Common alternative
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class UnitCenteredNoise(nn.Module):
    def __init__(self, scaling=0.02):
        super(UnitCenteredNoise, self).__init__()
        self.scaling = scaling
        # Corrected base for centering around 1: (1 - scaling/2) to (1 + scaling/2)
        self.offset = 1.0 - (scaling / 2.0)

    def forward(self, x):
        if self.training:
            noise = torch.rand_like(x)  # Use rand_like for same shape, device, dtype
            noise_scaled_shifted = (noise * self.scaling) + self.offset
            return x * noise_scaled_shifted
        else:
            # During inference, typically no noise or a deterministic factor (e.g., multiply by 1.0)
            return x


class DSMoE(nn.Module):
    def __init__(self, index, num_exp_routed_to=None):  # index might not be needed if config is global
        super().__init__()
        self.num_total_experts = config["n_experts"]
        if num_exp_routed_to is None:
            num_exp_routed_to = config['moe_top_k']
        self.num_experts_to_route_to = num_exp_routed_to

        self.experts = nn.ModuleList([MLP() for _ in range(self.num_total_experts)])
        self.gate = nn.Sequential(
            nn.Linear(config['n_embd'], self.num_total_experts, bias=False),
            UnitCenteredNoise(scaling=config.get('moe_gate_noise_scale', 0.02)),  # Make noise scale configurable
        )

    def forward(self, x):
        b, t, c = x.shape
        x_flat = x.reshape(b * t, c)  # (N, C) where N = B*T

        gate_logits = self.gate(x_flat)  # (N, num_total_experts)

        # Softmax over experts for each token
        gate_probs = F.softmax(gate_logits, dim=-1)  # (N, num_total_experts)

        # Select top-k experts and their weights
        # top_k_weights: (N, num_experts_to_route_to)
        # top_k_indices: (N, num_experts_to_route_to)
        top_k_weights, top_k_indices = torch.topk(gate_probs, self.num_experts_to_route_to, dim=-1, sorted=False)

        # Normalize the weights of the selected top-k experts so they sum to 1
        top_k_weights_norm = top_k_weights / torch.clamp(top_k_weights.sum(dim=-1, keepdim=True), min=1e-6)

        # Initialize output tensor
        output_flat = torch.zeros_like(x_flat)

        # Efficiently route tokens to experts
        # Create a flat batch_idx for advanced indexing
        flat_expert_indices = top_k_indices.flatten()  # (N * num_experts_to_route_to)
        flat_router_weights = top_k_weights_norm.flatten()  # (N * num_experts_to_route_to)

        # Repeat tokens for each expert they are routed to
        # x_flat_repeated: (N * num_experts_to_route_to, C)
        expanded_x_flat = x_flat.repeat_interleave(self.num_experts_to_route_to, dim=0)

        # Dispatch tokens to experts and compute outputs
        expert_outputs_collected = torch.zeros_like(expanded_x_flat)

        for expert_idx in range(self.num_total_experts):
            mask = (flat_expert_indices == expert_idx)
            if mask.any():
                tokens_for_this_expert = expanded_x_flat[mask]
                expert_outputs_collected[mask] = self.experts[expert_idx](tokens_for_this_expert)

        # Weight the expert outputs and sum them up
        weighted_expert_outputs = expert_outputs_collected * flat_router_weights.unsqueeze(-1)

        # Sum outputs for tokens that were routed to multiple experts (if any, though top-k usually unique)
        # Reshape back to (N, num_experts_to_route_to, C) and sum over num_experts_to_route_to
        output_flat = weighted_expert_outputs.view(b * t, self.num_experts_to_route_to, c).sum(dim=1)

        # For auxiliary loss (load balancing) - sparse router weights
        # This part is for potential auxiliary losses, not directly for the forward output value.
        router_weights_sparse = torch.zeros(x_flat.size(0), self.num_total_experts, device=x.device,
                                            dtype=top_k_weights_norm.dtype)
        # Scatter the normalized weights of the chosen experts
        router_weights_sparse.scatter_(1, top_k_indices, top_k_weights_norm)

        return output_flat.reshape(b, t, c), router_weights_sparse


class Block(nn.Module):
    def __init__(self, index):  # Pass index if Cfg for SimpleMHA needs to be layer-specific
        super().__init__()
        n_embd = config['n_embd']
        # MODIFICATION: Use SimpleMHA and pass the global config
        self.attn = SimpleMHA(config)  # Pass the global config dictionary
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
        # In decoder blocks, is_causal=True is standard for self-attention
        # For encoder blocks or cross-attention, it would be False.
        # Assuming this is a decoder block.
        x_attn = self.attn(self.rm1(x), is_causal=True)
        x = x + x_attn

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
        self.config = config  # Use global config
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
            # Standard deviation from GPT-2 paper; adjust if needed
            std_dev = 0.02
            # For residual projections in Transformer blocks (like in GPT-2/3)
            if hasattr(module, 'is_residual_proj') and module.is_residual_proj:
                std_dev /= math.sqrt(2 * self.config['n_layer'])  # Scale down for deep networks
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_dev)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Mark residual projections (output projection of MHA and FFN)
        # This is a common practice for better initialization scaling in deep transformers.
        # You'd need to identify these layers. For OrthonormalMHA, it's self.out_proj.
        # For MLP, it's self.c_proj.
        if hasattr(module, 'out_proj') and isinstance(module.out_proj, nn.Linear):  # For OrthonormalMHA
            module.out_proj.is_residual_proj = True
        if hasattr(module, 'c_proj') and isinstance(module.c_proj, nn.Linear):  # For MLP
            module.c_proj.is_residual_proj = True

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        x = tok_emb  # In some architectures, positional encoding is added here.
        # OrthonormalMHA adds its bias directly to attention scores.

        all_router_weights = []
        for block_module in self.blocks:
            x, router_weights = block_module(x)
            if router_weights is not None:
                all_router_weights.append(router_weights)

        x = self.rm_f(x)
        logits = self.lm_head(x)  # (B, T, VocabSize)

        loss = None
        if targets is not None:
            # Reshape for cross_entropy: (B*T, VocabSize) and (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)  # common to ignore padding

        return logits, loss, all_router_weights

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, tiktoken_vocab_size=None):
        self.eval()  # Set model to evaluation mode

        for _ in range(max_new_tokens):
            # Crop idx to the last 'ctx_len' tokens if it's longer
            idx_cond = idx if idx.size(1) <= self.config['ctx_len'] else idx[:, -self.config['ctx_len']:]

            logits, _, _ = self(idx_cond)  # Forward pass

            # Get logits for the last token
            logits = logits[:, -1, :]  # (B, VocabSize)

            if temperature == 0.0:  # Greedy decoding
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  # Apply temperature

                # Filter logits for tiktoken_vocab_size if specified
                if tiktoken_vocab_size is not None and tiktoken_vocab_size < logits.size(-1):
                    logits[:, tiktoken_vocab_size:] = float('-inf')

                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # (B, min(top_k, V))
                    # Set logits below the k-th smallest to -inf
                    logits[logits < v[:, [-1]]] = float('-inf')  # Broadcast comparison

                probs = F.softmax(logits, dim=-1)  # (B, VocabSize)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)  # Append predicted token

        self.train()  # Set model back to training mode
        return idx  # Return the generated sequence (no dummy kv_cache_gb)

    def configure_optimizers(self, weight_decay, learning_rate, device_type_str):
        # Separate parameters for weight decay
        decay_params = []
        nodecay_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                if p.ndim >= 2:  # Typically, weights of Linear and Embedding layers
                    decay_params.append(p)
                else:  # Typically, biases and LayerNorm/RMSNorm parameters
                    nodecay_params.append(p)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Check for fused AdamW support
        use_fused = False
        if device_type_str == 'cuda':
            # The 'fused' argument was added to AdamW in PyTorch 1.9 or 1.10.
            # More robust check:
            try:
                # Create a dummy parameter on CUDA to test
                dummy_param = nn.Parameter(torch.randn(1, device='cuda'))
                # Attempt to create an optimizer with fused=True
                torch.optim.AdamW([dummy_param], lr=1e-4, fused=True)
                use_fused = True
            except Exception:  # Catches RuntimeError if fused is not supported or CUDA not available
                use_fused = False

        optimizer_kwargs = {'lr': learning_rate, 'betas': (0.9, 0.95), 'eps': 1e-8}  # Added eps for stability
        if use_fused:
            optimizer_kwargs['fused'] = True

        optimizer = torch.optim.AdamW(optim_groups, **optimizer_kwargs)
        print(f"Using Fused AdamW: {use_fused and optimizer_kwargs.get('fused', False)}")
        return [optimizer]  # Return the optimizer directly, not in a list unless multiple optimizers

    def update_expert_biases(self, all_router_weights_list, update_rate):
        # This is a placeholder for potential MoE expert bias updates, common in some MoE papers.
        # For example, to encourage experts to be utilized.
        # The actual implementation depends on the specific load balancing or bias update strategy.
        if not all_router_weights_list:
            return

        # Example: Calculate average utilization per expert across layers/batches
        # This is a very simplified example and would need proper aggregation if used.
        # total_expert_load = torch.zeros(config['n_experts'], device=all_router_weights_list[0].device)
        # num_tokens_total = 0
        # for router_weights_batch_layer in all_router_weights_list: # List of (B*T, num_experts)
        #     total_expert_load += router_weights_batch_layer.sum(dim=0) # Sum over tokens
        #     num_tokens_total += router_weights_batch_layer.size(0)

        # if num_tokens_total > 0:
        #     avg_expert_load = total_expert_load / num_tokens_total
        #     # print(f"Average expert load: {avg_expert_load}")
        #     # Here you might adjust biases in the gate or experts based on load.
        pass

    def estimate_mfu(self, num_params_training, fwdbwd_per_iter, dt_s):
        # num_params_training: number of parameters being updated by the optimizer
        # fwdbwd_per_iter: 1 if not using gradient accumulation, else accumulation_steps
        # dt_s: time delta for one iteration in seconds

        N = num_params_training
        L = self.config['ctx_len']  # Sequence length
        H = self.config['n_layer']  # Number of layers
        Q = self.config['n_head']  # Number of heads
        C = self.config['n_embd']  # Embedding dimension
        V = self.config['vocab_size']

        # More detailed FLOPs estimation (Koyanaka et al. 2023, PaLM paper style):
        # For a dense transformer layer:
        # Self-attention: 2 * B * L * C * (2 * C) (for QKV proj) + 2 * B * Q * L^2 * (C/Q) (for scores) + 2 * B * Q * L^2 * (C/Q) (for weighted sum) + 2 * B * L * C * C (for output proj)
        # Simplified: Roughly 4 * B * L * C^2 (QKV, Out) + 4 * B * L^2 * C (Attention)
        # MLP: 2 * B * L * C * (4*C) (for fc1) + 2 * B * L * (4*C) * C (for fc2) = 16 * B * L * C^2
        # Total per layer (dense): ~20 * B * L * C^2 + 4 * B * L^2 * C
        # If using MoE, MLP part changes. For top-k MoE: k * (MLP_flops_per_expert)
        # For simplicity, using the common 6*N*B*L estimate for a forward pass
        # or 2*N*B*L for parameters (N) that are directly proportional to B*L operations.
        # Kaplan et al. (2020) "Scaling Laws for Neural Language Models" uses ~6*N FLOPs per token for fwd pass.

        # Using the 6*N estimate (N = total model parameters, not just trainable)
        # This N should be total model parameters, not just trainable, for MFU based on model size.
        N_total_model_params = sum(p.numel() for p in self.parameters())

        flops_per_token_fwd = 6 * N_total_model_params

        # Assuming batch_size is implicitly handled by the training loop's `fwdbwd_per_iter`
        # Let's assume `fwdbwd_per_iter` effectively gives number of sequences processed per optimizer step.
        # sequences_per_optimizer_step = micro_batch_size * gradient_accumulation_steps
        # Here, fwdbwd_per_iter seems to be just gradient_accumulation_steps.
        # So, the actual batch size needs to be known. Let's assume an effective batch size B_eff.
        # B_eff = micro_batch_size * gradient_accumulation_steps
        # For now, let's assume the `dt_s` is for processing `B_eff` sequences of length `L`.
        # And `fwdbwd_per_iter` is the number of forward/backward passes per optimizer step.
        # This estimation is tricky without knowing the exact batching strategy.

        # Let's use a simpler variant focusing on parameters updated:
        # If N is trainable parameters:
        # FLOPs for fwd pass ~ 2 * N (matrix-vector products dominate) per token
        # FLOPs for fwd + bwd ~ 3 * (2 * N) = 6 * N per token.
        # Total FLOPs per optimizer step = 6 * N_trainable_params * L_sequence_length * Effective_Batch_Size_per_step

        # Given the function signature, it seems N is trainable parameters.
        # fwdbwd_per_iter: if 1, means dt is for one fwd+bwd.
        # If > 1 (grad_accum_steps), dt is for grad_accum_steps * (fwd+bwd).
        # Let's assume dt is for ONE optimizer step, which includes `fwdbwd_per_iter` micro-batches.
        # And the batch size of each micro-batch is B_micro.
        # Effective batch size for the optimizer step = B_micro * fwdbwd_per_iter
        # The MFU needs to relate to a peak FLOPs of the hardware.

        # Using the 6*N (total params) per token for fwd.
        # Assume `dt` is time for `fwdbwd_per_iter` *micro_batches* of size `B_micro`.
        # Let's assume current `idx.shape` in forward gives `B_micro` and `L`.
        # This MFU is usually calculated per iteration (optimizer step).

        # From PaLM paper (Appendix D): Training FLOPs ~ 6 * B * s * N
        # B: batch size in tokens (batch_of_sequences * sequence_length)
        # s: sequence length
        # N: non-embedding parameters (roughly total_params * 2/3 if embeddings are 1/3)
        # This is for one forward pass. Fwd+Bwd is ~3x this. So ~18 * B * s * N_non_embedding
        # Or using total_params: ~ 6 * total_params * num_tokens_in_batch for FWD
        # ~ 18 * total_params * num_tokens_in_batch for FWD+BWD

        # Let num_tokens_processed_per_step = effective_batch_size_sequences * L
        # Effective_batch_size_sequences = actual_micro_batch_size * fwdbwd_per_iter (grad_accum_steps)
        # This MFU function needs the actual micro_batch_size to be accurate.
        # Let's assume `fwdbwd_per_iter` is `gradient_accumulation_steps` and that `dt` is for one optimizer step.
        # The `num_params` passed is `self.total_params` (trainable).

        # Assume `B_micro` (micro_batch_size, number of sequences per fwd/bwd pass) is needed.
        # If `B_micro` is, for example, 8:
        B_micro = config.get('micro_batch_size', 8)  # Example, should come from training script
        num_tokens_per_optimizer_step = B_micro * fwdbwd_per_iter * L

        # Using 6*N_total_params per token for FWD pass, so 18*N_total_params per token for FWD+BWD.
        # N_total_model_params = sum(p.numel() for p in self.parameters()) # Already calculated or can be passed

        flops_per_iter_total = 18 * N_total_model_params * num_tokens_per_optimizer_step

        flops_achieved_per_second = flops_per_iter_total / dt_s

        # GPU Peak FLOPs (e.g., A100 BF16 Tensor Core ~312 TFLOPs, FP32 ~19.5 TFLOPs)
        # This should be configured based on the hardware.
        # Let's use a placeholder for BF16 TFLOPs.
        peak_flops_gpu = config.get('gpu_peak_flops_bf16', 312e12)  # e.g., A100 BF16
        # If using FP32, use a different peak.

        mfu = flops_achieved_per_second / peak_flops_gpu
        return mfu


if __name__ == '__main__':
    print("--- Testing OrthonormalBasisBank ---")
    try:
        obb_test = OrthonormalBasisBank(
            basis_types=['fourier', 'hermite', 'wavelet'],
            num_functions=4,
            domain_size=32  # Small domain for quick test
        )
        test_dist = torch.linspace(0, 1, 10, device='cpu')  # Test on CPU first
        obb_responses = obb_test(test_dist)
        print(f"OrthonormalBasisBank output shape: {obb_responses.shape}")  # (10, 3, 4)
        # Check orthonormality of raw basis_values
        for i in range(obb_test.num_basis):
            basis_set = obb_test.basis_values[i]  # (num_func, domain_size)
            if basis_set.shape[0] > 0 and basis_set.shape[1] > 0:
                # Normalize rows before checking Gram matrix if they aren't already unit norm
                # basis_set_norm = basis_set / torch.norm(basis_set, dim=1, keepdim=True).clamp(min=1e-6)
                # gram_matrix = torch.matmul(basis_set_norm, basis_set_norm.T)
                # For QR, columns of Q are orthonormal. Q.T rows are orthonormal.
                # So basis_set itself should have orthonormal rows.
                gram_matrix = torch.matmul(basis_set, basis_set.T)
                identity_approx = torch.eye(basis_set.shape[0], device=basis_set.device)
                # For domain_size < num_functions, gram_matrix might not be full identity.
                # It will be identity up to min(num_functions, domain_size)
                expected_rank = min(basis_set.shape[0], basis_set.shape[1])
                identity_error = torch.norm(
                    gram_matrix[:expected_rank, :expected_rank] - torch.eye(expected_rank, device=basis_set.device))
                print(
                    f"Basis {obb_test.basis_types[i].value} pre-computed orthonormality error (rank {expected_rank}): {identity_error.item():.4f}")

    except Exception as e:
        print(f"Error in OrthonormalBasisBank test: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Testing OrthonormalMultiHeadAttention ---")
    try:
        omha_test = OrthonormalMultiHeadAttention(
            d_model=64,
            num_heads=4,
            basis_types=['fourier', 'chebyshev'],
            num_functions=8,
            basis_domain_size=128,
            dropout=0.0
        ).to('cpu')
        test_x_omha = torch.randn(2, 16, 64, device='cpu')  # B, T, C
        omha_output, omha_attn_weights = omha_test(test_x_omha, is_causal=True)
        print(f"OrthonormalMHA output shape: {omha_output.shape}")  # (2, 16, 64)
        print(f"OrthonormalMHA attention weights shape: {omha_attn_weights.shape}")  # (2, 4, 16, 16)
    except Exception as e:
        print(f"Error in OrthonormalMultiHeadAttention test: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Testing SimpleMHA with OrthonormalMHA ---")
    # Create a minimal config for SimpleMHA
    simple_mha_config = {
        'n_embd': 64,
        'n_head': 4,
        'dropout': 0.0,
        'ortho_basis_types': ['fourier', 'chebyshev'],
        'ortho_num_functions_per_basis': 6,
        'ortho_basis_domain_size': 64,
    }
    try:
        smha_test = SimpleMHA(Cfg=simple_mha_config).to('cpu')
        test_x_smha = torch.randn(2, 10, 64, device='cpu')
        smha_output = smha_test(test_x_smha, is_causal=True)
        print(f"SimpleMHA output shape: {smha_output.shape}")  # (2, 10, 64)
    except Exception as e:
        print(f"Error in SimpleMHA test: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Testing Full Transformer Model ---")
    # Use the global 'config' defined earlier for the full Transformer test
    try:
        model = Transformer().to('cpu')  # Test on CPU
        print(f"Transformer model created with {model.total_params:,} parameters.")

        test_idx = torch.randint(0, config['vocab_size'], (2, 32), device='cpu')  # B, T
        test_targets = torch.randint(0, config['vocab_size'], (2, 32), device='cpu')  # B, T

        logits, loss, router_weights = model(test_idx, test_targets)
        print(f"Logits shape: {logits.shape}")  # (B, T, VocabSize)
        if loss is not None:
            print(f"Loss: {loss.item()}")
        if router_weights:
            print(f"Number of MoE router weight tensors: {len(router_weights)}")
            print(f"Shape of first router_weights: {router_weights[0].shape}")  # (B*T, num_experts)

        print("\n--- Testing Generation ---")
        generated_ids = model.generate(idx=test_idx[:, :5], max_new_tokens=5)  # Generate 5 new tokens
        print(f"Generated IDs shape: {generated_ids[0].shape}")  # (B, T_original + 5)

        print("\n--- Testing Optimizer Configuration ---")
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-4, device_type_str='cpu')
        print(f"Optimizer configured: {type(optimizer)}")

        print("\n--- Testing MFU Estimation ---")
        # Dummy values for MFU test
        mfu = model.estimate_mfu(num_params_training=model.total_params, fwdbwd_per_iter=1, dt_s=0.1)
        print(f"Estimated MFU: {mfu * 100:.2f}%")


    except Exception as e:
        print(f"Error in Transformer test: {e}")
        import traceback

        traceback.print_exc()