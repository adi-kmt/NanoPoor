import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import List, Optional, Dict, Tuple, Union
from functools import lru_cache
import numpy as np


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

    def _fourier_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized Fourier basis with vectorized computation"""
        basis = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        
        # Vectorized computation
        n_indices = torch.arange(self.num_functions, device=x.device)
        freqs = (n_indices // 2) + 1
        
        cos_mask = (n_indices % 2 == 0)
        sin_mask = ~cos_mask
        
        if cos_mask.any():
            cos_freqs = freqs[cos_mask]
            basis[cos_mask] = torch.cos(2 * math.pi * cos_freqs[:, None] * x[None, :])
        
        if sin_mask.any():
            sin_freqs = freqs[sin_mask]
            basis[sin_mask] = torch.sin(2 * math.pi * sin_freqs[:, None] * x[None, :])
        
        return basis / math.sqrt(len(x))

    def _chebyshev_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized Chebyshev basis with iterative computation"""
        t = 2 * x - 1
        basis = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        
        if self.num_functions > 0: 
            basis[0] = torch.ones_like(t)
        if self.num_functions > 1: 
            basis[1] = t
        
        # Iterative computation is more stable than recursion
        for n in range(2, self.num_functions):
            basis[n] = 2 * t * basis[n - 1] - basis[n - 2]
        
        return basis / math.sqrt(len(x))

    def _legendre_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Legendre polynomials using iterative computation"""
        t = 2 * x - 1  # Map [0,1] to [-1,1]
        basis = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        
        if self.num_functions > 0:
            basis[0] = torch.ones_like(t)
        if self.num_functions > 1:
            basis[1] = t
        
        # Bonnet's recursion formula: (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
        for n in range(2, self.num_functions):
            basis[n] = ((2*n - 1) * t * basis[n-1] - (n-1) * basis[n-2]) / n
        
        # Normalize
        for n in range(self.num_functions):
            basis[n] = basis[n] * math.sqrt((2*n + 1) / 2)  # L2 normalization on [-1,1]
        
        return basis / math.sqrt(len(x))

    def _hermite_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Hermite polynomials with iterative computation and proper normalization"""
        t = 4 * (x - 0.5)  # Center and scale
        basis = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        
        # Compute unnormalized Hermite polynomials iteratively
        H = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        if self.num_functions > 0:
            H[0] = torch.ones_like(t)
        if self.num_functions > 1:
            H[1] = 2 * t
        
        for n in range(2, self.num_functions):
            H[n] = 2 * t * H[n-1] - 2 * (n-1) * H[n-2]
        
        # Apply normalization with Gaussian envelope
        gaussian = torch.exp(-t**2 / 2)
        for n in range(self.num_functions):
            # Normalization constant for physicist's Hermite
            norm_const = 1 / math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
            basis[n] = norm_const * gaussian * H[n]
        
        return basis

    def _laguerre_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Laguerre polynomials with proper normalization"""
        # Map [0,1] to [0, inf) with a reasonable scaling
        t = 8 * x  # Scale factor for better coverage
        basis = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        
        # Compute Laguerre polynomials iteratively
        L = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        if self.num_functions > 0:
            L[0] = torch.ones_like(t)
        if self.num_functions > 1:
            L[1] = 1 - t
        
        for n in range(2, self.num_functions):
            L[n] = ((2*n - 1 - t) * L[n-1] - (n-1) * L[n-2]) / n
        
        # Apply exponential weight and normalization
        exp_weight = torch.exp(-t / 2)
        for n in range(self.num_functions):
            basis[n] = exp_weight * L[n]
        
        return basis / math.sqrt(len(x))

    def _wavelet_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Morlet wavelet basis with different scales"""
        basis = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        
        # Center the domain
        t = 8 * (x - 0.5)  # Scale to [-4, 4]
        
        for n in range(self.num_functions):
            # Different scales and translations
            scale = 2 ** (n // 2 - 1) if n > 0 else 1
            translation = (n % 2 - 0.5) * 2  # Alternate between -1 and 1
            
            scaled_t = (t - translation) / scale
            # Morlet wavelet: complex exponential times Gaussian
            basis[n] = torch.exp(-scaled_t**2 / 2) * torch.cos(5 * scaled_t) / math.sqrt(scale)
        
        return basis / math.sqrt(len(x))

    def _sinc_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Sinc function basis with different bandwidths"""
        basis = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        
        # Center around 0.5
        t = 10 * (x - 0.5)  # Scale to [-5, 5]
        
        for n in range(self.num_functions):
            bandwidth = (n + 1) * 0.5
            scaled_t = bandwidth * t
            
            # Avoid division by zero
            sinc_vals = torch.where(
                torch.abs(scaled_t) < 1e-6,
                torch.ones_like(scaled_t),
                torch.sin(scaled_t) / scaled_t
            )
            basis[n] = sinc_vals
        
        return basis / math.sqrt(len(x))

    def _rbf_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Radial Basis Function (Gaussian) with different centers and widths"""
        basis = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        
        for n in range(self.num_functions):
            # Different centers across [0, 1]
            center = n / max(1, self.num_functions - 1)
            # Different widths
            width = 0.1 + 0.05 * n
            
            basis[n] = torch.exp(-(x - center)**2 / (2 * width**2))
        
        return basis / math.sqrt(len(x))

    def _bessel_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Bessel function approximation using series expansion"""
        basis = torch.zeros(self.num_functions, len(x), device=x.device, dtype=x.dtype)
        
        # Scale x for better numerical properties
        t = 10 * x
        
        for n in range(self.num_functions):
            # Approximated Bessel function of the first kind J_n
            # Using first few terms of series expansion
            order = n
            result = torch.zeros_like(t)
            
            # Series expansion: J_n(x) = sum_{k=0}^inf (-1)^k / (k!(n+k)!) * (x/2)^(n+2k)
            for k in range(min(10, 20 - order)):  # Limit terms for efficiency
                term = ((-1)**k / (math.factorial(k) * math.factorial(order + k))) * \
                       ((t / 2)**(order + 2*k))
                result += term
            
            basis[n] = result
        
        return basis / math.sqrt(len(x))

    def _orthonormalize(self):
        """Apply QR decomposition for orthonormalization"""
        with torch.no_grad():
            for i in range(self.num_basis):
                basis_set = self.basis_values[i]  # (num_functions, domain_size)
                
                if basis_set.numel() == 0:
                    continue
                
                # Add small noise for numerical stability
                basis_set += 1e-8 * torch.randn_like(basis_set)
                
                # Apply QR decomposition
                Q, R = torch.linalg.qr(basis_set.T, mode='reduced')
                
                # Ensure we don't exceed the rank
                rank = min(basis_set.shape[0], basis_set.shape[1])
                self.basis_values[i] = Q.T[:rank]

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Optimized basis function evaluation with vectorized interpolation"""
        original_shape = distances.shape
        distances_flat = distances.reshape(-1)
        distances_flat = torch.clamp(distances_flat, 0.0, 1.0 - 1e-6)

        # Vectorized interpolation
        idx_float = distances_flat * (self.domain_size - 1)
        idx0 = idx_float.floor().long()
        idx1 = idx0 + 1
        idx1 = torch.clamp(idx1, max=self.domain_size - 1)  # Ensure bounds
        alpha = idx_float - idx0.float()

        # Efficient gathering and interpolation
        batch_size = distances_flat.size(0)
        basis_responses = torch.zeros(
            batch_size, self.num_basis, self.num_functions,
            device=distances.device, dtype=distances.dtype
        )

        for i in range(self.num_basis):
            vals = self.basis_values[i]  # (num_functions, domain_size)
            
            # Vectorized gather
            val0 = vals[:, idx0].T  # (batch_size, num_functions)
            val1 = vals[:, idx1].T  # (batch_size, num_functions)
            
            # Vectorized interpolation
            interpolated = val0 * (1 - alpha.unsqueeze(-1)) + val1 * alpha.unsqueeze(-1)
            basis_responses[:, i, :] = interpolated

        return basis_responses.reshape(*original_shape, self.num_basis, self.num_functions)


class OrthonormalMultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention with caching and efficient positional bias
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 basis_types: Optional[List[str]] = None,
                 num_functions: int = 8,
                 head_basis_constraints: Optional[Dict[int, List[int]]] = None,
                 basis_domain_size: int = 256,
                 dropout: float = 0.1,
                 max_seq_len: int = 2048):  # Add max sequence length for caching
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

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

        # Optimized positional projection heads
        self.pos_proj_heads = nn.ModuleList()
        for head_idx in range(num_heads):
            allowed_basis_indices = self.head_basis_constraints[head_idx]
            total_features = len(allowed_basis_indices) * num_functions

            if total_features > 0:
                # Simplified projection for efficiency
                self.pos_proj_heads.append(
                    nn.Linear(total_features, 1, bias=True)
                )
            else:
                self.pos_proj_heads.append(nn.Identity())

        self.dropout = nn.Dropout(dropout)
        
        # Cache for positional bias - key optimization!
        self._cached_pos_bias = {}
        self._cache_device = None
        
        self._init_weights()

    def _create_default_constraints(self) -> Dict[int, List[int]]:
        """More balanced constraint assignment"""
        constraints = {}
        basis_per_head = max(1, self.num_basis_types // self.num_heads)
        
        for head_idx in range(self.num_heads):
            start_idx = (head_idx * basis_per_head) % self.num_basis_types
            end_idx = min(start_idx + basis_per_head, self.num_basis_types)
            constraints[head_idx] = list(range(start_idx, end_idx))
            
            # Ensure all heads have at least one basis
            if not constraints[head_idx]:
                constraints[head_idx] = [head_idx % self.num_basis_types]
        
        return constraints

    def _init_weights(self):
        # More stable initialization
        std = 0.02
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=std)

        nn.init.normal_(self.out_proj.weight, std=std)
        nn.init.zeros_(self.out_proj.bias)

        # Initialize distance scalers
        with torch.no_grad():
            self.distance_scalers.data = torch.linspace(-1, 1, self.num_heads)

        # Initialize positional heads
        for head in self.pos_proj_heads:
            if isinstance(head, nn.Linear):
                nn.init.normal_(head.weight, std=std)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

    def _get_cached_positional_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Efficient caching of positional bias"""
        # Clear cache if device changed
        if self._cache_device != device:
            self._cached_pos_bias.clear()
            self._cache_device = device
        
        # Check cache
        cache_key = seq_len
        if cache_key in self._cached_pos_bias:
            return self._cached_pos_bias[cache_key]
        
        # Compute and cache
        pos_bias = self._compute_positional_bias(seq_len, device)
        
        # Only cache reasonable sizes to avoid memory issues
        if seq_len <= 1024:
            self._cached_pos_bias[cache_key] = pos_bias
        
        return pos_bias

    def _compute_positional_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Optimized positional bias computation"""
        # Compute relative positions more efficiently
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        rel_pos = torch.abs(positions[:, None] - positions[None, :])  # (seq_len, seq_len)
        
        # Normalize relative positions
        max_dist = seq_len - 1
        if max_dist > 0:
            norm_rel_pos = rel_pos / max_dist
        else:
            norm_rel_pos = rel_pos
        
        # Compute bias for each head efficiently
        head_biases = []
        
        for head_idx in range(self.num_heads):
            # Apply head-specific scaling
            scale_factor = torch.sigmoid(self.distance_scalers[head_idx]) * 3 + 0.1
            scaled_pos = norm_rel_pos * scale_factor
            
            # Get basis responses
            allowed_basis_indices = self.head_basis_constraints[head_idx]
            if allowed_basis_indices and isinstance(self.pos_proj_heads[head_idx], nn.Linear):
                # Flatten for basis computation
                flat_pos = scaled_pos.reshape(-1)  # (seq_len^2,)
                basis_resp = self.basis_bank(flat_pos)  # (seq_len^2, num_basis, num_func)
                
                # Filter and project
                head_features = basis_resp[:, allowed_basis_indices, :].flatten(start_dim=1)
                bias_flat = self.pos_proj_heads[head_idx](head_features).squeeze(-1)
                bias_matrix = bias_flat.reshape(seq_len, seq_len)
            else:
                bias_matrix = torch.zeros(seq_len, seq_len, device=device)
            
            head_biases.append(bias_matrix)
        
        return torch.stack(head_biases)  # (num_heads, seq_len, seq_len)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k ** -0.5)

        # Add cached positional bias
        pos_bias = self._get_cached_positional_bias(seq_len, x.device)
        scores = scores + pos_bias.unsqueeze(0)

        # Apply masks
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
            else:
                scores = scores + attn_mask

        # Attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.out_proj(context)
        return output, attn_weights


class OptimizedDSMoE(nn.Module):
    """
    Memory-efficient Mixture of Experts implementation
    """
    
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, 
                 dropout: float = 0.1, capacity_factor: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Gate network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks - using nn.ModuleList for proper registration
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model, bias=False),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        
        # Load balancing loss
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (B*T, d_model)
        
        # Gate computation
        gate_logits = self.gate(x_flat)  # (B*T, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Efficient expert computation
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_id in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_id)
            if not expert_mask.any():
                continue
                
            # Get token indices and weights for this expert
            token_indices, k_indices = torch.where(expert_mask)
            if len(token_indices) == 0:
                continue
                
            expert_weights = top_k_probs[token_indices, k_indices]
            expert_input = x_flat[token_indices]
            
            # Compute expert output
            expert_output = self.experts[expert_id](expert_input)
            
            # Accumulate weighted output
            output[token_indices] += expert_weights.unsqueeze(-1) * expert_output
            
        # Update expert usage statistics for load balancing
        if self.training:
            with torch.no_grad():
                expert_usage = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts).float()
                self.expert_counts += expert_usage
        
        # Compute load balancing loss
        router_probs = gate_probs.mean(dim=0)  # Average probability per expert
        expert_usage_frac = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts).float()
        expert_usage_frac = expert_usage_frac / expert_usage_frac.sum()
        
        load_balance_loss = self.num_experts * torch.sum(router_probs * expert_usage_frac)
        
        output = output.view(batch_size, seq_len, d_model)
        return output, load_balance_loss


class ModelConfig:
    """Clean configuration class to replace global config dict"""
    
    def __init__(self,
                 # Model architecture
                 vocab_size: int = 50257,
                 n_embd: int = 512,
                 n_head: int = 8,
                 n_layer: int = 12,
                 ctx_len: int = 1024,
                 
                 # Regularization
                 dropout: float = 0.1,
                 rms_norm_eps: float = 1e-5,
                 
                 # MoE configuration
                 layer_types: Optional[List[str]] = None,
                 n_experts: int = 8,
                 moe_top_k: int = 2,
                 
                 # Orthonormal basis configuration
                 ortho_basis_types: Optional[List[str]] = None,
                 ortho_num_functions: int = 8,
                 ortho_basis_domain_size: int = 256,
                 
                 # Training
                 tie_weights: bool = True,
                 max_seq_len: int = 2048):
        
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.ctx_len = ctx_len
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        
        # Default layer types
        if layer_types is None:
            layer_types = ['mlp'] * (n_layer // 2) + ['moe'] * (n_layer - n_layer // 2)
        self.layer_types = layer_types
        
        self.n_experts = n_experts
        self.moe_top_k = moe_top_k
        
        # Orthonormal basis defaults
        if ortho_basis_types is None:
            ortho_basis_types = ['fourier', 'chebyshev']
        self.ortho_basis_types = ortho_basis_types
        self.ortho_num_functions = ortho_num_functions
        self.ortho_basis_domain_size = ortho_basis_domain_size
        
        self.tie_weights = tie_weights
        self.max_seq_len = max_seq_len


class OptimizedTransformer(nn.Module):
    """Optimized Transformer with clean configuration and better efficiency"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(config.n_layer):
            layer_type = config.layer_types[i] if i < len(config.layer_types) else 'mlp'
            self.blocks.append(self._create_block(layer_type, config))
        
        # Output layers
        self.final_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights if requested
        if config.tie_weights:
            self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Optimized model parameters: {self.total_params:,}")
    
    def _create_block(self, layer_type: str, config: ModelConfig) -> nn.Module:
        """Create a transformer block with specified layer type"""
        class TransformerBlock(nn.Module):
            def __init__(self, layer_type: str, config: ModelConfig):
                super().__init__()
                self.layer_type = layer_type
                
                # Attention
                self.attn = OrthonormalMultiHeadAttention(
                    d_model=config.n_embd,
                    num_heads=config.n_head,
                    basis_types=config.ortho_basis_types,
                    num_functions=config.ortho_num_functions,
                    basis_domain_size=config.ortho_basis_domain_size,
                    dropout=config.dropout,
                    max_seq_len=config.max_seq_len
                )
                
                # Feed-forward
                if layer_type == 'moe':
                    self.ffn = OptimizedDSMoE(
                        d_model=config.n_embd,
                        num_experts=config.n_experts,
                        top_k=config.moe_top_k,
                        dropout=config.dropout
                    )
                else:  # mlp
                    self.ffn = nn.Sequential(
                        nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
                        nn.Dropout(config.dropout)
                    )
                
                # Normalization
                self.norm1 = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
                self.norm2 = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
            
            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                # Self-attention with residual
                attn_out, _ = self.attn(self.norm1(x), is_causal=True)
                x = x + attn_out
                
                # Feed-forward with residual
                if self.layer_type == 'moe':
                    ffn_out, aux_loss = self.ffn(self.norm2(x))
                    x = x + ffn_out
                    return x, aux_loss
                else:
                    ffn_out = self.ffn(self.norm2(x))
                    x = x + ffn_out
                    return x, None
        
        return TransformerBlock(layer_type, config)
    
    def _init_weights(self, module):
        """Improved weight initialization"""
        if isinstance(module, nn.Linear):
            # Use scaled initialization for deep networks
            std = 0.02
            if hasattr(module, '_is_residual'):
                std /= math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                idx: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        batch_size, seq_len = idx.shape
        
        # Token embeddings
        x = self.token_embedding(idx)
        
        # Apply transformer blocks
        aux_losses = []
        for block in self.blocks:
            x, aux_loss = block(x)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        
        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Add auxiliary MoE losses
            if aux_losses:
                aux_loss_total = sum(aux_losses) / len(aux_losses)
                loss = loss + 0.01 * aux_loss_total  # Weight auxiliary loss
        
        # Return auxiliary losses for monitoring
        aux_loss_tensor = torch.stack(aux_losses) if aux_losses else None
        
        return logits, loss, aux_loss_tensor
    
    @torch.no_grad()
    def generate(self, 
                 idx: torch.Tensor, 
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """Optimized generation with top-p sampling"""
        
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx if idx.size(1) <= self.config.ctx_len else idx[:, -self.config.ctx_len:]
            
            # Forward pass
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx


# Example usage and testing
if __name__ == '__main__':
    print("Testing optimized implementations...")
    
    # Test with smaller config for speed
    config = ModelConfig(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=4,
        ctx_len=64,
        layer_types=['mlp', 'moe', 'mlp', 'moe'],
        ortho_basis_types=['fourier', 'chebyshev', 'legendre'],
        ortho_num_functions=4,
        ortho_basis_domain_size=64
    )
    
    try:
        model = OptimizedTransformer(config)
        
        # Test forward pass
        test_input = torch.randint(0, config.vocab_size, (2, 32))
        test_targets = torch.randint(0, config.vocab_size, (2, 32))
        
        logits, loss, aux_losses = model(test_input, test_targets)
        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        if aux_losses is not None:
            print(f"  Aux losses: {aux_losses.mean().item():.4f}")
        
        # Test generation
        generated = model.generate(test_input[:, :5], max_new_tokens=10, temperature=0.8, top_p=0.9)
        print(f"✓ Generation successful")
        print(f"  Generated shape: {generated.shape}")
        
        print("\n✓ All optimizations working correctly!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc() 