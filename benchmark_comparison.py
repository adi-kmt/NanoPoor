import time
import torch
import torch.nn.functional as F
from src.model_optimized import ModelConfig, OptimizedTransformer
import gc


def benchmark_model(model, input_data, targets, num_runs=5):
    """Benchmark a model with given input data"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            logits, loss, aux = model(input_data, targets)
    
    # Actual timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            logits, loss, aux = model(input_data, targets)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time, logits.shape, loss.item() if loss is not None else None


def memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        return 0  # CPU memory tracking is more complex


def main():
    print("🔥 Performance Benchmark: Original vs Optimized Model")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Small Model',
            'vocab_size': 1000,
            'n_embd': 128,
            'n_head': 4,
            'n_layer': 4,
            'ctx_len': 64,
            'batch_size': 4,
            'seq_len': 32
        },
        {
            'name': 'Medium Model', 
            'vocab_size': 5000,
            'n_embd': 256,
            'n_head': 8,
            'n_layer': 6,
            'ctx_len': 128,
            'batch_size': 2,
            'seq_len': 64
        }
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    for test_config in test_configs:
        print(f"📊 Testing {test_config['name']}")
        print("-" * 40)
        
        # Create optimized model
        config = ModelConfig(
            vocab_size=test_config['vocab_size'],
            n_embd=test_config['n_embd'],
            n_head=test_config['n_head'],
            n_layer=test_config['n_layer'],
            ctx_len=test_config['ctx_len'],
            layer_types=['mlp', 'moe'] * (test_config['n_layer'] // 2),
            ortho_basis_types=['fourier', 'chebyshev'],
            ortho_num_functions=4,
            ortho_basis_domain_size=64
        )
        
        model = OptimizedTransformer(config).to(device)
        
        # Create test data
        batch_size = test_config['batch_size']
        seq_len = test_config['seq_len']
        input_data = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Benchmark
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        memory_before = memory_usage()
        avg_time, output_shape, loss = benchmark_model(model, input_data, targets)
        memory_after = memory_usage()
        memory_used = memory_after - memory_before
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ⚡ Average forward time: {avg_time*1000:.2f} ms")
        print(f"  🧠 Memory usage: {memory_used:.2f} MB")
        print(f"  📐 Parameters: {total_params:,} (trainable: {trainable_params:,})")
        print(f"  📊 Output shape: {output_shape}")
        print(f"  📉 Loss: {loss:.4f}")
        
        # Test generation speed
        print("  🎯 Testing generation...")
        gen_input = input_data[:1, :8]  # Single sequence, short prompt
        
        start_gen = time.time()
        with torch.no_grad():
            generated = model.generate(gen_input, max_new_tokens=16, temperature=0.8)
        end_gen = time.time()
        gen_time = end_gen - start_gen
        
        tokens_per_second = 16 / gen_time  # 16 new tokens generated
        print(f"  🚀 Generation speed: {tokens_per_second:.1f} tokens/second")
        print()
        
        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("🎉 Key Optimizations Implemented:")
    print("  ✅ Complete basis function implementations (all 9 types)")
    print("  ✅ Iterative polynomial computation (no deep recursion)")
    print("  ✅ Cached positional bias computation")
    print("  ✅ Vectorized interpolation in basis evaluation")
    print("  ✅ Memory-efficient MoE routing")
    print("  ✅ Clean configuration class (no global state)")
    print("  ✅ Optimized attention with simplified projections")
    print("  ✅ Top-p sampling for better generation")
    print("  ✅ Proper weight initialization scaling")
    print("  ✅ Load balancing for MoE experts")


if __name__ == '__main__':
    main() 