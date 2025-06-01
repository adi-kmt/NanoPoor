import os
import math
import time
import glob
import torch
import string
import random
import pickle
import argparse
import csv
import numpy as np
from contextlib import nullcontext
from itertools import product
from typing import List, Tuple, Dict, Any

import torch.amp as amp
import torch._dynamo
import torch.distributed as dist

# Use our optimized model
from model_optimized import ModelConfig, OptimizedTransformer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# Keep the Muon import from original
try:
    from muon import Muon
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("Warning: Muon optimizer not available")


def setup_distributed():
    """Setup distributed training if needed"""
    distributed_initialized = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if 'cuda' in str(device) and MUON_AVAILABLE:
        try:
            backend = 'nccl' if dist.is_nccl_available() else 'gloo'
            init_url = "tcp://localhost:12355"
            dist.init_process_group(backend=backend, init_method=init_url, world_size=1, rank=0)
            distributed_initialized = True
            print(f"✓ Distributed initialized with {backend} backend")
        except Exception as e:
            print(f"Warning: Could not initialize distributed training: {e}")
    
    return distributed_initialized


def get_batch(split: str, data_dir: str, batch_size: int, block_size: int, device: torch.device):
    """Optimized data loading with better error handling"""
    split_filenames = glob.glob(os.path.join("data", data_dir, f"{split}_*.bin"))
    
    if not split_filenames:
        raise FileNotFoundError(f"No {split} shard files found in data/{data_dir}")

    # Try up to 3 times to get a valid batch
    for attempt in range(3):
        try:
            shard_file = np.random.choice(split_filenames)
            data = np.memmap(shard_file, dtype=np.uint16, mode='r', offset=1024)
            
            if len(data) <= block_size + 1:
                if attempt == 2:  # Last attempt
                    raise ValueError(f"All shards too small. Minimum size needed: {block_size + 2}")
                continue
            
            # Vectorized batch creation
            max_start_idx = len(data) - block_size - 1
            start_indices = torch.randint(0, max_start_idx, (batch_size,))
            
            # Create sequences efficiently
            sequences = []
            targets = []
            for start_idx in start_indices:
                seq = torch.from_numpy(data[start_idx:start_idx + block_size].astype(np.int64))
                target = torch.from_numpy(data[start_idx + 1:start_idx + block_size + 1].astype(np.int64))
                sequences.append(seq)
                targets.append(target)
            
            x = torch.stack(sequences)
            y = torch.stack(targets)
            
            # Efficient device transfer
            if device.type == 'cuda':
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            
            return x, y
            
        except Exception as e:
            if attempt == 2:
                raise e
            print(f"Attempt {attempt + 1} failed: {e}, retrying...")
    
    raise RuntimeError("Failed to load batch after 3 attempts")


@torch.no_grad()
def estimate_loss(model, config, data_dir: str, eval_iters: int, device: torch.device, ctx) -> Dict[str, float]:
    """Optimized loss estimation"""
    model.eval()
    losses = {}
    
    for split in ['train', 'val']:
        split_losses = torch.zeros(eval_iters, device=device)
        
        for k in range(eval_iters):
            try:
                X, Y = get_batch(split, data_dir, config['batch_size'], config['ctx_len'], device)
                with ctx:
                    logits, loss, _ = model(X, Y)
                split_losses[k] = loss
            except Exception as e:
                print(f"Warning: Error in eval batch {k}: {e}")
                split_losses[k] = float('inf')
        
        losses[split] = split_losses.mean().item()
    
    model.train()
    return losses


def configure_optimizers(model, weight_decay: float, learning_rate: float, device: torch.device):
    """Configure optimizers with defensive checks"""
    optimizers = []
    
    # Separate parameters for weight decay
    decay_params = []
    nodecay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)
    
    # AdamW optimizer
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    use_fused = device.type == 'cuda'
    if use_fused:
        try:
            # Test fused AdamW
            dummy_param = torch.nn.Parameter(torch.randn(1, device=device))
            test_opt = torch.optim.AdamW([dummy_param], lr=learning_rate, fused=True)
            del test_opt, dummy_param
        except:
            use_fused = False
    
    adamw_kwargs = {
        'lr': learning_rate,
        'betas': (0.9, 0.95),
        'eps': 1e-8,
        'weight_decay': weight_decay
    }
    if use_fused:
        adamw_kwargs['fused'] = True
    
    adamw_optimizer = torch.optim.AdamW(optim_groups, **adamw_kwargs)
    optimizers.append(adamw_optimizer)
    
    # Add Muon if available and distributed is initialized
    if MUON_AVAILABLE and dist.is_initialized():
        try:
            muon_optimizer = Muon(model.parameters(), lr=learning_rate)
            optimizers.insert(0, muon_optimizer)  # Muon first
            print("✓ Using Muon + AdamW optimizers")
        except Exception as e:
            print(f"Warning: Could not initialize Muon: {e}")
    
    if len(optimizers) == 1:
        print("✓ Using AdamW optimizer only")
    
    return optimizers


def train_single_config(
    config: Dict[str, Any],
    lr: float,
    min_lr: float,
    run_name: str,
    csv_writer,
    csv_file
) -> float:
    """Train a single configuration and return validation loss"""
    
    print(f"\n🚀 Training with lr={lr:.2e}, min_lr={min_lr:.2e}")
    print("-" * 50)
    
    # Setup device and distributed
    device = torch.device(config['device'])
    distributed_initialized = setup_distributed()
    
    # Mixed precision setup
    use_amp = device.type == 'cuda'
    ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext()
    scaler = amp.GradScaler(enabled=use_amp)
    
    # Load vocabulary
    meta_path = f"data/{config['data_dir']}/meta.pkl"
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        print(f"✓ Loaded vocab_size = {vocab_size}")
    else:
        vocab_size = 50257
        print(f"Warning: Using default vocab_size = {vocab_size}")
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=vocab_size,
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        ctx_len=config['ctx_len'],
        dropout=config['dropout'],
        layer_types=config['types'],
        n_experts=config['n_experts'],
        moe_top_k=2,
        ortho_basis_types=['fourier', 'chebyshev', 'hermite'],
        ortho_num_functions=8,
        ortho_basis_domain_size=256,
        tie_weights=True,
        max_seq_len=config['ctx_len']
    )
    
    # Initialize model
    model = OptimizedTransformer(model_config).to(device)
    
    # Configure optimizers
    optimizers = configure_optimizers(model, config['weight_decay'], lr, device)
    adamw_optimizer = optimizers[-1]  # AdamW is always last
    
    # Setup scheduler
    warmup_scheduler = LinearLR(adamw_optimizer, start_factor=1e-3, total_iters=config['warmup_iters'])
    cosine_scheduler = CosineAnnealingLR(
        adamw_optimizer, 
        T_max=config['max_iters'] - config['warmup_iters'], 
        eta_min=min_lr
    )
    scheduler = SequentialLR(
        adamw_optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[config['warmup_iters']]
    )
    
    # Compile model if CUDA
    if device.type == 'cuda':
        try:
            model = torch.compile(model, fullgraph=False, dynamic=False)
            print("✓ Model compiled")
        except Exception as e:
            print(f"Warning: Compilation failed: {e}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training for {config['max_iters']} iterations...")
    start_time = time.time()
    
    try:
        for iter_num in range(config['max_iters'] + 1):
            
            # Evaluation
            if iter_num % config['eval_interval'] == 0 or iter_num == config['max_iters']:
                if iter_num > 0:  # Skip first eval
                    losses = estimate_loss(
                        model, config, config['data_dir'], 
                        config['eval_iters'], device, ctx
                    )
                    
                    train_loss = losses['train']
                    val_loss = losses['val']
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    
                    elapsed = time.time() - start_time
                    print(f"Step {iter_num:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed/60:.1f}m")
                    
                    # Log to CSV every 10 steps or at eval intervals
                    if iter_num % 10 == 0 or iter_num % config['eval_interval'] == 0:
                        csv_writer.writerow([
                            run_name, lr, min_lr, iter_num, train_loss, val_loss, elapsed
                        ])
                        csv_file.flush()
                    
                    best_val_loss = min(best_val_loss, val_loss)
                    
                    # Early stopping if loss explodes
                    if val_loss > 10.0:
                        print("⚠️  Loss exploded, stopping early")
                        break
            
            if iter_num == config['max_iters']:
                break
            
            # Training step
            loss_accum = 0.0
            model.train()
            
            for micro_step in range(config['grad_accum']):
                try:
                    X, Y = get_batch('train', config['data_dir'], config['batch_size'], config['ctx_len'], device)
                    
                    with ctx:
                        logits, loss, aux_losses = model(X, Y)
                        loss = loss / config['grad_accum']
                    
                    scaler.scale(loss).backward()
                    loss_accum += loss.item()
                    
                except Exception as e:
                    print(f"Warning: Error in training step {iter_num}.{micro_step}: {e}")
                    continue
            
            # Optimizer steps
            for opt in optimizers:
                scaler.unscale_(opt)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            for opt in optimizers:
                scaler.step(opt)
                opt.zero_grad(set_to_none=True)
            
            scaler.update()
            scheduler.step()
            
            # Log training loss every 10 steps
            if iter_num % 10 == 0 and iter_num > 0:
                csv_writer.writerow([
                    run_name, lr, min_lr, iter_num, loss_accum, None, time.time() - start_time
                ])
                csv_file.flush()
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        best_val_loss = float('inf')
    
    finally:
        # Cleanup
        if distributed_initialized and dist.is_initialized():
            dist.destroy_process_group()
    
    print(f"✓ Training completed. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def grid_search_lr(config: Dict[str, Any]) -> Tuple[float, float, float]:
    """Perform grid search for learning rate and min_lr"""
    
    # Grid search parameters
    lr_values = [6e-4, 8e-4, 1e-3, 1.2e-3, 1.5e-3]
    min_lr_ratios = [0.1, 0.3, 0.5, 0.7]  # min_lr as ratio of lr
    
    print(f"🔍 Starting Grid Search")
    print(f"LR values: {lr_values}")
    print(f"Min LR ratios: {min_lr_ratios}")
    print(f"Total combinations: {len(lr_values) * len(min_lr_ratios)}")
    
    # Create results directory and CSV
    os.makedirs("grid_search_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"grid_search_results/lr_search_{timestamp}.csv"
    
    best_lr = None
    best_min_lr = None
    best_val_loss = float('inf')
    
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['run_name', 'lr', 'min_lr', 'step', 'train_loss', 'val_loss', 'elapsed_time'])
        
        for i, (lr, min_lr_ratio) in enumerate(product(lr_values, min_lr_ratios)):
            min_lr = lr * min_lr_ratio
            run_name = f"lr{lr:.0e}_minlr{min_lr:.0e}"
            
            print(f"\n{'='*60}")
            print(f"Grid Search {i+1}/{len(lr_values) * len(min_lr_ratios)}")
            print(f"{'='*60}")
            
            # Reduce max_iters for grid search to save time
            grid_config = config.copy()
            grid_config['max_iters'] = min(500, config['max_iters'])  # Shorter runs for grid search
            grid_config['eval_interval'] = 50  # More frequent evaluation
            
            val_loss = train_single_config(
                grid_config, lr, min_lr, run_name, csv_writer, csvfile
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr
                best_min_lr = min_lr
                print(f"🌟 New best: lr={best_lr:.2e}, min_lr={best_min_lr:.2e}, val_loss={best_val_loss:.4f}")
            
            # Clean up GPU memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n🎉 Grid Search Complete!")
    print(f"Best LR: {best_lr:.2e}")
    print(f"Best Min LR: {best_min_lr:.2e}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Results saved to: {csv_path}")
    
    return best_lr, best_min_lr, best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Optimized Training with Grid Search")
    
    # Model architecture
    parser.add_argument('--ctx_len', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_experts', type=int, default=32)
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--min_lr', type=float, default=3e-4)
    parser.add_argument('--dropout', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Training schedule
    parser.add_argument('--max_iters', type=int, default=1500)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=150)
    parser.add_argument('--eval_iters', type=int, default=20)
    
    # Batch configuration
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=12)
    
    # Infrastructure
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='tokenized_data')
    
    # Layer types
    parser.add_argument('--types', nargs='*', type=str, default=['mlp'] * 8)
    
    # Grid search option
    parser.add_argument('--grid_search', action='store_true', help='Perform grid search for LR')
    parser.add_argument('--use_best_lr', action='store_true', help='Use best LR from previous grid search')
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = {
        'ctx_len': args.ctx_len,
        'n_embd': args.n_embd,
        'n_head': args.n_head,
        'n_layer': args.n_layer,
        'n_experts': args.n_experts,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'max_grad_norm': args.max_grad_norm,
        'max_iters': args.max_iters,
        'warmup_iters': args.warmup_iters,
        'eval_interval': args.eval_interval,
        'eval_iters': args.eval_iters,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'device': args.device,
        'data_dir': args.data_dir,
        'types': args.types
    }
    
    print("🔥 Optimized Training Script")
    print("=" * 50)
    print(f"Model: {args.n_layer}L-{args.n_embd}D-{args.n_head}H")
    print(f"Context: {args.ctx_len}")
    print(f"Device: {args.device}")
    print(f"Data: {args.data_dir}")
    
    if args.grid_search:
        # Perform grid search
        best_lr, best_min_lr, best_val_loss = grid_search_lr(config)
        
        # Run final training with best parameters
        print(f"\n🎯 Final training with best parameters...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_csv_path = f"grid_search_results/final_training_{timestamp}.csv"
        
        with open(final_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['run_name', 'lr', 'min_lr', 'step', 'train_loss', 'val_loss', 'elapsed_time'])
            
            final_val_loss = train_single_config(
                config, best_lr, best_min_lr, 
                f"final_lr{best_lr:.0e}_minlr{best_min_lr:.0e}",
                csv_writer, csvfile
            )
        
        print(f"\n🏁 Final Results:")
        print(f"Best LR: {best_lr:.2e}")
        print(f"Best Min LR: {best_min_lr:.2e}")
        print(f"Final Val Loss: {final_val_loss:.4f}")
        
    else:
        # Single training run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"single_training_{timestamp}.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['run_name', 'lr', 'min_lr', 'step', 'train_loss', 'val_loss', 'elapsed_time'])
            
            run_name = f"single_lr{args.lr:.0e}_minlr{args.min_lr:.0e}"
            final_val_loss = train_single_config(
                config, args.lr, args.min_lr, run_name, csv_writer, csvfile
            )
        
        print(f"\n🏁 Training Complete!")
        print(f"Final Val Loss: {final_val_loss:.4f}")
        print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main() 