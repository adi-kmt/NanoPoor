# train_optimized.py

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
from model_optimized import ModelConfig, OptimizedTransformer  # Assuming this is your optimized model file
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
    # Determine device first
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        device_str = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # For M1/M2 Macs
        device_str = 'mps'
    else:
        device_str = 'cpu'

    # Only init_process_group if CUDA is explicitly chosen and Muon is used
    # Muon might strictly require distributed setup even for single GPU.
    if device_str == 'cuda' and MUON_AVAILABLE:  # Only attempt if CUDA and Muon is around
        try:
            if not dist.is_initialized():  # Check if not already initialized
                backend = 'nccl' if dist.is_nccl_available() else 'gloo'
                # Ensure MASTER_ADDR and MASTER_PORT are set if not using file:// or tcp://localhost
                # For single-node, single-GPU (or multi-GPU with launch utility), localhost is fine.
                if os.environ.get('MASTER_ADDR') is None:
                    os.environ['MASTER_ADDR'] = 'localhost'
                if os.environ.get('MASTER_PORT') is None:
                    os.environ['MASTER_PORT'] = '12355'  # Or any free port

                # init_method='env://' is often preferred if env vars are set.
                # If world_size/rank are not in env, need to pass them.
                # For this script, assuming single process (world_size=1, rank=0)
                dist.init_process_group(backend=backend, init_method='env://',
                                        world_size=int(os.environ.get('WORLD_SIZE', 1)),
                                        rank=int(os.environ.get('RANK', 0)))
                distributed_initialized = True
                print(f"✓ Distributed initialized with {backend} backend using env://")
            else:
                print("✓ Distributed already initialized.")
                distributed_initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize distributed training: {e}")
            print("Consider setting MASTER_ADDR and MASTER_PORT environment variables if using 'env://'.")

    return distributed_initialized


def get_batch(split: str, data_dir: str, batch_size: int, block_size: int, device: torch.device):
    """Optimized data loading with better error handling"""
    split_filenames = glob.glob(os.path.join("data", data_dir, f"{split}_*.bin"))

    if not split_filenames:
        raise FileNotFoundError(f"No {split} shard files found in data/{data_dir}")

    for attempt in range(3):
        try:
            shard_file = np.random.choice(split_filenames)
            # Use 'offset=0' if the header is handled by np.memmap's own logic for .npy files,
            # or ensure 1024 is correct for your .bin file structure.
            data = np.memmap(shard_file, dtype=np.uint16, mode='r', offset=1024)

            if len(data) <= block_size + 1:  # Need at least block_size for x and block_size for y
                if attempt == 2:
                    raise ValueError(
                        f"Shard {shard_file} too small ({len(data)} tokens) after 3 attempts. Min size needed: {block_size + 2}")
                print(f"Warning: Shard {shard_file} too small ({len(data)} tokens), retrying...")
                continue

            max_start_idx = len(data) - block_size - 1
            # Ensure max_start_idx is not negative if shard is barely large enough
            if max_start_idx < 0:
                if attempt == 2:
                    raise ValueError(
                        f"Shard {shard_file} too small for sequence length after offset. Max start index < 0.")
                print(f"Warning: Shard {shard_file} has max_start_idx {max_start_idx}, retrying...")
                continue

            start_indices = torch.randint(0, max_start_idx + 1, (batch_size,))  # Max is inclusive for randint

            sequences = [torch.from_numpy(data[start_idx: start_idx + block_size].astype(np.int64)) for start_idx in
                         start_indices]
            targets = [torch.from_numpy(data[start_idx + 1: start_idx + block_size + 1].astype(np.int64)) for start_idx
                       in start_indices]

            x = torch.stack(sequences)
            y = torch.stack(targets)

            if device.type == 'cuda':
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)

            return x, y

        except Exception as e:
            if attempt == 2:
                print(f"Error loading batch from {shard_file if 'shard_file' in locals() else 'unknown shard'}")
                raise e
            print(f"Attempt {attempt + 1} failed: {e}, retrying...")

    raise RuntimeError("Failed to load batch after 3 attempts")


@torch.no_grad()
def estimate_loss(model, config: Dict[str, Any], data_dir: str, eval_iters: int, device: torch.device, ctx) -> Dict[
    str, float]:
    """Optimized loss estimation"""
    model.eval()
    losses = {}

    for split in ['train', 'val']:
        split_losses_list = []
        for k in range(eval_iters):
            try:
                X, Y = get_batch(split, data_dir, config['batch_size'], config['ctx_len'], device)
                with ctx:
                    logits, loss, _ = model(X, Y)  # Assuming model returns (logits, loss, aux_loss)
                split_losses_list.append(loss.item())
            except Exception as e:
                print(f"Warning: Error in eval batch {k} for split {split}: {e}")
                # Optionally append a high value or skip, or handle as needed
                # For now, let's skip problematic batches in eval

        if split_losses_list:
            losses[split] = sum(split_losses_list) / len(split_losses_list)
        else:
            losses[split] = float('inf')  # Or NaN, if no eval batches succeeded

    model.train()
    return losses


def configure_optimizers(model, weight_decay: float, learning_rate: float, device: torch.device) -> List[
    torch.optim.Optimizer]:
    """Configure optimizers with defensive checks"""
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Weight decay typically applied to weights of Linear/Conv layers, not biases or norm layers
            if param.ndim >= 2:  # Heuristic: weights are 2D or more
                decay_params.append(param)
            else:
                nodecay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    use_fused_adam = False
    if device.type == 'cuda':
        try:
            dummy_param = torch.nn.Parameter(torch.randn(1, device=device))
            torch.optim.AdamW([dummy_param], lr=learning_rate, fused=True)  # Test if fused is supported
            use_fused_adam = True
            del dummy_param
        except RuntimeError:  # PyTorch < 1.9 or CUDA not fully ready for fused
            use_fused_adam = False
            print("Note: Fused AdamW not available or compatible, using standard AdamW.")

    adamw_kwargs = {
        'lr': learning_rate,
        'betas': (0.9, 0.95),
        'eps': 1e-8,
        # weight_decay is applied per group
    }
    if use_fused_adam:
        adamw_kwargs['fused'] = True

    adamw_optimizer = torch.optim.AdamW(optim_groups, **adamw_kwargs)

    all_optimizers = [adamw_optimizer]  # Start with AdamW

    if MUON_AVAILABLE and dist.is_initialized():
        try:
            # Muon might take all parameters or specific groups. Assuming all for now.
            muon_optimizer = Muon(model.parameters(), lr=learning_rate)
            all_optimizers.insert(0, muon_optimizer)  # Muon first, if used
            print("✓ Using Muon + AdamW optimizers")
        except Exception as e:
            print(f"Warning: Could not initialize Muon: {e}. Using AdamW only.")
    else:
        if MUON_AVAILABLE and not dist.is_initialized() and device.type == 'cuda':
            print("Note: Muon available but distributed not initialized. Using AdamW only.")

    if len(all_optimizers) == 1:
        print(f"✓ Using AdamW optimizer only (Fused: {use_fused_adam if device.type == 'cuda' else 'N/A'})")

    return all_optimizers


def train_single_config(
        config: Dict[str, Any],
        lr: float,
        min_lr: float,
        run_name: str,
        csv_writer,
        csv_file
) -> float:
    print(f"\n🚀 Training with lr={lr:.2e}, min_lr={min_lr:.2e} for run: {run_name}")
    print("-" * 50)

    device_str = config['device']
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA specified but not available. Falling back to CPU.")
        device_str = 'cpu'
    elif device_str == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("Warning: MPS specified but not available. Falling back to CPU.")
        device_str = 'cpu'
    device = torch.device(device_str)

    distributed_initialized = setup_distributed()  # Call after device is determined

    use_amp = (device.type == 'cuda')  # AMP typically for CUDA
    # Define dtype for autocast if CUDA
    autocast_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    ctx = torch.amp.autocast(device_type=device.type, dtype=autocast_dtype) if use_amp else nullcontext()
    scaler = amp.GradScaler(enabled=use_amp)

    meta_path = f"data/{config['data_dir']}/meta.pkl"
    vocab_size = config.get('vocab_size', 50257)  # Default if not in config
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        print(f"✓ Loaded vocab_size = {vocab_size} from meta.pkl")
    else:
        print(f"Warning: meta.pkl not found at {meta_path}. Using vocab_size = {vocab_size}")

    model_config_obj = ModelConfig(
        vocab_size=vocab_size,
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        ctx_len=config['ctx_len'],
        dropout=config['dropout'],
        layer_types=config['types'],
        n_experts=config['n_experts'],
        moe_top_k=config.get('moe_top_k', 2),
        ortho_basis_types=config.get('ortho_basis_types', ['fourier', 'chebyshev']),
        ortho_num_functions=config.get('ortho_num_functions', 8),
        ortho_basis_domain_size=config.get('ortho_basis_domain_size', 256),
        tie_weights=config.get('tie_weights', True),
        max_seq_len=config['ctx_len']  # Or a larger pre-defined max_seq_len for caching
    )

    model = OptimizedTransformer(model_config_obj).to(device)

    optimizers = configure_optimizers(model, config['weight_decay'], lr, device)
    adamw_optimizer = optimizers[-1]

    warmup_scheduler = LinearLR(adamw_optimizer, start_factor=1e-3, total_iters=config['warmup_iters'])
    cosine_scheduler_iters = config['max_iters'] - config['warmup_iters']
    if cosine_scheduler_iters <= 0:  # Handle cases where max_iters is very small
        print(
            f"Warning: max_iters ({config['max_iters']}) <= warmup_iters ({config['warmup_iters']}). Cosine scheduler will not run.")
        # Only use warmup or make T_max=1 for CosineAnnealingLR if you want it to do something minimal
        scheduler = warmup_scheduler
    else:
        cosine_scheduler = CosineAnnealingLR(adamw_optimizer, T_max=cosine_scheduler_iters, eta_min=min_lr)
        scheduler = SequentialLR(adamw_optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                 milestones=[config['warmup_iters']])

    if device.type == 'cuda':  # Try to compile if on CUDA
        # torch._dynamo.reset() # Optional: useful if recompiling often with changes
        print("Attempting to compile the model...")
        try:
            # dynamic=False can sometimes help with complex models or specific ops
            model = torch.compile(model, fullgraph=False, dynamic=False)
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}. Proceeding without compilation.")

    train_losses_log = []  # For simple list logging if needed
    val_losses_log = []
    best_val_loss = float('inf')

    print(f"Starting training for {config['max_iters']} iterations...")
    start_time = time.time()

    try:
        for iter_num in range(config['max_iters'] + 1):
            if iter_num > 0 and (iter_num % config['eval_interval'] == 0 or iter_num == config['max_iters']):
                losses = estimate_loss(model, config, config['data_dir'], config['eval_iters'], device, ctx)
                train_loss_eval = losses.get('train', float('nan'))
                val_loss_eval = losses.get('val', float('nan'))

                train_losses_log.append(train_loss_eval)
                val_losses_log.append(val_loss_eval)

                elapsed = time.time() - start_time
                current_adamw_lr = adamw_optimizer.param_groups[0]['lr']
                print(
                    f"Step {iter_num:4d} | LR: {current_adamw_lr:.2e} | Train (eval): {train_loss_eval:.4f} | Val: {val_loss_eval:.4f} | Time: {elapsed / 60:.1f}m")

                csv_writer.writerow([
                    run_name, lr, min_lr, iter_num, None,  # No step train loss here
                    train_loss_eval, val_loss_eval, elapsed
                ])
                csv_file.flush()

                if not math.isnan(val_loss_eval):
                    best_val_loss = min(best_val_loss, val_loss_eval)

                if val_loss_eval > 10.0 and iter_num > config['warmup_iters'] * 2:  # Check for explosion after warmup
                    print(f"⚠️ Loss exploded ({val_loss_eval:.4f}), stopping early.")
                    break

            if iter_num == config['max_iters']:
                break

            # --- Training step ---
            model.train()  # Ensure model is in train mode

            # Zero gradients for all optimizers BEFORE the accumulation loop
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            accumulated_loss_for_log = 0.0
            successful_microsteps = 0

            for micro_step in range(config['grad_accum']):
                try:
                    X, Y = get_batch('train', config['data_dir'], config['batch_size'], config['ctx_len'], device)

                    with ctx:  # Autocast for mixed precision
                        _logits, loss_raw, _aux_loss = model(X, Y)
                        # Ensure loss_raw is scalar (it should be from F.cross_entropy default)
                        loss = loss_raw / config['grad_accum']  # Scale loss for accumulation

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(
                            f"Warning: NaN/Inf loss in micro_step {micro_step} of iter {iter_num}. Skipping backward for this micro_step.")
                        continue

                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    accumulated_loss_for_log += loss_raw.item()  # Log sum of raw losses
                    successful_microsteps += 1

                except Exception as e:
                    print(
                        f"Warning: Error in micro_step {micro_step} of iter {iter_num}: {e}. Skipping this micro_batch.")
                    # Do not zero gradients here; just skip this micro_batch's contribution
                    # The main zero_grad is at the start of the iteration.
                    continue

                    # --- Optimizer Step ---
            if successful_microsteps > 0:
                avg_step_train_loss = accumulated_loss_for_log / successful_microsteps

                if use_amp:
                    for opt in optimizers:  # Unscale ALL optimizers
                        scaler.unscale_(opt)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

                    for opt in optimizers:  # Step ALL optimizers
                        scaler.step(opt)

                    scaler.update()  # Single update after all steps
                else:  # Not using AMP
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    for opt in optimizers:
                        opt.step()

                if scheduler:  # Step the scheduler (tied to adamw_optimizer)
                    scheduler.step()

                    # Log step training loss
                if iter_num % 10 == 0 and iter_num > 0:  # Log every 10 steps
                    elapsed_log = time.time() - start_time
                    csv_writer.writerow([
                        run_name, lr, min_lr, iter_num, avg_step_train_loss,
                        None, None, elapsed_log  # No eval losses for this row
                    ])
                    csv_file.flush()
            else:  # No successful microsteps
                if iter_num % 10 == 0 and iter_num > 0:
                    print(f"Iter {iter_num}: No successful microsteps, skipping optimizer step and logging.")
                # Gradients would have been zeroed at the start of next iter anyway.

    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user for run {run_name}.")
    except Exception as e:
        print(f"\nTRAINING FAILED for run {run_name}: {e}")
        import traceback
        traceback.print_exc()
        best_val_loss = float('inf')  # Indicate failure

    finally:
        if distributed_initialized and dist.is_initialized():
            dist.destroy_process_group()
            print(f"✓ Distributed process group destroyed for run {run_name}.")

    print(f"✓ Training completed for {run_name}. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def grid_search_lr(config: Dict[str, Any]) -> Tuple[float, float, float]:
    lr_values = config.get('lr_values', [6e-4, 8e-4, 1e-3, 1.2e-3, 1.5e-3])
    min_lr_ratios = config.get('min_lr_ratios', [0.1, 0.3, 0.5])

    print(f"🔍 Starting Grid Search")
    print(f"LR values: {lr_values}")
    print(f"Min LR ratios (of LR): {min_lr_ratios}")
    total_combinations = len(lr_values) * len(min_lr_ratios)
    print(f"Total combinations: {total_combinations}")

    os.makedirs("grid_search_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"grid_search_results/lr_search_{timestamp}.csv"

    best_lr_overall = None
    best_min_lr_overall = None
    best_val_loss_overall = float('inf')

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['run_name', 'lr_config', 'min_lr_config', 'step',
                             'step_train_loss', 'eval_train_loss', 'eval_val_loss', 'elapsed_time_s'])

        for i, (lr_val, min_lr_ratio_val) in enumerate(product(lr_values, min_lr_ratios)):
            current_min_lr = lr_val * min_lr_ratio_val
            # Sanitize run_name for file systems
            run_name_val = f"gs_lr{lr_val:.1e}_minlr{current_min_lr:.1e}".replace('.', 'p')

            print(f"\n{'=' * 60}")
            print(f"Grid Search Run {i + 1}/{total_combinations}: {run_name_val}")
            print(f"{'=' * 60}")

            grid_run_config = config.copy()
            grid_run_config['max_iters'] = min(config.get('grid_max_iters', 500), config['max_iters'])
            grid_run_config['eval_interval'] = min(config.get('grid_eval_interval', 50), config['eval_interval'])

            current_run_best_val_loss = train_single_config(
                grid_run_config, lr_val, current_min_lr, run_name_val, csv_writer, csvfile
            )

            if not math.isnan(current_run_best_val_loss) and current_run_best_val_loss < best_val_loss_overall:
                best_val_loss_overall = current_run_best_val_loss
                best_lr_overall = lr_val
                best_min_lr_overall = current_min_lr
                print(
                    f"🌟 New overall best: lr={best_lr_overall:.2e}, min_lr={best_min_lr_overall:.2e}, val_loss={best_val_loss_overall:.4f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n🎉 Grid Search Complete!")
    if best_lr_overall is not None:
        print(f"Best LR: {best_lr_overall:.2e}")
        print(f"Best Min LR: {best_min_lr_overall:.2e}")
        print(f"Best Val Loss: {best_val_loss_overall:.4f}")
    else:
        print("Grid search did not find any valid results.")
    print(f"Results logged to: {csv_path}")

    return best_lr_overall, best_min_lr_overall, best_val_loss_overall


def main():
    parser = argparse.ArgumentParser(description="Optimized Training with Grid Search")

    # Model architecture
    parser.add_argument('--ctx_len', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=8)  # Reduced for faster testing
    parser.add_argument('--n_experts', type=int, default=8)  # For MoE layers
    parser.add_argument('--moe_top_k', type=int, default=2)
    parser.add_argument('--types', nargs='*', type=str, default=['mlp'] * 4 + ['moe'] * 4,
                        help="Layer types per block e.g., mlp moe")

    # Orthonormal Basis Attention params (add more if needed by ModelConfig)
    parser.add_argument('--ortho_basis_types', nargs='*', default=['fourier', 'chebyshev'])
    parser.add_argument('--ortho_num_functions', type=int, default=8)

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-3)  # Default LR for single run
    parser.add_argument('--min_lr', type=float, default=1e-4)  # Default Min LR for single run
    parser.add_argument('--dropout', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # Training schedule
    parser.add_argument('--max_iters', type=int, default=1500)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=100)  # Eval more often
    parser.add_argument('--eval_iters', type=int, default=10)  # Fewer eval iters for speed

    # Batch configuration
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=8)  # Effective batch size = 64

    # Infrastructure
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--data_dir', type=str, default='tokenized_data')  # Example data dir

    # Grid search options
    parser.add_argument('--grid_search', action='store_true', help='Perform grid search for LR and Min LR ratio')
    parser.add_argument('--grid_lr_values', nargs='*', type=float, help='LR values for grid search')
    parser.add_argument('--grid_min_lr_ratios', nargs='*', type=float, help='Min LR ratios for grid search')
    parser.add_argument('--grid_max_iters', type=int, default=500, help='Max iters per grid search run')
    parser.add_argument('--grid_eval_interval', type=int, default=50, help='Eval interval per grid search run')

    args = parser.parse_args()

    config = vars(args)  # Convert argparse Namespace to dict

    print("🔥 Optimized Training Script")
    print("=" * 50)
    print(
        f"Model: {args.n_layer}L-{args.n_embd}D-{args.n_head}H | Experts: {args.n_experts} | Types: {' '.join(args.types)}")
    print(f"Context: {args.ctx_len} | Dropout: {args.dropout}")
    print(f"Device: {args.device} | Data: {args.data_dir}")
    print(
        f"Batch: {args.batch_size} (micro) x {args.grad_accum} (accum) = {args.batch_size * args.grad_accum} (effective)")

    if args.grid_search:
        if args.grid_lr_values: config['lr_values'] = args.grid_lr_values
        if args.grid_min_lr_ratios: config['min_lr_ratios'] = args.grid_min_lr_ratios

        best_lr, best_min_lr, _ = grid_search_lr(config)

        if best_lr is not None and best_min_lr is not None:
            print(
                f"\n🎯 Proceeding to final, longer training run with best found LR={best_lr:.2e}, Min LR={best_min_lr:.2e}")
            # Use original max_iters for the final run
            final_config = config.copy()
            final_config['max_iters'] = args.max_iters  # Restore original max_iters
            final_config['eval_interval'] = args.eval_interval  # Restore original eval_interval

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            final_run_name = f"final_lr{best_lr:.1e}_minlr{best_min_lr:.1e}".replace('.', 'p')
            final_csv_path = f"grid_search_results/final_training_{final_run_name}_{timestamp}.csv"

            with open(final_csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['run_name', 'lr_config', 'min_lr_config', 'step',
                                     'step_train_loss', 'eval_train_loss', 'eval_val_loss', 'elapsed_time_s'])
                train_single_config(
                    final_config, best_lr, best_min_lr,
                    final_run_name,
                    csv_writer, csvfile
                )
            print(f"Final training results logged to: {final_csv_path}")
        else:
            print("Grid search did not yield valid parameters for a final run.")

    else:  # Single training run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"single_lr{args.lr:.1e}_minlr{args.min_lr:.1e}".replace('.', 'p')
        csv_path = f"training_log_{run_name}_{timestamp}.csv"

        print(f"\n🚀 Starting single training run: {run_name} with LR={args.lr:.2e}, Min LR={args.min_lr:.2e}")

        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Matching header with grid search for consistency
            csv_writer.writerow(['run_name', 'lr_config', 'min_lr_config', 'step',
                                 'step_train_loss', 'eval_train_loss', 'eval_val_loss', 'elapsed_time_s'])
            train_single_config(
                config, args.lr, args.min_lr, run_name, csv_writer, csvfile
            )
        print(f"\n🏁 Training Complete! Results logged to: {csv_path}")


if __name__ == "__main__":
    # For torch.compile and multiprocessing/distributed to work well on some systems:
    torch.multiprocessing.set_start_method('spawn', force=True) if \
        torch.multiprocessing.get_start_method(allow_none=True) != 'spawn' else None
    main()