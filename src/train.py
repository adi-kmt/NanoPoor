# Training Loop

import os
import math
import time
import glob
import torch
import string
import random
import pickle
import argparse
#import heavyball # No longer used
import numpy as np
from contextlib import nullcontext

import torch.amp as amp  # For GradScaler
import torch._dynamo
import torch.distributed as dist # <-- Added for distributed init

import model
from model import Transformer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import plot
from plot import plot_loss

from muon import Muon

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--ctx_len', type=int, default=1024)
parser.add_argument('--eval_interval', type=int, default=20)
parser.add_argument('--grad_accum', type=int, default=4)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--min_lr', type=float, default=1e-4) # <-- Changed type to float
parser.add_argument('--dropout', type=float, default=0.02)

parser.add_argument('--max_iters', type=int, default=200)
parser.add_argument('--eval_iters', type=int, default=20)
parser.add_argument('--warmup_iters', type=int, default=10)

parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--res_path', type=str, default="")

parser.add_argument('--data_dir', type=str, default="shakespeare")

parser.add_argument('--n_embd', type=int, default=16)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--n_experts', type=int, default=32)
parser.add_argument('--use_expert_bias', type=bool, default=True)

parser.add_argument('--types', nargs='*', type=str, default= ['mlp','moe','mlp','moe'])
parser.add_argument('--device', type=str, default="cpu")

args = parser.parse_args()

# Update the config with parsed arguments
model.config['ctx_len'] = args.ctx_len
model.config['device'] = args.device
model.config['n_embd'] = args.n_embd
model.config['n_head'] = args.n_head
model.config['n_layer'] = args.n_layer
model.config['n_experts'] = args.n_experts
model.config['type'] = args.types
model.config['use_expert_bias'] = args.use_expert_bias
model.config['dropout'] = args.dropout

# hyperparams
batch_size = args.batch_size
block_size = args.ctx_len # ctx_len
model.config['ctx_len'] = args.ctx_len
eval_interval = args.eval_interval
grad_accum_steps = args.grad_accum  # Num microbatches

lr = args.lr # 3x lr, was in early ngpt spdrns
min_lr = args.min_lr # Now float

max_iters = args.max_iters
eval_iters = args.eval_iters
warmup_iters = args.warmup_iters

beta1 = 0.9 # AdamW beta1
beta2 = 0.95 # AdamW beta2
weight_decay = 1e-1
max_grad_norm = 1.0  # Grad clipping

train_losses_history = []
val_losses_history = []

# continue or scratch
resume = args.resume
data_dir = args.data_dir
resume_checkpoint = args.res_path

device = args.device
model.config['device'] = args.device
model.config.update(vars(args))

# --- Distributed Initialization for Muon (TCP method) ---
distributed_initialized = False
print(f"\n--- Attempting Distributed Initialization (TCP Method, Device: {device}) ---")
# Only attempt initialization if CUDA is used and Muon is present in the expected optimizers
# (Muon might strictly require CUDA/distributed backend)
muon_in_use = True # Assume Muon is intended unless configure_optimizers excludes it
if 'cuda' in device:
    try:
        # Choose backend
        backend = 'gloo'
        if dist.is_nccl_available():
            backend = 'nccl'
        else:
            print("WARNING: NCCL backend not available, using 'gloo'. Muon performance might be affected.")

        # Use TCP initialization (simpler for single node)
        init_url = f"tcp://localhost:12355" # Use a free port
        dist.init_process_group(backend=backend, init_method=init_url, world_size=1, rank=0)
        print(f"Successfully called init_process_group (TCP) with backend: {backend}.")
        distributed_initialized = True
        print(f"Is distributed initialized after setup? {dist.is_initialized()}")

    except Exception as e:
        print(f"ERROR: Failed to initialize process group (TCP): {e}")
        print("Muon optimizer might fail if distributed initialization is required.")
        # Consider exiting if Muon is critical and initialization fails:
        # exit(1)
else:
     print("INFO: Not initializing distributed process group (device is not CUDA).")
print("--- Finished Distributed Initialization Attempt ---")

# Mixed precision, dtype, etc.
ctx = nullcontext() if args.device == 'cpu' else torch.amp.autocast(device_type=args.device, dtype=torch.float16) # Keep float16 for GradScaler
scaler = amp.GradScaler(enabled=("cuda" in device))
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # Check for bfloat later if needed

# torch compile stuff
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.verbose = False # Set to True for more dynamo details
# os.environ["TORCH_LOGS"] = "recompiles"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# os.environ["TORCH_LOGS"] = "+dynamo"

# run name
characters = string.ascii_letters + string.digits
run_name = ''.join(random.choice(characters) for i in range(6))

# --- get data func ---
def get_batch(split):
    split_filenames = glob.glob(os.path.join("data", f"{data_dir}", f"{split}_*.bin"))
    if not split_filenames:
        raise FileNotFoundError(f"No {split} shard files found in {data_dir}")

    shard_file = np.random.choice(split_filenames)
    try:
        # Assuming header is 256 * 4 bytes based on original code
        data = np.memmap(shard_file, dtype=np.uint16, mode='r', offset=1024)
    except Exception as e:
         print(f"Error memory-mapping file {shard_file}: {e}")
         # Handle error, maybe retry or skip shard
         return get_batch(split) # Simple retry

    num_tokens_in_shard = len(data)

    if num_tokens_in_shard <= block_size + 1:
        print(f"Warning: Shard {shard_file} too small ({num_tokens_in_shard} tokens), resampling...")
        return get_batch(split)

    ix = torch.randint(num_tokens_in_shard - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# --- getting vocab size ---
meta_path = f'data/{data_dir}/meta.pkl'
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    model.config["vocab_size"] = meta_vocab_size
else:
    print(f"Warning: meta.pkl not found at {meta_path}. Using default vocab size: {model.config['vocab_size']}")

# --- loss check ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Ensure model is in eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Use the appropriate context manager for evaluation
            with ctx:
                logits, loss, _ = model(X, Y) # Ignore router weights in eval
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set back to train mode
    return out

# --- Model, Optimizer, Scheduler Init ---
start_iter = 0
scheduler = None # Initialize scheduler variable

if resume:
    print(f"Resuming training from {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    # Load model config from checkpoint if available, otherwise use current args
    if 'config' in checkpoint:
         model.config.update(checkpoint['config'])
         print("Loaded model config from checkpoint.")
    else:
         print("Warning: No config found in checkpoint, using current script args.")

    model = Transformer() # Initialize model with potentially updated config
    state_dict = checkpoint['model']
    # Handle potential issues with compiled models state_dict keys
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            unwrapped_state_dict[k[len('_orig_mod.'):]] = v
        else:
            unwrapped_state_dict[k] = v
    model.load_state_dict(unwrapped_state_dict)
    m = model.to(device)
    print("Model loaded from checkpoint.")

    optimizers = model.configure_optimizers(weight_decay, lr, device) # Get the list
    adamw_optimizer = optimizers[-1] # Assume AdamW is the last optimizer

    # Load optimizer states
    if 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) == len(optimizers):
         for i, opt_state in enumerate(checkpoint['optimizer_states']):
              try:
                   optimizers[i].load_state_dict(opt_state)
                   print(f"Loaded state for optimizer {i} ({type(optimizers[i]).__name__})")
              except Exception as e:
                   print(f"Warning: Could not load state for optimizer {i}: {e}")
    else:
         print("Warning: Optimizer states not found or mismatch in checkpoint. Initializing optimizers from scratch.")

    # Create and load scheduler (assuming it's for AdamW)
    warmup_scheduler = LinearLR(adamw_optimizer, start_factor=1e-3, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(adamw_optimizer, T_max=max_iters - warmup_iters, eta_min=min_lr)
    scheduler = SequentialLR(adamw_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])
    if 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Scheduler state loaded.")
        except Exception as e:
             print(f"Warning: Could not load scheduler state: {e}. Initializing scheduler from scratch.")
    else:
         print("Warning: Scheduler state not found in checkpoint. Initializing scheduler from scratch.")


    # Restore previous run state
    start_iter = checkpoint.get('iter', 0) + 1 # Start from next iteration
    run_name = checkpoint.get('run_name', run_name) # Keep original name if not in ckpt

    train_losses_history = checkpoint.get('train_losses_history', [])
    val_losses_history = checkpoint.get('val_losses_history', [])

    print(f"Resuming run {run_name} from iteration {start_iter}")

else:
    print(f"Starting run {run_name} from scratch")
    model = Transformer()
    m = model.to(device)

    optimizers = model.configure_optimizers(weight_decay, lr, device) # Get the list of optimizers
    # Check if Muon was actually included (based on defensive check in configure_optimizers)
    if len(optimizers) == 1:
         print("WARNING: configure_optimizers returned only one optimizer. Assuming AdamW only.")
         adamw_optimizer = optimizers[0]
         muon_in_use = False
    elif len(optimizers) == 2:
         adamw_optimizer = optimizers[1] # Access AdamW optimizer (index 1)
         muon_in_use = True
         print("Using Muon (optimizer 0) and AdamW (optimizer 1)")
    else:
         print(f"ERROR: Unexpected number of optimizers ({len(optimizers)}) returned. Exiting.")
         exit(1)

    # Create scheduler (for AdamW)
    warmup_scheduler = LinearLR(adamw_optimizer, start_factor=1e-3, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(adamw_optimizer, T_max=max_iters - warmup_iters, eta_min=min_lr)
    scheduler = SequentialLR(adamw_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])

    start_iter = 0

# Compile the model AFTER potentially loading state dict and moving to device
# And AFTER distributed init is done
if "cuda" in device:
    print("compiling the model...")
    try:
        # Disable dynamic shapes if causing issues with compile/distributed
        model = torch.compile(model, fullgraph=False, dynamic=False)
        print("compiled")
    except Exception as e:
        print(f"Torch compile failed: {e}. Running without compile.")


p = sum(p.numel() for p in m.parameters() if p.requires_grad) # Count trainable params
print(f"{p/1e6:.6f} M trainable parameters")

# Create directories if they don't exist
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
if not os.path.exists("plots"):
    os.makedirs("plots")

# --- Training Loop ---
print(f"\nStarting training loop from iteration {start_iter}...")
time_s = time.time()
prev_time = time_s

try: # Wrap training loop in try...finally for cleanup
    for iter in range(start_iter, max_iters + 1):

        # Evaluate loss periodically
        if (iter % eval_interval == 0 or (iter < 100 and iter % 10 == 0) or iter == max_iters) and iter > 0:
            losses = estimate_loss()
            val_losses_history.append(losses['val'])

            time_n = time.time()
            elapsed = time_n - time_s
            dt = time_n - prev_time if iter > start_iter else time_n - time_s # Handle first dt calculation
            prev_time = time_n

            # MFU calculation might need adjustment for MoE/complex attn
            # mfu = model.estimate_mfu(p, batch_size * grad_accum_steps, dt) if hasattr(model, 'estimate_mfu') else 0.0
            # total_flops = 65e12 * elapsed * mfu # Placeholder

            print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, elapsed time: {elapsed/60:.2f} min, dt: {dt*1000:.2f} ms") # Removed MFU/flops for now

            # Save checkpoint
            # Ensure state dicts are handled correctly for compiled/uncompiled models
            model_state_to_save = model.state_dict()
            if hasattr(model, '_orig_mod'):
                 model_state_to_save = model._orig_mod.state_dict()

            # Save optimizer states as a list
            optimizer_states_to_save = [opt.state_dict() for opt in optimizers]

            checkpoint = {
                'model': model_state_to_save,
                'optimizer_states': optimizer_states_to_save, # Save list of states
                'scheduler': scheduler.state_dict() if scheduler else None,
                'iter': iter,
                'run_name': run_name,
                'train_losses_history': train_losses_history,
                'val_losses_history': val_losses_history,
                'config': model.config, # Save model config
                'args': args # Save script arguments
            }

            if losses['val'] < 3.28:

              checkpoint_path = f'checkpoints/{run_name}_check_{iter}.pt'
              print(f"Saving checkpoint to {checkpoint_path}")
              torch.save(checkpoint, checkpoint_path)

            # Plot loss
            #plot_loss(train_losses_history, val_losses_history, eval_interval, iter, run_name)

        # --- Training Step ---
        if iter == max_iters: # Don't do a training step after the last eval
             break

        loss_accum = 0.0
        all_router_weights_accum = []

        # Gradient Accumulation Loop
        for micro_step in range(grad_accum_steps):
            xb, yb = get_batch('train')
            # Determine context based on GradScaler enabled status
            current_ctx = ctx if scaler.is_enabled() else nullcontext()

            with current_ctx:
                 logits, loss, rw = model(xb, yb)
                 loss = loss / grad_accum_steps # Scale loss for accumulation

            if rw: # Check if router weights were returned
                all_router_weights_accum.extend(rw)

            # Scaled backward pass
            scaler.scale(loss).backward()
            loss_accum += loss.item() * grad_accum_steps # Unscale loss item for logging

        train_losses_history.append(loss_accum / grad_accum_steps) # Log average loss for the step

        # Unscale gradients for ALL optimizers BEFORE clipping
        for opt in optimizers:
            scaler.unscale_(opt)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Step EACH optimizer
        for opt in optimizers:
            # Inside loop check for distributed init status (for debugging)
            #if isinstance(opt, Muon):
                 #print(f"Inside loop, before scaler.step(Muon): dist.is_initialized() = {dist.is_initialized()}")
                 #if not dist.is_initialized():
                     #print("ERROR: Default process group IS NOT INITIALIZED right before Muon step!")
            scaler.step(opt)

        # Update GradScaler
        scaler.update()

        # Zero gradients for ALL optimizers
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # Step the scheduler (only affects AdamW)
        if scheduler:
             scheduler.step()

        # Update expert biases (if using DS-MoE and weights were collected)
        if all_router_weights_accum and hasattr(model, 'update_expert_biases'):
            model.update_expert_biases(all_router_weights_accum, 1e-3)


finally: # Ensure cleanup happens even if errors occur
    # --- Distributed Cleanup ---
    if distributed_initialized:
        if dist.is_initialized(): # Check before destroying
            dist.destroy_process_group()
            print("Destroyed default process group.")
        else:
            print("INFO: Default process group was already destroyed or not initialized.")

print('\nTraining finished or interrupted.')
