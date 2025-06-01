#!/usr/bin/env python3
"""
Demo script to test the optimized training implementation
"""

import os
import sys
import subprocess
import time
import tempfile
import numpy as np

def create_dummy_data():
    """Create dummy training data for testing"""
    print("📁 Creating dummy data...")
    
    # Create data directory
    data_dir = "data/demo_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create dummy shards
    vocab_size = 1000
    seq_len = 1024
    num_sequences = 1000  # Small dataset for demo
    
    # Generate random token sequences
    for split in ['train', 'val']:
        for shard in range(2):  # 2 shards per split
            filename = f"{data_dir}/{split}_{shard:04d}.bin"
            
            # Create header (1024 bytes) + data
            with open(filename, 'wb') as f:
                # Write header (zeros for simplicity)
                f.write(b'\x00' * 1024)
                
                # Write token data
                tokens = np.random.randint(0, vocab_size, size=num_sequences * seq_len, dtype=np.uint16)
                f.write(tokens.tobytes())
    
    # Create meta.pkl
    import pickle
    meta = {'vocab_size': vocab_size}
    with open(f"{data_dir}/meta.pkl", 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"✅ Created dummy data in {data_dir}")
    return data_dir

def run_demo_training():
    """Run a quick demo training"""
    
    print("🚀 Running Demo Training")
    print("=" * 50)
    
    # Create dummy data
    data_dir = create_dummy_data()
    
    # Demo configuration (small and fast)
    cmd = [
        "python", "src/train_optimized.py",
        "--ctx_len", "128",          # Shorter context
        "--n_embd", "128",           # Smaller embedding
        "--n_head", "4",             # Fewer heads
        "--n_layer", "2",            # Fewer layers
        "--lr", "1e-3",
        "--min_lr", "1e-4",
        "--max_iters", "50",         # Very short training
        "--eval_interval", "10",
        "--batch_size", "4",
        "--grad_accum", "2",
        "--device", "cpu",           # Use CPU for demo
        "--data_dir", "demo_data",
        "--types", "mlp", "mlp"      # Simple MLP layers only
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        print("📤 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("📥 STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Demo training completed successfully!")
        else:
            print(f"❌ Demo training failed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Demo training timed out (took longer than 5 minutes)")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def run_demo_grid_search():
    """Run a mini grid search demo"""
    
    print("\n🔍 Running Demo Grid Search")
    print("=" * 50)
    
    # Use the same dummy data
    cmd = [
        "python", "src/train_optimized.py",
        "--ctx_len", "64",           # Even smaller for grid search
        "--n_embd", "64",
        "--n_head", "2",
        "--n_layer", "2",
        "--max_iters", "20",         # Very short for each config
        "--eval_interval", "5",
        "--batch_size", "2",
        "--grad_accum", "1",
        "--device", "cpu",
        "--data_dir", "demo_data",
        "--types", "mlp", "mlp",
        "--grid_search"              # Enable grid search
    ]
    
    print("Running grid search command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        print("📤 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("📥 STDERR:")  
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Demo grid search completed successfully!")
            
            # Try to analyze results
            try:
                print("\n📊 Analyzing results...")
                import subprocess
                analysis_result = subprocess.run(["python", "analyze_grid_search.py"], 
                                               capture_output=True, text=True, timeout=60)
                if analysis_result.returncode == 0:
                    print("✅ Analysis completed!")
                else:
                    print("⚠️  Analysis had some issues, but that's expected for demo data")
            except:
                print("⚠️  Could not run analysis (missing dependencies?)")
                
        else:
            print(f"❌ Demo grid search failed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Demo grid search timed out")
    except Exception as e:
        print(f"❌ Error running demo grid search: {e}")

def main():
    print("🎯 Optimized Training Demo")
    print("=" * 60)
    print("This script will test the optimized training implementation")
    print("with small dummy data and configurations.")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Please run this script from the project root directory")
        print("   (should contain src/ directory)")
        sys.exit(1)
    
    # Test 1: Simple training
    run_demo_training()
    
    print("\n" + "="*60)
    
    # Test 2: Grid search (optional)
    user_input = input("\n🤔 Would you like to run a demo grid search? (y/N): ").strip().lower()
    if user_input in ['y', 'yes']:
        run_demo_grid_search()
    else:
        print("⏭️  Skipping grid search demo")
    
    print("\n🎉 Demo complete!")
    print("\nNext steps:")
    print("1. 📁 Prepare your real tokenized data in data/tokenized_data/")
    print("2. 🔍 Run grid search: python src/train_optimized.py --grid_search")
    print("3. 📊 Analyze results: python analyze_grid_search.py")
    print("4. 🎯 Train with best params: python src/train_optimized.py --lr <best_lr> --min_lr <best_min_lr>")

if __name__ == "__main__":
    main() 