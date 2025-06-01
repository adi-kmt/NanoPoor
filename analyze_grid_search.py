import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
from typing import Optional

def load_latest_grid_search() -> Optional[pd.DataFrame]:
    """Load the most recent grid search results"""
    pattern = "grid_search_results/lr_search_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print("No grid search results found!")
        return None
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df

def analyze_grid_search_results(df: pd.DataFrame):
    """Analyze and visualize grid search results"""
    
    # Filter to get final validation losses for each run
    final_results = df[df['val_loss'].notna()].groupby(['lr', 'min_lr']).agg({
        'val_loss': 'min',  # Best validation loss achieved
        'step': 'max'       # Final step reached
    }).reset_index()
    
    print("\n🔍 Grid Search Analysis")
    print("=" * 50)
    
    # Best configuration
    best_idx = final_results['val_loss'].idxmin()
    best_config = final_results.iloc[best_idx]
    
    print(f"🌟 Best Configuration:")
    print(f"  Learning Rate: {best_config['lr']:.2e}")
    print(f"  Min Learning Rate: {best_config['min_lr']:.2e}")
    print(f"  Best Val Loss: {best_config['val_loss']:.4f}")
    print(f"  Steps Completed: {best_config['step']}")
    
    # Top 5 configurations
    print(f"\n🏆 Top 5 Configurations:")
    top_5 = final_results.nsmallest(5, 'val_loss')
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {i}. lr={row['lr']:.2e}, min_lr={row['min_lr']:.2e}, val_loss={row['val_loss']:.4f}")
    
    # Statistics
    print(f"\n📊 Statistics:")
    print(f"  Total configurations tested: {len(final_results)}")
    print(f"  Mean val loss: {final_results['val_loss'].mean():.4f}")
    print(f"  Std val loss: {final_results['val_loss'].std():.4f}")
    print(f"  Best/worst ratio: {final_results['val_loss'].min() / final_results['val_loss'].max():.3f}")
    
    # Create visualizations
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Grid Search Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Heatmap of validation losses
    pivot_table = final_results.pivot(index='min_lr', columns='lr', values='val_loss')
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis_r', ax=axes[0, 0])
    axes[0, 0].set_title('Validation Loss Heatmap')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Min Learning Rate')
    
    # 2. Learning rate vs validation loss
    for min_lr in final_results['min_lr'].unique():
        subset = final_results[final_results['min_lr'] == min_lr]
        axes[0, 1].plot(subset['lr'], subset['val_loss'], 'o-', label=f'min_lr={min_lr:.2e}')
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('LR vs Val Loss')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log')
    
    # 3. Min LR ratio vs validation loss
    final_results['min_lr_ratio'] = final_results['min_lr'] / final_results['lr']
    axes[1, 0].scatter(final_results['min_lr_ratio'], final_results['val_loss'], 
                      c=final_results['lr'], cmap='plasma', s=60, alpha=0.7)
    axes[1, 0].set_xlabel('Min LR / LR Ratio')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_title('Min LR Ratio vs Val Loss')
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Learning Rate')
    
    # 4. Distribution of validation losses
    axes[1, 1].hist(final_results['val_loss'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(best_config['val_loss'], color='red', linestyle='--', 
                      label=f'Best: {best_config["val_loss"]:.4f}')
    axes[1, 1].set_xlabel('Validation Loss')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Validation Losses')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('grid_search_results', exist_ok=True)
    plot_path = 'grid_search_results/analysis_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Analysis plot saved to: {plot_path}")
    
    plt.show()
    
    return best_config

def plot_training_curves(df: pd.DataFrame, max_runs: int = 5):
    """Plot training curves for best runs"""
    
    # Get unique runs and their final val losses
    run_summary = df[df['val_loss'].notna()].groupby('run_name')['val_loss'].min().sort_values()
    best_runs = run_summary.head(max_runs).index
    
    plt.figure(figsize=(12, 8))
    
    for i, run_name in enumerate(best_runs):
        run_data = df[df['run_name'] == run_name]
        train_data = run_data[run_data['train_loss'].notna()]
        val_data = run_data[run_data['val_loss'].notna()]
        
        color = plt.cm.tab10(i)
        
        # Plot training loss
        if not train_data.empty:
            plt.plot(train_data['step'], train_data['train_loss'], 
                    color=color, alpha=0.6, linewidth=1, label=f'{run_name} (train)')
        
        # Plot validation loss
        if not val_data.empty:
            plt.plot(val_data['step'], val_data['val_loss'], 
                    color=color, linewidth=2, marker='o', markersize=4,
                    label=f'{run_name} (val)')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'Training Curves - Top {len(best_runs)} Configurations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    curve_path = 'grid_search_results/training_curves.png'
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    print(f"💾 Training curves saved to: {curve_path}")
    plt.show()

def recommend_lr_schedule(best_config: dict):
    """Generate recommendations based on results"""
    
    print(f"\n🎯 Recommendations:")
    print(f"=" * 50)
    
    lr = best_config['lr']
    min_lr = best_config['min_lr']
    min_lr_ratio = min_lr / lr
    
    print(f"✅ Use learning rate: {lr:.2e}")
    print(f"✅ Use min learning rate: {min_lr:.2e}")
    print(f"✅ Min LR ratio: {min_lr_ratio:.3f}")
    
    print(f"\n📝 Suggested command:")
    print(f"python src/train_optimized.py \\")
    print(f"    --lr {lr:.2e} \\")
    print(f"    --min_lr {min_lr:.2e} \\")
    print(f"    --ctx_len 1024 \\")
    print(f"    --n_embd 768 \\")
    print(f"    --n_head 12 \\")
    print(f"    --n_layer 8 \\")
    print(f"    --types mlp mlp mlp mlp mlp mlp mlp mlp \\")
    print(f"    --max_iters 1500 \\")
    print(f"    --eval_interval 150 \\")
    print(f"    --batch_size 8 \\")
    print(f"    --grad_accum 12 \\")
    print(f"    --device cuda \\")
    print(f"    --data_dir tokenized_data")

def main():
    print("🔍 Grid Search Results Analyzer")
    print("=" * 50)
    
    # Load data
    df = load_latest_grid_search()
    if df is None:
        return
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Columns: {list(df.columns)}")
    
    # Analyze results
    best_config = analyze_grid_search_results(df)
    
    # Plot training curves
    plot_training_curves(df)
    
    # Generate recommendations
    recommend_lr_schedule(best_config)

if __name__ == "__main__":
    main() 