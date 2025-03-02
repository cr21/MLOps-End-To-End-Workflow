import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from typing import Dict, Any, List
import seaborn as sns
import numpy as np

def read_experiment_data(log_dir: Path) -> List[Dict[str, Any]]:
    experiments = []
    for run_dir in log_dir.glob("*/*"):
        metrics_file = list(run_dir.glob("logs/csv_logs/version_*/metrics.csv"))
        hparams_file = list(run_dir.glob("logs/csv_logs/version_*/hparams.yaml"))
        
        if not metrics_file or not hparams_file:
            continue
        
        metrics_df = pd.read_csv(metrics_file[0])
        with open(hparams_file[0], 'r') as f:
            hparams = yaml.safe_load(f)
        
        experiment = {
            'hparams': hparams,
            'metrics': metrics_df
        }
        experiments.append(experiment)
    
    return experiments

def create_hyperparameter_table(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, exp in enumerate(experiments, start=1):
        hparams = exp['hparams']
        metrics_df = exp['metrics']
        
        row = {'experiment': f"{hparams['base_model']}_{i}", **hparams}
        
        # Add aggregated metrics
        metrics = ['loss', 'acc']
        phases = ['train', 'val', 'test']
        
        for phase in phases:
            for metric in metrics:
                col_name = f'{phase}/{metric}'
                if col_name in metrics_df.columns:
                    if metric == 'acc':
                        row[f'{phase}_{metric}_max'] = metrics_df[col_name].max()
                    else:  # loss
                        row[f'{phase}_{metric}_min'] = metrics_df[col_name].min()
        
        # Add test_best_acc
        if 'test/acc' in metrics_df.columns:
            row['test_best_acc'] = metrics_df['test/acc'].max()
        else:
            row['test_best_acc'] = None
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def experiments_to_dataframe(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    all_data = []
    for i, exp in enumerate(experiments, start=1):
        metrics_df = exp['metrics']
        hparams = exp['hparams']
        
        exp_data = metrics_df.copy()
        exp_data['experiment'] = i
        
        for key, value in hparams.items():
            exp_data[key] = value
        
        all_data.append(exp_data)
    
    return pd.concat(all_data, ignore_index=True)

def plot_experiment_metrics(df: pd.DataFrame):
    metrics = ['loss', 'acc']
    phases = ['train', 'val']
    
    # Create directories for saving plots and markdown results
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    markdown_dir = Path("markdown_results")
    markdown_dir.mkdir(exist_ok=True)

    # Plot train and val metrics over epochs
    for metric in metrics:
        for phase in phases:
            column_name = f'{phase}/{metric}'
            if column_name in df.columns:
                plt.figure(figsize=(12, 6))
                experiments = df['experiment'].unique()
                color_palette = sns.color_palette("husl", n_colors=len(experiments))
                
                for i, exp in enumerate(experiments):
                    exp_data = df[df['experiment'] == exp]
                    exp_data = exp_data.dropna(subset=[column_name])
                    if not exp_data.empty:
                        final_value = exp_data[column_name].iloc[-1]
                        plt.plot(exp_data['epoch'], exp_data[column_name], 
                                 label=f'Experiment {exp} ({final_value:.4f})', 
                                 color=color_palette[i])
                
                plt.title(f'{phase.capitalize()} {metric.capitalize()} Over Epochs', fontsize=16)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(f'{phase.capitalize()} {metric.capitalize()}', fontsize=12)
                plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(plots_dir / f'{phase}_{metric}_plot.png', dpi=300, bbox_inches='tight')
                plt.close()

    # Plot test metrics and best test acc as bar plots
    test_metrics = ['test/loss', 'test/acc', 'test/acc_best']
    for metric in test_metrics:
        plt.figure(figsize=(12, 6))
        experiments = df['experiment'].unique()
        color_palette = sns.color_palette("husl", n_colors=len(experiments))
        
        values = []
        for i, exp in enumerate(experiments):
            exp_data = df[df['experiment'] == exp]
            if metric in exp_data.columns:
                if metric == 'test/loss':
                    value = exp_data[metric].min()
                else:  # acc metrics (both 'test/acc' and 'test_best_acc')
                    value = exp_data[metric].max()
                values.append(value)
                plt.bar(exp, value, color=color_palette[i], 
                        label=f'Experiment {exp} ({value:.4f})')
        
        metric_name = 'Loss' if 'loss' in metric else 'Accuracy'
        plt.title(f'Test {metric_name} for Each Experiment', fontsize=16)
        plt.xlabel('Experiment', fontsize=12)
        plt.ylabel(f'Test {metric_name}', fontsize=12)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Use a different naming convention for the test metric plots
        if metric == 'test_best_acc':
            plot_filename = 'test_best_acc_plot.png'
        else:
            plot_filename = f'test_{metric.split("/")[-1]}_plot.png'
        plt.savefig(plots_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

    print("\nMetrics for each experiment:")
    for exp in experiments:
        exp_data = df[df['experiment'] == exp]
        print(f"\nExperiment {exp}:")
        for phase in phases + ['test']:
            for metric in metrics:
                column_name = f'{phase}/{metric}'
                if column_name in exp_data.columns:
                    if metric == 'acc':
                        best_value = exp_data[column_name].max()
                        print(f"Best {phase} {metric}: {best_value:.4f}")
                    else:  # loss
                        best_value = exp_data[column_name].min()
                        print(f"Best {phase} {metric}: {best_value:.4f}")
        if 'test_best_acc' in exp_data.columns:
            test_best_acc = exp_data['test_best_acc'].max()
            print(f"Best test accuracy: {test_best_acc:.4f}")

# def plot_train_loss(df: pd.DataFrame):
#     # Check if required columns exist
#     if 'experiment' not in df.columns or 'train/loss' not in df.columns:
#         print("Error: Required columns 'experiment' and 'train/loss' not found in the DataFrame.")
#         return
    
#     # Check if 'epoch' column exists, if not, try to use 'step' instead
#     if 'epoch' not in df.columns:
#         if 'step' in df.columns:
#             print("Warning: 'epoch' column not found. Using 'step' instead.")
#             df['epoch'] = df['step']
#         else:
#             print("Error: Neither 'epoch' nor 'step' column found in the DataFrame.")
#             return
    
#     # Group by experiment and epoch, then take the minimum non-NaN value for train/loss
#     df_min_loss = df.groupby(['experiment', 'epoch'])['train/loss'].apply(lambda x: x.min() if x.notna().any() else np.nan).reset_index()
    
#     # Create a directory for saving plots
#     plots_dir = Path("plots")
#     plots_dir.mkdir(exist_ok=True)
    
#     # Create a new figure
#     plt.figure(figsize=(12, 6))
    
#     # Get unique experiments
#     experiments = df_min_loss['experiment'].unique()
    
#     # Create a color palette for the experiments
#     color_palette = sns.color_palette("husl", n_colors=len(experiments))
    
#     # Plot train loss for each experiment
#     for i, exp in enumerate(experiments):
#         exp_data = df_min_loss[df_min_loss['experiment'] == exp]
#         exp_data = exp_data.dropna(subset=['train/loss'])  # Remove rows with NaN loss
#         if not exp_data.empty:
#             final_loss = exp_data['train/loss'].iloc[-1]  # Get the last (minimum) non-NaN loss value
#             plt.plot(exp_data['epoch'], exp_data['train/loss'], 
#                      label=f'Experiment {exp} ({final_loss:.4f})', 
#                      color=color_palette[i])
    
#     plt.title('Minimum Training Loss Over Epochs', fontsize=16)
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Minimum Training Loss', fontsize=12)
#     plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.savefig(plots_dir / 'train_loss_plot.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Print the minimum loss for each experiment and epoch
#     print("\nMinimum loss for each experiment and epoch:")
#     for exp in experiments:
#         exp_data = df_min_loss[df_min_loss['experiment'] == exp]
#         exp_data = exp_data.dropna(subset=['train/loss'])  # Remove rows with NaN loss
#         print(f"\nExperiment {exp}:")
#         for _, row in exp_data.iterrows():
#             print(f"Epoch {row['epoch']}: {row['train/loss']:.4f}")

def main():
    log_dir = Path("logs/train/multiruns")
    experiments = read_experiment_data(log_dir)
    print(f"Found {len(experiments)} experiments")
    
    # Create markdown_results directory if it doesn't exist
    markdown_dir = Path("markdown_results")
    markdown_dir.mkdir(exist_ok=True)
    
    # Create and save hyperparameter table
    table = create_hyperparameter_table(experiments)
    table_sorted = table.sort_values(
        by=['test_best_acc', 'test_acc_max', 'test_loss_min'], 
        ascending=[False, False, True]
    )
    table_sorted.to_markdown(markdown_dir / 'hyperparameter_table.md', index=False, floatfmt=".6f")
    
    # Convert experiments to DataFrame and save as CSV
    experiments_df = experiments_to_dataframe(experiments)
    #experiments_df.to_csv('experiments_data.csv', index=False)
    
    # Plot experiment metrics using the DataFrame
    plot_experiment_metrics(experiments_df)

if __name__ == "__main__":
    main()
