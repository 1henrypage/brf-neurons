import re
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


# ----------------------------------------------------------------------------
# Plot accuracy curves for a dataset
# ----------------------------------------------------------------------------
dataset = 'shd' # or 'ecg'
ablation = False  # Set to True if you want to include ablation results
files = {
    'BRF': f'experiments/{dataset}/reproduction_results/brf/run-brf-tag-accuracy_test.csv',
    'RF': f'experiments/{dataset}/reproduction_results/rf/run-rf-tag-accuracy_test.csv',
    'ALIF': f'experiments/{dataset}/reproduction_results/alif/run-alif-tag-accuracy_test.csv'
}
if ablation:
    files['BRF-No RP'] = f'experiments/{dataset}/ablation/no_rp/run-rp-tag-accuracy_test.csv'
    files['BRF-No SmR'] = f'experiments/{dataset}/ablation/no_smr/run-smr-tag-accuracy_test.csv'
    files['BRF-No DB'] = f'experiments/{dataset}/ablation/no_db/run-db-tag-accuracy_test.csv'

def plot_acc_curve():
    plt.figure(figsize=(10, 6))

    for label, file in files.items():
        df = pd.read_csv(file)
        plt.plot(df['Step'], df['Value'], label=f'{label}')

    
    plt.title(f'{dataset.upper()}: Test Accuracy vs. Epoch (Step)', fontsize=24)
    plt.xlabel('Epoch (Step)', fontsize=20)
    plt.ylabel('Test Accuracy', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)

    plt.tight_layout()
    ab = '_ablation' if ablation else ""
    plt.savefig(f'experiments/{dataset}/{dataset}_accuracy_plot{ab}.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_acc_curve()


# ----------------------------------------------------------------------------
# Plot Dampening Factor (b_c) vs. Angular Frequency (ω)
# ----------------------------------------------------------------------------
model_files = glob.glob(f"experiments/{dataset}/models/reproduction/*BRF*.pt")

init_file = [f for f in model_files if "init" in f][0]
trained_file = [f for f in model_files if "init" not in f][0]

print(f"Init file: {init_file}")
print(f"Trained file: {trained_file}")

# Constants
DELTA = 0.01  # Time step from the paper

def extract_brf_params(state_dict):
    """Extract omega (ω) and b' parameters from state dict"""
    params = {}
    for k, v in state_dict.items():
        if 'omega' in k:
            params['omega'] = v.numpy()
        elif 'b_offset' in k or 'b_' in k:
            params['b_offset'] = v.numpy()
    return params

def calculate_b_c(omega, b_offset):
    """Calculate b_c = p(ω) - b' using divergence boundary formula"""
    p_omega = (-1 + np.sqrt(1 - (DELTA * omega)**2)) / DELTA
    return p_omega - b_offset

def plot_brf_parameters(init_file, trained_file):
    """Plot initial and optimized parameters on same axes"""
    # Load models
    init_state = torch.load(init_file, map_location='cpu')
    trained_state = torch.load(trained_file, map_location='cpu')
    
    # Extract parameters
    init_params = extract_brf_params(init_state.get('model_state_dict', init_state))
    trained_params = extract_brf_params(trained_state.get('model_state_dict', trained_state))
    
    # Calculate b_c values
    init_b_c = calculate_b_c(init_params['omega'], init_params['b_offset'])
    trained_b_c = calculate_b_c(trained_params['omega'], trained_params['b_offset'])
    
    plt.figure(figsize=(10, 6))
    
    # Plot divergence boundary first (background)
    omega_range = np.linspace(0, 20, 400)
    boundary = (-1 + np.sqrt(1 - (DELTA * omega_range)**2)) / DELTA
    plt.plot(omega_range, boundary, 'k--', label='Divergence Boundary', linewidth=1.5)
    
    # Plot parameters with different transparency
    plt.scatter(init_params['omega'], init_b_c, 
                c='red', alpha=1, label='Initial', edgecolors='black', linewidths=0.5)
    plt.scatter(trained_params['omega'], trained_b_c, 
                c='red', alpha=0.4, label='Optimized', edgecolors='black', linewidths=0.5)
    

    plt.title(f'{dataset.upper()}: BRF Parameter Combinations', pad=20)
    plt.xlabel('Angular Frequency (ω) [rad/s]', labelpad=10)
    plt.ylabel("Dampening Factor (b_c = p(ω) - b')", labelpad=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    

    plt.xlim(-1, 20)
    plt.ylim(-5, 0)  # b_c should be negative
    
    plt.tight_layout()
    plt.savefig(f'experiments/{dataset}/{dataset}_brf_parameters_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_brf_parameters(init_file, trained_file)
