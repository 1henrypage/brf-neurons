import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
brf_model_pairs = [
    (
    "models/05-21_20-48-01_init_7675_Adam(0.1),script-bw,NLL,LinearLR,no_gc,4,36,6,bs=16,ep=400,BRF(omega3.0,5.0,b0.1,1.0)LI(20.0,1.0).pt",
    "models/05-21_20-48-01_7675_Adam(0.1),script-bw,NLL,LinearLR,no_gc,4,36,6,bs=16,ep=400,BRF(omega3.0,5.0,b0.1,1.0)LI(20.0,1.0).pt")
    # Add other BRF model pairs here (init, trained)
]


# Analysis functions
def extract_brf_params(state_dict):
    """Extract omega and b_offset from model state dict"""
    params = {}
    for k in state_dict.keys():
        if 'omega' in k:
            params['omega'] = state_dict[k].numpy()
        elif 'b_offset' in k or 'b_' in k:
            params['b_offset'] = state_dict[k].numpy()
    return params


def divergence_boundary(omega, delta=0.01):
    """Compute the divergence boundary p(ω)"""
    return (-1 + np.sqrt(1 - (delta * omega) ** 2)) / delta


def plot_brf_analysis(init_file, trained_file):
    """Plot initial vs trained parameters for one BRF model run"""
    # Load models
    init_state = torch.load(init_file, map_location='cpu')
    trained_state = torch.load(trained_file, map_location='cpu')

    # Extract parameters
    init_params = extract_brf_params(init_state.get('model_state_dict', init_state))
    trained_params = extract_brf_params(trained_state.get('model_state_dict', trained_state))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot parameters
    for ax, params, title in zip([ax1, ax2],
                                 [init_params, trained_params],
                                 ['Initial Parameters', 'Optimized Parameters']):
        # Compute b_c = p(ω) - b'
        omega = params['omega']
        b_c = divergence_boundary(omega) - params['b_offset']

        # Scatter plot
        ax.scatter(omega, b_c, alpha=0.6, label='Neurons')

        # Divergence boundary
        omega_range = np.linspace(0, 10, 100)
        ax.plot(omega_range, divergence_boundary(omega_range), 'r--', label='Divergence Boundary')

        # Formatting
        ax.set_title(title)
        ax.set_xlabel('Angular Frequency (ω) [rad/s]')
        ax.set_ylabel('Dampening Factor (b_c)')
        ax.grid(True)
        ax.legend()

    # Super title with model info
    model_name = Path(trained_file).stem
    plt.suptitle(f'BRF Parameter Analysis\n{model_name}', y=1.02)
    plt.tight_layout()
    plt.savefig("test.png")
    plt.show()


# Run analysis for all BRF model pairs
for init_file, trained_file in brf_model_pairs:
    try:
        plot_brf_analysis(init_file, trained_file)
    except Exception as e:
        print(f"Error processing {trained_file}: {str(e)}")