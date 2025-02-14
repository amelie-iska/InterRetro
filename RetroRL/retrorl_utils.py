import os
import torch
import random
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem

def minicut(input, minirun_rate):
    output = input[:int(len(input)/minirun_rate)]
    return output
    

def plot_value_estimation(estimated_steps, real_steps, epoch, save_path):
    mse = mean_squared_error(real_steps, estimated_steps)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_steps, estimated_steps)
    r2 = r2_score(real_steps, estimated_steps)
    corr = np.corrcoef(estimated_steps, real_steps)[0, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(7,6))

    # Scatter plot
    ax.scatter(estimated_steps, real_steps,
               alpha=0.7, color='blue', edgecolors='k', s=80, label='Data points')

    # Plot the diagonal y=x for reference
    min_val = min(estimated_steps.min(), real_steps.min())
    max_val = max(estimated_steps.max(), real_steps.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    
    # ax.set_xlim(0, 25)
    # ax.set_ylim(0, 25)
    
    # Add labels, legend, and grid
    ax.set_xlabel('Estimated Steps', fontsize=12)
    ax.set_ylabel('Real Steps', fontsize=12)
    ax.set_title(f'Estimated vs. Real Steps (Epoch {epoch})', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Prepare metrics text
    metrics_text = (
        f"Correlation: {corr:.2f}\n"
        f"R²: {r2:.2f}\n"
        f"MSE: {mse:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"MAE: {mae:.2f}"
    )

    # Place the text box in the lower-right corner
    ax.text(
        0.95, 0.05,
        metrics_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.2)
    )

    # Save BEFORE show
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Now show the figure (if desired)
    # plt.show()
    plt.close(fig)
    
    return {'correlation': corr,
            'r2': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae}

    

def prepare_starting_molecules(filename):
    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)
    return starting_mols

def smiles2array(X, fp_dim):
    # Compute fingerprint from mol to feature
    mol = Chem.MolFromSmiles(X)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim),useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    # arr = (arr - arr.mean())/(arr.std() + 0.000001)
    # arr = arr / fp_dim
    # X = fps_to_arr(X)
    return arr

def set_seeds(seed):
    if seed < 0:
        seed = random.randint(0, 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return seed

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def create_exp(seed, group):
    base_dir = 'experiments'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    current_time = datetime.now().strftime("%m%d-%H:%M:%S")
    run_dir = os.path.join(base_dir, f"{group}{current_time}-s{seed}")
    os.makedirs(run_dir)
    os.makedirs(f'{run_dir}/figures')
    return run_dir
