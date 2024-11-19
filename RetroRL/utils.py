import os
import torch
import random
import pickle
import pandas as pd
import numpy as np

from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem

from alg import online_mcts


def prepare_onlinemcts_planner(one_step, value_fn, prior_fn, starting_mols, expansion_topk,
                         iterations, args, viz=False, viz_dir=None):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)

    plan_handle = lambda x, y=0, test=False: online_mcts(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        value_fn=value_fn,
        prior_fn=prior_fn,
        iterations=iterations,
        args=args,
        viz=viz,
        viz_dir=viz_dir,
        test=test,
    )

    return plan_handle

def minicut(input, minirun):
    if minirun:
        output = input[:int(len(input)/100)]
        return output
    else:
        return input
    
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
    return run_dir