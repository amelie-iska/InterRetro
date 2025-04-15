import torch
import logging
import time

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from retro_star.retro_star.common import prepare_starting_molecules, prepare_mlp, \
    prepare_molstar_planner, smiles_to_fp
from retro_star.retro_star.model import ValueMLP
from retro_star.retro_star.utils import setup_logger

import pickle
from rdkit import Chem

import os
from tqdm import tqdm
dirpath = parent_dir

# For feasibility
import numpy as np
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import Predictive
from bayesian_reactivity_prediction.jax_model import BNN_NUTS
from bayesian_reactivity_prediction.utils import cal_DRFP, disentangle_uncertainty


def parse_route_to_reactions(route_str):
    """
    Parse a retrosynthesis route string into a list of 'reactants>>product' format reactions.

    Args:
        route_str (str): A retrosynthesis route string of the form:
                         product>cost>reactants|intermediate>cost>reactants|...

    Returns:
        List[str]: A list of reaction strings like 'reactant1.reactant2>>product'
    """
    reactions = []
    steps = route_str.split('|')
    product = steps[0].split('>')[0]  # Final target molecule

    for step in steps:
        parts = step.split('>')
        if len(parts) < 3:
            continue  # Skip malformed steps

        step_product = parts[0].strip()
        reactants = parts[2].strip()
        reaction_str = f"{reactants}>>{step_product}"
        reactions.append(reaction_str)

    return reactions


def get_feasibility(rxn, verbose=False):

    def init_model(model_path="../bayesian_reactivity_prediction/models/BNN_NUTS.npz"):
        model = BNN_NUTS(2048, 16, 1)
        with np.load(model_path, mmap_mode='r') as data:
            model_samples = {key: jnp.array(data[key]) for key in data.keys()}
        predictive_probs = Predictive(
            model.jax_model, posterior_samples=model_samples, return_sites=["prob"])
        return predictive_probs

    def inf(rxn_lst, predictive_probs, seed=666, verbose=False):
        fps = jnp.array(cal_DRFP(rxn_lst))
        _, rng_key_predict = random.split(random.PRNGKey(seed))
        probs_posterior = np.array(predictive_probs(
            rng_key_predict, fps, None, 16)["prob"])
        predictions = probs_posterior.T
        mean, predictive_uncertainty, aleatoric_uncertainty, epistemic_uncertainty = disentangle_uncertainty(
            predictions)
        if verbose:
            print(
                f"mean: {mean}, \npredictive_uncertainty: {predictive_uncertainty}, \naleatoric_uncertainty: {aleatoric_uncertainty}, \nepistemic_uncertainty: {epistemic_uncertainty}")
        flag = (mean > 0.5).astype(int)
        return flag

    predictive_probs = init_model()
    flag = inf(rxn, predictive_probs, seed=666, verbose=verbose)
    return flag

class RSPlanner:
    def __init__(self,
                 one_step,
                 gpu=-1,
                 expansion_topk=50,
                 iterations=500,
                 use_value_fn=False,
                 starting_molecules=dirpath+'/dataset/origin_dict_canonical.csv',
                 save_folder=dirpath+'/saved_models',
                 value_model='best_epoch_final_4.pt',
                 fp_dim=2048,
                 viz=False,
                 viz_dir='viz'):

        setup_logger()
        device = 'mps'
        starting_mols = prepare_starting_molecules(starting_molecules)


        if use_value_fn:
            model = ValueMLP(
                n_layers=1,
                fp_dim=fp_dim,
                latent_dim=128,
                dropout_rate=0.1,
                device=device
            ).to(device)
            model_f = '%s/%s' % (save_folder, value_model)
            logging.info('Loading value nn from %s' % model_f)
            model.load_state_dict(torch.load(model_f, map_location=device))
            model.eval()

            def value_fn(mol):
                fp = smiles_to_fp(mol, fp_dim=fp_dim).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
        else:
            value_fn = lambda x: 0.

        self.plan_handle = prepare_molstar_planner(
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
        )

    def plan(self, target_mol, target_mol_id):
        t0 = time.time()
        succ, msg = self.plan_handle(target_mol, target_mol_id)

        if succ:
            result = {
                'succ': succ,
                'time': time.time() - t0,
                'iter': msg[1],
                'routes': msg[0].serialize(),
                'route_cost': msg[0].total_cost,
                'route_len': msg[0].length
            }
            return result

        else:
            # logging.info('Synthesis path for %s not found. Please try increasing the number of iterations.' % target_mol)
            return None


if __name__ == '__main__':
    from config_parser import get_args
    from agents import return_agent
    args = get_args()
    one_step = return_agent(args)
    eval_routes = '../dataset/routes_possible_test_hard.pkl'

    checkpoint = torch.load('models_epoch3_update1563.pth', map_location='cpu')
    one_step.load_state_dict(checkpoint)

    eval_routes = pickle.load(open(eval_routes, 'rb'))
    print(f"{len(eval_routes)} evaluation routes have been loaded.")
    eval_mols = []
    for route in eval_routes:
        eval_mols.append(Chem.CanonSmiles(route[0].split('>')[0]))
    del eval_routes
    eval_mols = list(set(eval_mols))
    print(f"{len(eval_mols)} evaluation molecules are loaded after removing duplicates. ")

    planner = RSPlanner(
        one_step=one_step,
        gpu=0,
        use_value_fn=False,
        iterations=50,
        expansion_topk=1,
        viz=False,
        viz_dir='viz'
    )
    logs = {'successes': [], 'depth': [], 'feasibility': []}
    for id, mol in enumerate(tqdm(eval_mols)):
        result = planner.plan(mol, id)

        if result is not None:
            reactions = parse_route_to_reactions(result['routes'])
            feasibility = []
            for rxn in reactions:
                f = get_feasibility(rxn, verbose=True)
                feasibility.append(f.item())

            logs['feasibility'].append(min(feasibility))
            logs['successes'].append(int(result['succ']))
            logs['depth'].append(int(result['route_len']))
        else:
            logs['successes'].append(0)

        if id >= 99:
            break

    print(logs)
    print('Feasible routes among successful routes', sum(logs['feasibility']) / len(logs['feasibility']))
    print('Average Feasible rate', sum(logs['feasibility']) / len(logs['successes']))
    print('Success rate', sum(logs['successes']) / len(logs['successes']))
    print('Average Depth of Success Path', sum(logs['depth']) / len(logs['depth']))

