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
dirpath = os.path.dirname(os.path.abspath(__file__))


class RSPlanner:
    def __init__(self,
                 one_step,
                 gpu=-1,
                 expansion_topk=50,
                 iterations=500,
                 use_value_fn=False,
                 starting_molecules=dirpath+'/dataset/origin_dict.csv',
                 save_folder=dirpath+'/saved_models',
                 value_model='best_epoch_final_4.pt',
                 fp_dim=2048,
                 viz=False,
                 viz_dir='viz'):

        setup_logger()
        device = 'cpu'
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

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import (
     LocalRetroModel, Graph2EditsModel, ChemformerModel, MEGANModel)
class OneStep:
    
    def __init__(self, model_name):
        if model_name == 'LocalRetroModel':
            self.model = LocalRetroModel()
        elif model_name == 'Graph2EditsModel':
            self.model = Graph2EditsModel()
        elif model_name == 'ChemformerModel':
            self.model = ChemformerModel()
        elif model_name == 'MEGANModel':
            self.model = MEGANModel()
        else:
            raise Exception(f'{model_name} does not exist.')
    
    def mols_to_str(self, mols) -> str:
        return ".".join([mol.smiles for mol in mols])
    
    def run(self, p_smi, topk=10):
        test_mol = Molecule(p_smi)
        [results] = self.model([test_mol], num_results=max(10, topk))
        
        all_reactants = []
        for idx, prediction in enumerate(results):
            all_reactants.append(self.mols_to_str(prediction.reactants))
        return {'reactants': all_reactants[:topk],
                'scores': [1 for i in range(len(all_reactants))][:topk],
                'template': [1 for i in range(len(all_reactants))][:topk],
                'scores_reference': [1 for i in range(len(all_reactants))][:topk]}


if __name__ == '__main__':
    from config_parser import get_args
    from agents import return_agent
    args = get_args()
    # one_step = return_agent(args)
    one_step = OneStep('MEGANModel')
    eval_routes = '../dataset/routes_possible_test_hard.pkl'

    eval_routes = pickle.load(open(args.eval_routes, 'rb'))
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
        iterations=100,
        expansion_topk=50,
        viz=False,
        viz_dir='viz'
    )


    logs = {'successes': [], 'depth': []}
    for id, mol in enumerate(tqdm(eval_mols)):
        result = planner.plan(mol, id)
        if result is not None:
            logs['successes'].append(int(result['succ']))
            logs['depth'].append(int(result['route_len']))
        else:
            logs['successes'].append(0)

    print(logs)
    print(sum(logs['successes']) / len(logs['successes']))
    print(sum(logs['depth']) / len(logs['depth']))

