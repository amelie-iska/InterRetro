import torch
import logging
import time
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
                 agent,
                 gpu=-1,
                 expansion_topk=50,
                 iterations=500,
                 use_value_fn=False,
                 starting_molecules=dirpath+'/dataset/origin_dict.csv',
                 mlp_templates=dirpath+'/one_step_model/template_rules_1.dat',
                 mlp_model_dump=dirpath+'/one_step_model/saved_rollout_state_1_2048.ckpt',
                 save_folder=dirpath+'/saved_models',
                 value_model='best_epoch_final_4.pt',
                 fp_dim=2048,
                 viz=False,
                 viz_dir='viz'):

        setup_logger()
        device = 'cpu'
        starting_mols = prepare_starting_molecules(starting_molecules)

        # one_step = prepare_mlp(mlp_templates, mlp_model_dump)

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
            one_step=agent,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=expansion_topk,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir
        )

    def plan(self, target_mol):
        t0 = time.time()
        succ, msg = self.plan_handle(target_mol)

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
            logging.info('Synthesis path for %s not found. Please try increasing '
                         'the number of iterations.' % target_mol)
            return None


if __name__ == '__main__':
    from config_parser import get_args
    from agent import RetroRL
    args = get_args()
    agent = RetroRL(args)

    eval_routes = pickle.load(open(args.eval_routes, 'rb'))
    print(f"{len(eval_routes)} evaluation routes have been loaded.")
    eval_mols = []
    for route in eval_routes:
        eval_mols.append(Chem.CanonSmiles(route[0].split('>')[0]))
    del eval_routes
    eval_mols = list(set(eval_mols))
    print(f"{len(eval_mols)} evaluation molecules are loaded after removing duplicates. ")

    planner = RSPlanner(
        agent=agent,
        gpu=0,
        use_value_fn=False,
        iterations=20,
        expansion_topk=1
    )
    '''
    result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
    print(result)
    '''
    logs = {'successes': []}
    for mol in tqdm(eval_mols):
        result = planner.plan(mol)
        if result is not None:
            logs['successes'].append(int(result['succ']))
        else:
            logs['successes'].append(0)

    print(logs)
    print(sum(logs['successes']) / len(logs['successes']))
