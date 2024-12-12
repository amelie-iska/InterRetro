import os
import numpy as np

from alg import online_mcts, molstar
from models import MLPModel, ValueMLP, PriorMLP
from utils import prepare_starting_molecules, smiles2array

import logging

import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
from multiprocessing import Queue
from time import sleep
from viz_tree.mol_tree_to_paroutes_dict import mol_tree_to_paroutes_dict

mp.set_start_method('spawn', force=True)

class SerialRunner:
    def __init__(self, args):
        self.args = args
        device = torch.device('cuda' if args.n_gpus > 0 else 'cpu')
        
        building_blocks = prepare_starting_molecules(args.building_blocks_path)
        logging.info(f"{len(building_blocks)} building blocks have been loaded. ")

        # load one-step model
        one_step = MLPModel(state_path=args.state_path, template_path=args.template_path, 
                            device=args.device, realistic_filter=args.realistic_filter)
        self.model_net = one_step.net

        # prepare value function
        self.value_net = ValueMLP(n_layers=1, fp_dim=args.fp_dim, latent_dim=128,
                                  dropout_rate=0.1, device=device)
        self.value_net.eval()

        if args.use_value_fn:
            def value_fn(mol):
                fp = smiles2array(mol, fp_dim=args.fp_dim).reshape(1,-1)
                fp = torch.FloatTensor(fp).to(device)
                v = self.value_net(fp).item()
                return v
        else:
            value_fn = lambda x: 0

        # prepare prior function
        self.prior_net = PriorMLP(n_layers=1, fp_dim=args.fp_dim, latent_dim=128,
                                  dropout_rate=0.1, device=device)
        self.prior_net.eval()

        def prior_fn(mol):
            fp = smiles2array(mol, fp_dim=args.fp_dim).reshape(1,-1)
            fp = torch.FloatTensor(fp).to(device)
            p = self.prior_net(fp).item()
            return p

        self.onlinemcts_handle = lambda x, y=0, test=False: online_mcts(
            target_mol=x,
            target_mol_id=y,
            starting_mols=building_blocks,
            expand_fn=lambda x: one_step.run(x, topk=args.expansion_topk),
            value_fn=value_fn,
            prior_fn=prior_fn,
            iterations=args.iterations,
            args=args,
            viz=False,
            viz_dir='/viz',
            test=test,
        )
        

    def run(self, target_mols, target_mol_ids, test=False):
        batch_result = {
            'ids': [],
            'succ': [],
            'iter': [],
            'route_costs': [],
            'route_lens': [],
        }
        mols_batch, vals_batch, tmps_batch = [], [], []
        for mol_id, target_mol in zip(target_mol_ids, target_mols):
            # succ, (mol_tree, iter) = self.rollout_handle(target_mol, mol_id, test=test)
            succ, (mol_tree, iter) = self.rollout_handle(target_mol, mol_id)
            # generate training data
            mols, vals, tmps = mol_tree.root.generate_training_dataset()
            mols_batch += mols
            vals_batch += vals
            tmps_batch += tmps

            # record batch results
            batch_result['ids'].append(mol_id)
            batch_result['succ'].append(succ)
            batch_result['iter'].append(iter)
            batch_result['route_costs'].append(mol_tree.value)
            batch_result['route_lens'].append(mol_tree.depth)

        return mols_batch, vals_batch, tmps_batch, batch_result

    def mcts(self, target_mols, target_mol_ids, test=False):
        batch_result = {
            "ids": [],
            "succ": [],
            "simulate_succ": [],
            "iter": [],
            "route_lens": [],
            "num_succ_before_take": 0,
            "num_take_unsucc_when_succ": 0,
            "num_deadends": 0,
            "num_building_blocks": 0,
            "num_unexpanded": 0,
            "route_shortest_cost": [],
            "mol_nodes": [],
            "reaction_nodes": [],
            "nll_reference": [], # negative log likelihood given by the reference model
            "nll": [], # negative log likelihood given by the trained model
            "succ_len": [],
        }
        mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch = [], [], [], [], [], []
        for mol_id, target_mol in zip(target_mol_ids, target_mols):
            succ, (mol_tree, iter), result = self.onlinemcts_handle(target_mol, mol_id, test=test)
            mols_r_batch += result["mols_r"]
            pris_batch += result["pris"]
            mols_v_batch += result["mols_v"]
            vals_batch += result["vals"]
            mols_t_batch += result["mols_t"]
            tmps_batch += result["tmps"]

            # record batch results
            batch_result['ids'].append(mol_id)
            batch_result['succ'].append(succ)
            batch_result['simulate_succ'].append(mol_tree.simulate_succ)
            batch_result['iter'].append(iter)
            batch_result['route_lens'].append(mol_tree.depth)
            batch_result["num_succ_before_take"] += result["num_succ_before_take"]
            batch_result["num_take_unsucc_when_succ"] += result["num_take_unsucc_when_succ"]
            batch_result["num_deadends"] += result["num_deadends"]
            batch_result["num_building_blocks"] += result["num_building_blocks"]
            batch_result["num_unexpanded"] += result["num_unexpanded"]
            batch_result['mol_nodes'].append(len(mol_tree.mol_nodes))
            batch_result['reaction_nodes'].append(len(mol_tree.reaction_nodes))
            batch_result['nll'].append(mol_tree.root.cumulative_nll)
            batch_result['nll_reference'].append(mol_tree.root.cumulative_nll_reference)
            batch_result['succ_len'].append(mol_tree.root.succ_len)
            if mol_tree.root.succ_len != np.inf:
                batch_result["route_shortest_cost"].append(mol_tree.root.succ_len)

        return mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result

    def plan(self, target_mols, target_mol_ids):
        mols_batch, vals_batch, tmps_batch = [], [], []
        batch_result = {
            'ids': [],
            'succ': [],
            'iter': [],
            'route_costs': [],
            'route_lens': [],
        }
        for mol_id, target_mol in zip(target_mol_ids, target_mols):
            # generate training data
            succ, (mols, vals, tmps) = self.retrostar_handle(target_mol, mol_id)
            mols_batch += mols
            vals_batch += vals
            tmps_batch += tmps

            # record batch results
            batch_result['ids'].append(mol_id)
            batch_result['succ'].append(succ)

        return mols_batch, vals_batch, tmps_batch, batch_result

    def retro(self, target_mols, target_mol_ids):
        batch_result = {
            "ids": [],
            "succ": [],
            "iter": [],
            "mol_nodes": [],
            "reaction_nodes": [],
            "nll_reference": [], # negative log likelihood given by the reference model
            "nll": [], # negative log likelihood given by the trained model
            "succ_len": [],
        }
        for mol_id, target_mol in zip(target_mol_ids, target_mols):
            # generate training data
            succ, msg = self.molstar_handle(target_mol, mol_id)

            # record batch results
            batch_result['ids'].append(mol_id)
            batch_result['succ'].append(succ)
            batch_result['iter'].append(msg[1])
            batch_result['mol_nodes'].append(len(msg[2].mol_nodes))
            batch_result['reaction_nodes'].append(len(msg[2].reaction_nodes))     
            if succ:
                # save successful route
                succ_route = mol_tree_to_paroutes_dict(msg[2].root)
                # f = open(self.args.result_folder + 'routes_dict/' + 'mol_{}.json'.format(mol_id), 'w')
                # json.dump(succ_route, f, indent=4)
                # f.close()

                batch_result['nll_reference'].append(succ_route["value_reference"])
                batch_result['nll'].append(succ_route["succ_value"])
                batch_result['succ_len'].append(msg[0].length)
            else:
                batch_result['nll_reference'].append(0)
                batch_result['nll'].append(0)
                batch_result['succ_len'].append(0)

        return batch_result

    def close(self):
        pass

    def load_model(self, model_state_dict, prior_state_dict=None, value_state_dict=None):
        self.model_net.load_state_dict(model_state_dict)
        if value_state_dict:
            self.value_net.load_state_dict(value_state_dict)
        if prior_state_dict:
            self.prior_net.load_state_dict(prior_state_dict)