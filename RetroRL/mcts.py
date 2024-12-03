import numpy as np

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