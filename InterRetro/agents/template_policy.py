
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import copy

from rdkit import Chem

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from models import DMPNNValue, MLPModel
from retrorl_utils import smiles2array
from Graph2Edits.prepare_data import apply_edit_to_mol
from Graph2Edits.preprocess import check_edits
from Graph2Edits.models import Graph2Edits, BeamSearch
from Graph2Edits.models.model_utils import get_seq_edit_accuracy
from Graph2Edits.utils.generate_edits import generate_reaction_edits
from Graph2Edits.utils.rxn_graphs import MolGraph, RxnGraph, Vocab
from Graph2Edits.utils.collate_fn import get_batch_graphs, prepare_edit_labels

from rxnmapper import RXNMapper

import sys
import joblib
import os

class TemplatePolicy:
    def __init__(self, args):
        self.args = args
        self.updates = 0
        
        self.fp_dim = 2048

        self.policy = MLPModel(state_path='../retro_data/one_step_model/saved_rollout_state_1_2048.ckpt', 
                               template_path='../retro_data/one_step_model/template_rules_1.dat', 
                               device=args.device)
        for m in self.policy.net.modules():
            m.eval()
        self.policy_optim = torch.optim.Adam(self.policy.net.parameters(), lr=args.lr)
        
        
        checkpoint = torch.load(args.graph2edits_model, map_location=torch.device('cpu'))
        config = checkpoint['saveables']
        self.value_fn = DMPNNValue(**config, device=args.device)
        self.value_fn.load_state_dict(checkpoint['state'])
        self.value_fn.to(args.device)
        self.value_fn.eval()
        self.value_optim = torch.optim.Adam(self.value_fn.parameters(), lr=args.lr)

    def run(self, mol_node, topk=10, backward=True, sample_mode=None, test=False):
        run_res = self.policy.run(mol_node, 
                                  topk=topk, 
                                  backward=backward, 
                                  sample_mode=sample_mode, 
                                  test=test)
        return run_res
                        
                                  
    def compute_react_values(self, react_smiles):
        react_values_list = []
        for reacts in react_smiles:
            if len(reacts) > 1:
                # Multiple reactant SMILES
                val_batch = self.value_fn.forward_with_smiles(reacts)
                min_val = val_batch.min()
                react_values_list.append(min_val)
            else:
                # Single reactant SMILES
                val_single = self.value_fn.forward_with_smiles(reacts)
                react_values_list.append(val_single.squeeze())
        react_values = torch.stack(react_values_list).to(device=self.args.device).unsqueeze(dim=-1)
        return react_values

    def fit_value(self, rxn_batch):
        prod_smiles = rxn_batch['obs']
        react_smiles = rxn_batch['next_obs']
        rewards = rxn_batch['rewards']

        prod_values = self.value_fn.forward_with_smiles(prod_smiles)
        with torch.no_grad():
            react_values = self.compute_react_values(react_smiles)

            rewards_list = []
            for rwds in rewards:
                rewards_list.append(int(len(rwds) == sum(rwds)))
            rewards_tensor = torch.tensor(rewards_list).to(device=self.args.device).unsqueeze(dim=-1)

            targets = rewards_tensor + 0.98 * (1 - rewards_tensor) * react_values
            targets = torch.clamp(targets, min=0, max=1.0)

        loss = F.mse_loss(prod_values, targets)
        self.value_optim.zero_grad()
        loss.backward()
        self.value_optim.step()

        return {'value_fn/training_loss': loss.item(),
                'value_fn/target': targets.mean().item()}

    def fit_policy(self, rxn_batch):
        for m in self.policy.net.modules():
            m.train()

        prod_smiles = rxn_batch['obs']
        react_smiles = rxn_batch['next_obs']
        rewards = rxn_batch['rewards']
        template_idx = rxn_batch['actions']

        prod_values = self.value_fn.forward_with_smiles(prod_smiles)  # shape: (batch_size, 1)
        with torch.no_grad():
            react_values = self.compute_react_values(react_smiles)

            rewards_list = []
            for rwds in rewards:
                rewards_list.append(int(len(rwds) == sum(rwds)))
            rewards_tensor = torch.tensor(rewards_list).to(device=self.args.device).unsqueeze(dim=-1)

            targets = rewards_tensor + 0.98 * (1 - rewards_tensor) * react_values
            advantage = targets - prod_values
            advantage = advantage.squeeze(dim=-1)  # shape: (batch_size,)
            advantage = torch.clamp(advantage, min=-1, max=1)

            weights = torch.exp(5 * advantage).detach()  # shape: (batch_size,)
            weights = torch.clamp(weights, min=0, max=100).detach()

        # ----------------------------------------------------
        # 2) Forward pass through MLPModel to get predictions
        # ----------------------------------------------------
        fps = [smiles2array(smi, fp_dim=self.fp_dim) for smi in prod_smiles]
        fps = torch.FloatTensor(fps).to(self.args.device)
        action_logits = self.policy.net(fps)

        template_idx = torch.LongTensor(template_idx).to(self.args.device)
        ce_loss_per_example = F.cross_entropy(action_logits, template_idx, reduction='none')
        weighted_loss = ce_loss_per_example * weights
        total_loss = torch.mean(weighted_loss)

        # ----------------------------------------------------
        # 4) Backprop and update
        # ----------------------------------------------------
        self.policy_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.net.parameters(), 10)
        self.policy_optim.step()
    
        # Compute accuracy
        pred = torch.argmax(action_logits, dim=-1)
        accuracy = (pred == template_idx).float().mean().item()
    
        for m in self.policy.net.modules():
            m.eval()
    
        return {
            'policy/training_loss': total_loss.item(),
            'policy/training_accuracy': accuracy,
            'policy/mean_advantage': advantage.mean().item(),
            'policy/mean_weight': weights.mean().item()
        }
    
    
    def get_reactants(self, p_smi, sampling=False, max_steps=5):
        res = self.policy.run(p_smi, topk=5)
        reactants = res['reactants']
        actions = res['templates_idx']
        if sampling:
            idx = random.choice(range(len(reactants)))
            return reactants[idx].split('.'), actions[idx]
        else:
            return reactants[0].split('.'), actions[0]
    
    def canonicalize(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            print('no mol', flush=True)
            return smi
        if mol is None:
            return smi
        mol = Chem.RemoveHs(mol)
        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
        return Chem.MolToSmiles(mol)

    def load_state_dict(self, state_dict):
        self.policy.net.to(device="cpu")
        self.policy.net.load_state_dict(state_dict)
        self.policy.net.to(device=self.args.device)
    
    def state_dict(self):
        return self.policy.net.state_dict()

    def policy_state_dict(self):
        return self.policy.net.state_dict()
    
    def value_state_dict(self):
        return self.value_fn.state_dict()