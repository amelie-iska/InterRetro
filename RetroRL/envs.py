
import random
import numpy as np
import logging

import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from anytree import AnyNode, RenderTree, PreOrderIter
from rdchiral.main import rdchiralRunText

from utils import smiles2array


class RetroEnv:
    def __init__(self, target_mols, blocking_blocks):
        self.target_mols = target_mols
        self.building_blocks = blocking_blocks

    def reset(self, mol_idx=None):
        self.tree_mdp = None
        self.tobevisited_nodes = []

        if mol_idx == None:
            self.obs = random.choice(self.target_mols)
        else:
            self.obs = self.target_mols[mol_idx]
            
        self.obs_node = AnyNode(id=self.obs, action=None, inbb=int(self.obs in self.building_blocks),
                                value=-1)
        self.tree_mdp = self.obs_node
        return self.obs

    def step(self, a):
        self.obs_node.action = a
        try:
            reactants = rdchiralRunText(a, self.obs)
        except RuntimeError as e:
            logging.info(f"rdchiralRunText error: {e}")
            return None, 0, 1, None
        
        if len(reactants) == 0:
            return None, 0, 1, None
        else:
            reactants = list(set(reactants[0].split('.')))
        
        for r in reactants:
            r_node = AnyNode(id=r, parent=self.obs_node, action=None, inbb=int(r in self.building_blocks),
                             value=-1)
            if not r in self.building_blocks:
                self.tobevisited_nodes.append(r_node)
        
        if self.tobevisited_nodes:
            self.obs = self.tobevisited_nodes[-1].id
            self.obs_node = self.tobevisited_nodes[-1]
            self.tobevisited_nodes.remove(self.tobevisited_nodes[-1])
            return self.obs, 0, 0, None
        else:
            self.obs = None
            self.obs_node = None
            return None, 1, 1, None
    
    @torch.no_grad()
    def eval(self, agent, mode, critic=None, print_last_tree=True):
        eval_info = {}
        rewards = np.zeros((100, 1))
        for mol_idx in range(100):
            obs = self.reset()
            for t in range(50):
                a = agent.get_action(obs)
                
                if not critic == None and mol_idx == 99:
                    obs_arr = smiles2array(obs, agent.fp_dim)
                    obs_arr = np.reshape(obs_arr,[-1, obs_arr.shape[0]])
                    obs_arr = torch.tensor(obs_arr, dtype=torch.float32).to(device='cuda')
                    
                    rxn_idx = agent.one_step.rules2idx[a]
                    rxn_arr = np.zeros(len(agent.one_step.idx2rules))
                    rxn_arr[rxn_idx] = 1
                    rxn_arr = torch.tensor(rxn_arr[np.newaxis], dtype=torch.float32).to(device='cuda')
                    
                    value = critic(obs_arr, rxn_arr).item()
                    self.obs_node.value = value
                 
                next_obs, reward, done, info = self.step(a)
                rewards[mol_idx] = reward
                if done:
                    break
                obs = next_obs
        eval_info['evaluation/' + mode + '_success_rate'] = rewards.mean()
        if print_last_tree and not (critic==None):
            self.print_tree()
        return eval_info
    
    def build_tree(self, agent, mol_idx=None, epsilon=0.1):
        obs = self.reset(mol_idx)
        for t in range(50):
            if random.random() < epsilon:
                a = agent.get_action(obs, exploration=True)
            else:
                a = agent.get_action(obs)
            next_obs, reward, done, info = self.step(a)
            if done:
                break
            obs = next_obs
        return self.tree_mdp, int(reward==1)
    
    def print_tree(self, root=None):
        if root == None:
            root = self.tree_mdp
        for pre, fill, node in RenderTree(root):
            in_bb = 'BB,' if node.inbb == 1 else ''          
            print(f"{pre}{node.id}----{in_bb} {node.value:.2f}") 
