
import random
import numpy as np
import logging

import torch
from tqdm import tqdm
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from anytree import AnyNode, RenderTree, PreOrderIter
from rdchiral.main import rdchiralRunText

from retrorl_utils import smiles2array


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

    def step(self, reactants):
        self.obs_node.action = None
       
        if len(reactants) == 0:
            return None, 0, 1, None
        
        for r in reactants:
            inbb = r in self.building_blocks
            r_node = AnyNode(id=r, parent=self.obs_node, action=None, 
                             inbb=int(inbb), value=-1)
            if inbb == False:
                self.tobevisited_nodes.append(r_node)
        
        if len(self.tobevisited_nodes) > 0:
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
        for mol_idx in tqdm(range(100)):
            obs = self.reset()
            print(obs)
            for t in range(20):
                reactants = agent.get_reactants(obs)
                next_obs, reward, done, info = self.step(reactants)
                rewards[mol_idx] = reward
                if done: break
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