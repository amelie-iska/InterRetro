
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
    def __init__(self, blocking_blocks):
        self.building_blocks = blocking_blocks

    def reset(self, mol):
        self.tree_mdp = None
        self.tobevisited_nodes = []

        self.obs = mol
        self.obs_node = AnyNode(id=self.obs, action=None, inbb=int(self.obs in self.building_blocks))
        self.tree_mdp = self.obs_node
        return self.obs

    def step(self, reactants):
        self.obs_node.action = None
       
        if len(reactants) == 0:
            return None, 0, 1, None
        
        for r in reactants:
            inbb = r in self.building_blocks
            r_node = AnyNode(id=r, parent=self.obs_node, action=None, inbb=int(inbb))
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
    def eval(self, agent, mol):
        eval_info = {}
        rewards = np.zeros((20, 1))
        for mol_idx in tqdm(range(20)):
            obs = self.reset(mol)
            for t in range(20):
                reactants = agent.get_reactants(obs)
                next_obs, reward, done, info = self.step(reactants)
                rewards[mol_idx] = reward
                if done: 
                    break
                obs = next_obs
        eval_info['evaluation/success_rate'] = rewards.mean()
        return eval_info
    
    def build_tree(self, agent, mol):
        obs = self.reset(mol)
        for t in range(1, 16):
            reactants = agent.get_reactants(obs, sampling=True)
            next_obs, reward, done, info = self.step(reactants)
            if done:
                break
            obs = next_obs
            
        # Largest successful subtrees: successful nodes whose parent is not successful
        successful_subtrees = []
        for node in PreOrderIter(self.tree_mdp):
            if self.is_successful_subtree(node):
                if node.is_root or not self.is_successful_subtree(node.parent):
                    if node.inbb == False:
                        successful_subtrees.append(node)

        return self.tree_mdp, successful_subtrees, \
                {'success': int(reward==1), 'model_calls': t}
    
    
    def is_successful_subtree(self, node):
        # A subtree is successful if all its leaves are building blocks
        return all(leaf.inbb == 1 for leaf in node.leaves)
    
    
    def is_failed_subtree(self, node):
        # A subtree is failing if it has at least one leaf not a building block
        return not all(leaf.inbb == 1 for leaf in node.leaves)
    
    def print_tree(self, root=None):
        if root == None:
            root = self.tree_mdp
        for pre, fill, node in RenderTree(root):
            in_bb = '----BB' if node.inbb == 1 else ''          
            print(f"{pre}{node.id}{in_bb}") 