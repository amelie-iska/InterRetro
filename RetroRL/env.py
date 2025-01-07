
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
        obs_mol = Chem.MolFromSmiles(self.obs)
        for i, atom in enumerate(obs_mol.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
        mapped_smi = Chem.MolToSmiles(obs_mol)
        self.obs_node = AnyNode(id=self.obs, action=None, inbb=int(self.obs in self.building_blocks),
                                mapping=mapped_smi)
        self.tree_mdp = self.obs_node
        return self.obs

    def step(self, reactants, edits):
        if len(reactants) == 0:
            return None, 0, 1, None
        
        self.obs_node.action = edits
        for r in reactants:
            unmapped_r = self.canonicalize(r)
            inbb = unmapped_r in self.building_blocks
            r_node = AnyNode(id=unmapped_r, parent=self.obs_node, action=None, inbb=int(inbb),
                             mapping=r)
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
    
    def build_tree(self, agent, mol, test=False):
        obs = self.reset(mol)
        for t in range(1, 16):
            if test:
                reactants, edits = agent.get_reactants(obs, sampling=False)
            else:
                reactants, edits = agent.get_reactants(obs, sampling=True)
            next_obs, reward, done, info = self.step(reactants, edits)
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
