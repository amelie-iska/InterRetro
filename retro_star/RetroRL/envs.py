
import random
import numpy as np

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
            
        self.obs_node = AnyNode(id=self.obs, action=None, inbb=int(self.obs in self.building_blocks))
        self.tree_mdp = self.obs_node
        return self.obs

    def step(self, a):
        self.obs_node.action = a
        reactants = rdchiralRunText(a, self.obs)
        if len(reactants) == 0:
            return None, 0, 1, None
        else:
            reactants = list(set(reactants[0].split('.')))
        
        for r in reactants:
            r_node = AnyNode(id=r, parent=self.obs_node, action=None, inbb=int(r in self.building_blocks))
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
    
    def eval(self, agent, mode):
        eval_info = {}
        rewards = np.zeros((min(len(self.target_mols), 200), 1))
        for mol_idx in range(min(len(self.target_mols), 200)):
            obs = self.reset(mol_idx)
            for t in range(100):
                a, _ = agent.get_action(obs)
                next_obs, reward, done, info = self.step(a)
                rewards[mol_idx] = reward
                if done:
                    break
                obs = next_obs
        eval_info['evaluation/' + mode + '_success_rate'] = rewards.mean()
        return eval_info
    
    def build_tree(self, agent, mol_idx=None, exploration_rate=0.1):
        obs = self.reset(mol_idx)
        for t in range(100):
            if random.random() < exploration_rate:
                _, a = agent.get_random_action(obs)
            else:
                a, _ = agent.get_action(obs)
            next_obs, reward, done, info = self.step(a)
            if done:
                break
            obs = next_obs
        return self.tree_mdp
