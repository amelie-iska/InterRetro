
import random
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from anytree import AnyNode, RenderTree, PreOrderIter
from rdchiral.main import rdchiralRunText

def eval(envs, agent):
    eval_info = {}
    for env_id, env in enumerate(envs):
        rewards = np.zeros((min(len(env.target_mols), 200), 1))
        for mol_idx in range(min(len(env.target_mols), 200)):
            obs = env.reset(mol_idx)
            for t in range(100):
                a, _ = agent.get_action(obs)
                next_obs, reward, done, info = env.step(a)
                rewards[mol_idx] = reward
                if done:
                    break
                obs = next_obs
        eval_info[f'env_{env_id}/success_rate'] = rewards.mean()
    return eval_info


def collect_trajs(env, agent):
    obs = env.reset()
    for t in range(100):
        a, _ = agent.get_action(obs)
        next_obs, reward, done, info = env.step(a)
        if done:
            break
        obs = next_obs

    valid_subtrees = []
    find_valid_subtrees(env.tree_mdp, valid_subtrees)
    valid_subtrees = remove_joint_subtrees(env.tree_mdp, valid_subtrees)

    traj = []
    for root in valid_subtrees:
        traj += get_paths_from_tree(root)
    return traj

def is_node_valid(node):
    # Check if the node is valid (all leaf nodes have inbb == 1)
    if not node.children:
        return node.inbb == 1
    return all(is_node_valid(child) for child in node.children)

def find_valid_subtrees(node, valid_subtrees):
    if node is None:
        return

    # Check if the current node is valid and has children
    if is_node_valid(node) and node.children:
        valid_subtrees.append(node)

    # Recursively check each child
    for child in node.children:
        find_valid_subtrees(child, valid_subtrees)

def is_ancestor(ancestor, node):
    current = node
    while current is not None:
        if current == ancestor:
            return True
        current = current.parent
    return False

def remove_joint_subtrees(root, valid_subtrees):
    # Create a set to track nodes to remove
    to_remove = set()
    
    # Check each node against all other nodes
    for node in valid_subtrees:
        for other_node in valid_subtrees:
            if node != other_node and is_ancestor(node, other_node):
                to_remove.add(other_node)
    
    # Remove nodes that are descendants of any node in the list
    filtered_nodes = [node for node in valid_subtrees if node not in to_remove]
    return filtered_nodes

def get_paths_from_tree(node, path=None, reward=0):
    if path is None:
        path = []  
    path.append(node.id)
    path.append(node.action)
    
    if not node.children:
        if reward == 1:
            return [list(path)]
        return [] 

    all_paths = []
    for child in node.children:
        all_paths += get_paths_from_tree(child, path.copy(), reward=child.inbb)
    return all_paths


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