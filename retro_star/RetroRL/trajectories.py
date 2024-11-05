
import numpy as np

from utils import smiles2array
from anytree import AnyNode, RenderTree, PreOrderIter

class TrajectoryCreator:
    def __init__(self, agent):
        self.agent = agent
    
    def print_tree(self, root):
        for pre, fill, node in RenderTree(root):
            print("%s%s----%s" % (pre, node.id, node.inbb))
    
    def tree2traj(self, root):
        valid_subtrees = []
        self.find_valid_subtrees(root, valid_subtrees)
        valid_subtrees = self.remove_joint_subtrees(root, valid_subtrees)

        traj = []
        for root in valid_subtrees:
            traj += self.get_paths_from_tree(root)
        return traj
    
    def is_node_valid(self, node):
        # Check if the node is valid (all leaf nodes have inbb == 1)
        if not node.children:
            return node.inbb == 1
        return all(self.is_node_valid(child) for child in node.children)

    def find_valid_subtrees(self, node, valid_subtrees):
        if node is None:
            return

        # Check if the current node is valid and has children
        if self.is_node_valid(node) and node.children:
            valid_subtrees.append(node)

        # Recursively check each child
        for child in node.children:
            self.find_valid_subtrees(child, valid_subtrees)

    def is_ancestor(self, ancestor, node):
        current = node
        while current is not None:
            if current == ancestor:
                return True
            current = current.parent
        return False

    def remove_joint_subtrees(self, root, valid_subtrees):
        # Create a set to track nodes to remove
        to_remove = set()
        
        # Check each node against all other nodes
        for node in valid_subtrees:
            for other_node in valid_subtrees:
                if node != other_node and self.is_ancestor(node, other_node):
                    to_remove.add(other_node)
        
        # Remove nodes that are descendants of any node in the list
        filtered_nodes = [node for node in valid_subtrees if node not in to_remove]
        return filtered_nodes

    def get_paths_from_tree(self, node, path=None, reward=0):
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
            all_paths += self.get_paths_from_tree(child, path.copy(), reward=child.inbb)
        return all_paths

    def translate_trajs(self, trajs):
        for traj in trajs:
            for i in range(len(traj)-1):
                if i%2 == 0:
                    traj[i] = smiles2array(traj[i], self.agent.fp_dim)
                else:
                    rxn_idx = self.agent.one_step.rules2idx[traj[i]]
                    traj[i] = np.zeros(len(self.agent.one_step.idx2rules))
                    traj[i][rxn_idx] = 1
            traj.pop()
        return trajs
