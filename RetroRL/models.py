from typing import Dict, List, Tuple, Union

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from Graph2Edits.prepare_data import apply_edit_to_mol
from Graph2Edits.utils.collate_fn import get_batch_graphs
from Graph2Edits.utils.rxn_graphs import MolGraph, Vocab

from Graph2Edits.models.encoder import Global_Attention, MPNEncoder
from Graph2Edits.models.model_utils import (creat_edits_feats, index_select_ND,
                                            unbatch_feats)


class DMPNNValue(nn.Module):
    def __init__(self,
                 config: Dict,
                 atom_vocab: Vocab,
                 bond_vocab: Vocab,
                 device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        config: Dict, Model arguments
        atom_vocab: atom and LG edit labels
        bond_vocab: bond edit labels
        device: str, Device to run the model on.
        """
        super(DMPNNValue, self).__init__()

        self.config = config
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.atom_outdim = len(atom_vocab)
        self.bond_outdim = len(bond_vocab)
        self.device = device

        self._build_layers()

    def _build_layers(self) -> None:
        """Builds the different layers associated with the model."""
        config = self.config
        self.encoder = MPNEncoder(atom_fdim=config['n_atom_feat'],
                                  bond_fdim=config['n_bond_feat'],
                                  hidden_size=config['mpn_size'],
                                  depth=config['depth'],
                                  dropout=config['dropout_mpn'],
                                  atom_message=config['atom_message'])

        self.W_vv = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)
        nn.init.eye_(self.W_vv.weight)
        self.W_vc = nn.Linear(config['mpn_size'],
                              config['mpn_size'], bias=False)

        if config['use_attn']:
            self.attn = Global_Attention(
                d_model=config['mpn_size'], heads=config['n_heads'])

        self.atom_linear = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.atom_outdim))
        self.bond_linear = nn.Sequential(
            nn.Linear(config['mpn_size'] * 2, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.bond_outdim))

        self.graph_linear = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], 1))

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            tensors = [tensor.to(self.device, non_blocking=True)
                       for tensor in tensors]
            return tensors
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(self.device, non_blocking=True)
        else:
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")
            
            
    def forward(self, prod_tensors, prod_scopes):
        
        prod_tensors = self.to_device(prod_tensors)
        atom_scope, bond_scope = prod_scopes
        
        a_feats = self.encoder(prod_tensors, mask=None)
        atom_feats = F.relu(self.W_vc(a_feats))
        graph_vecs = torch.stack(
            [atom_feats[st: st + le].sum(dim=0) for st, le in atom_scope])
        values = self.graph_linear(graph_vecs)
        return values

    def forward_with_smiles(self, smiles_list):
        product_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        
        for mol in product_mols:
            Chem.Kekulize(mol)
        
        prod_graphs = [
            MolGraph(mol=Chem.Mol(mol), rxn_class=None, use_rxn_class=False)
            for mol in product_mols
        ]
        prod_tensors, prod_scopes = get_batch_graphs(prod_graphs, use_rxn_class=False)
        values = self.forward(prod_tensors, prod_scopes)
        return values

    def get_saveables(self) -> Dict:
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}
        saveables['config'] = self.config
        saveables['atom_vocab'] = self.atom_vocab
        saveables['bond_vocab'] = self.bond_vocab

        return saveables