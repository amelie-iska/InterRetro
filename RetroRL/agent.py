import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import copy

from rdkit import Chem

from models import MLPModel, ActionRepresentation, ValueMLP, PriorMLP
from retrorl_utils import smiles2array
from Graph2Edits.models import Graph2Edits, BeamSearch
from Graph2Edits.utils.generate_edits import generate_reaction_edits
from Graph2Edits.utils.rxn_graphs import MolGraph, RxnGraph, Vocab
from Graph2Edits.utils.reaction_actions import (AddGroupAction, AtomEditAction,
                                                BondEditAction, Termination)
from Graph2Edits.utils.mol_features import ATOM_FDIM, BOND_FDIM

from rxnmapper import RXNMapper

from typing import Any, List, Tuple
import sys
from collections import Counter
import joblib
import pandas as pd
from rdkit import Chem
import os

class RetroRL:
    def __init__(self, args):
        self.args = args
        self.updates = 0
        
        self.rxn_mapper = RXNMapper()
        self.bond_vocab = Vocab(joblib.load(self.args.bond_vocab_file))
        self.atom_vocab = Vocab(joblib.load(self.args.atom_vocab_file))
        
        # Load Graph2Edits
        print(args.graph2edits_model)
        checkpoint = torch.load(args.graph2edits_model)
        config = checkpoint['saveables']
        self.graph2edits = Graph2Edits(**config, device=args.device)
        self.graph2edits.load_state_dict(checkpoint['state'])
        self.graph2edits.to(args.device)
        self.graph2edits.eval()
        self.graph2edits_optim = torch.optim.Adam(self.graph2edits.parameters(), lr=args.lr)
        
        # Load bream search
        self.beam_model = BeamSearch(model=self.graph2edits, step_beam_size=10,
                                     beam_size=10, use_rxn_class=False)
    
    def fit(self, rxn_batch):
        
        products_list    = rxn_batch['obs']     # each is a product SMILES
        edits_list       = rxn_batch['actions'] # each is a list of (action, atomlist) pairs
        reactants_list   = rxn_batch['next_obs']
        
        
        batch_graphs = []
        for product_smi, all_edits, children_smi in zip(products_list, edits_list, reactants_list):
            reactant_string = '.'.join(children_smi)
            graph_seq = self.build_graph_seq(product_smi, reactant_string, all_edits)
            if len(graph_seq) > 0:
                batch_graphs.append(graph_seq)
                
        if len(batch_graphs) == 0:
            return {}
        
        graph_seq_tensors, seq_labels, seq_mask = self.process_batch(batch_graphs)
        
        seq_edit_scores = self.graph2edits(graph_seq_tensors)
        max_seq_len, batch_size = seq_mask.size()
    
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        seq_loss = []
    
        for step_idx in range(max_seq_len):
            target_labels = self.graph2edits.to_device(seq_labels[step_idx])
            # Sum the cross-entropy over the batch, but only for active steps
            loss_batch = []
            for i in range(batch_size):
                if seq_mask[step_idx][i] == 1:  # valid step
                    pred_logits  = seq_edit_scores[step_idx][i].unsqueeze(0)
                    gold_label   = torch.argmax(target_labels[i]).unsqueeze(0).long()
                    loss_batch.append(loss_fn(pred_logits, gold_label).sum())
            if len(loss_batch) > 0:
                seq_loss.append(torch.stack(loss_batch).mean())
    
        if len(seq_loss) == 0:
            return {}
    
        total_loss = torch.stack(seq_loss).mean()
    
        accuracy = self.get_seq_edit_accuracy(seq_edit_scores, seq_labels, seq_mask)
    
        self.graph2edits_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.graph2edits.parameters(), 10)
        self.graph2edits_optim.step()
    
        logs = {
           'graph2edits/training_loss': total_loss.item(),
           'graph2edits/training_accuracy': accuracy
        }
        return logs
        
        
        
        
    def build_graph_seq(self, product_smi, reactants_smi, edit_seq):
        """
        product_smi:    e.g. 'CCC(=O)N'
        reactants_smi:  e.g. 'CC.CN' or 'CC.O' or single 'CC'
        edit_seq:       a list of ((actionType, actionParams), [atomMap(s)]) steps.
                        e.g. [ (('Delete Bond',(None,None)), [1,2]),
                               (('Change Atom',(2,0)), [3]),
                               ('Terminate', None) ]
        Returns: a list of RxnGraph objects, each describing one step in the transformation.
        """
        try:
            r_mol = Chem.MolFromSmiles(reactants_smi)
            p_mol = Chem.MolFromSmiles(product_smi)
            Chem.Kekulize(p_mol)
        except:
            return []  # can't parse?
    
        graph_seq = []
        int_mol = p_mol
        for (edit_action, edit_atoms) in edit_seq:
            # Create the RxnGraph for this step
            rxn_graph = RxnGraph(
                prod_mol=Chem.Mol(int_mol),
                edit_to_apply=edit_action,
                edit_atom=edit_atoms,
                reac_mol=Chem.Mol(r_mol),
                rxn_class=None, 
                use_rxn_class=False
            )
            graph_seq.append(rxn_graph)
    
            # If not 'Terminate', apply the edit to int_mol
            if edit_action == 'Terminate':
                # after Termination, we just break or let it continue with a None check
                pass
            else:
                int_mol = self.apply_edit_to_mol(
                    Chem.Mol(int_mol), edit_action, edit_atoms
                )
                if int_mol is None:
                    print("Warning: applying edit resulted in None molecule. Stopping.")
                    break
    
        return graph_seq

        
        
        
        
        
        
       
    
    def get_seq_edit_accuracy(self, seq_edit_scores, seq_labels, seq_mask):
        max_seq_len = seq_mask.shape[0]
        batch_size = seq_mask.shape[1]
        assert len(seq_edit_scores) == max_seq_len
        assert len(seq_labels) == max_seq_len
        assert len(seq_edit_scores[0]) == batch_size
        lengths = seq_mask.sum(dim=0).flatten()
    
        def check_equals(x, y): return torch.argmax(x) == torch.argmax(y)
    
        all_acc = 0
        for batch_id in range(batch_size):
            step_acc = 0
            seq_length = lengths[batch_id]
            for idx in range(seq_length):
                if check_equals(seq_edit_scores[idx][batch_id], seq_labels[idx][batch_id]):
                    step_acc += 1
    
            if step_acc == seq_length:
                all_acc += 1
    
        accuracy = all_acc / batch_size
        return accuracy
    
    def process_batch(self, batch_graphs):
        lengths = torch.tensor([len(graph_seq)
                               for graph_seq in batch_graphs], dtype=torch.long)
        max_length = max([len(graph_seq) for graph_seq in batch_graphs])
    
    
        graph_seq_tensors = []
        edit_seq_labels = []
        seq_mask = []
    
        for idx in range(max_length):
            graphs_idx = [copy.deepcopy(batch_graphs[i][min(idx, length-1)]).get_components(attrs=['prod_graph', 'edit_to_apply', 'edit_atom'])
                          for i, length in enumerate(lengths)]
            mask = (idx < lengths).long()
            prod_graphs, edits, edit_atoms = list(zip(*graphs_idx))
            assert all([isinstance(graph, MolGraph) for graph in prod_graphs])
    
            edit_labels = self.prepare_edit_labels(
                prod_graphs, edits, edit_atoms, self.bond_vocab, self.atom_vocab)
            current_graph_tensors = self.get_batch_graphs(
                prod_graphs, use_rxn_class=None)
    
            graph_seq_tensors.append(current_graph_tensors)
            edit_seq_labels.append(edit_labels)
            seq_mask.append(mask)
    
        seq_mask = torch.stack(seq_mask).long()
        assert seq_mask.shape[0] == max_length
        assert seq_mask.shape[1] == len(batch_graphs)
    
        return graph_seq_tensors, edit_seq_labels, seq_mask
    
    def get_batch_graphs(self, graph_batch: List[MolGraph], use_rxn_class: bool = False) -> Tuple[torch.Tensor, List[Tuple[int]]]:
        """
        Featurization of a batch of molecules.
        """
        # Start n_atoms and n_bonds at 1 b/c zero padding
        n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule
    
        # All start with zero padding so that indexing with zero padding returns zeros
        atom_fdim = ATOM_FDIM
        bond_fdim = atom_fdim + BOND_FDIM
    
        f_atoms = [[0] * atom_fdim]  # atom features
        f_bonds = [[0] * bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        undirected_b2a = [[]] # mapping from the undirected bond index to the beginindex and endindex of the atoms
        
        for mol_graph in graph_batch:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
    
            for a in range(mol_graph.n_atoms):
                a2b.append([b + n_bonds for b in mol_graph.a2b[a]])
    
            for b in range(mol_graph.n_bonds):
                b2a.append(n_atoms + mol_graph.b2a[b])
                b2revb.append(n_bonds + mol_graph.b2revb[b])
    
            n_undirected_bonds = len(undirected_b2a)
            for bond in mol_graph.mol.GetBonds():
                undirected_b2a.append(sorted([bond.GetBeginAtomIdx() + n_atoms, bond.GetEndAtomIdx() + n_atoms]))
    
            a_scope.append((n_atoms, mol_graph.n_atoms))
            b_scope.append((n_undirected_bonds, mol_graph.num_bonds))
            n_atoms += mol_graph.n_atoms
            n_bonds += mol_graph.n_bonds
    
        f_atoms = torch.FloatTensor(f_atoms)
        f_bonds = torch.FloatTensor(f_bonds)
        a2b = self.create_pad_tensor(a2b)
        b2a = torch.LongTensor(b2a)
        b2revb = torch.LongTensor(b2revb)
        undirected_b2a = self.create_pad_tensor(undirected_b2a)
    
        graph_tensors = (f_atoms, f_bonds, a2b, b2a, b2revb, undirected_b2a)
        scopes = (a_scope, b_scope)
    
        return graph_tensors, scopes
    
    def create_pad_tensor(self, alist):
        max_len = max([len(a) for a in alist])
        for a in alist:
            pad_len = max_len - len(a)
            a.extend([0] * pad_len)
        return torch.tensor(alist, dtype=torch.long)
    
    def prepare_edit_labels(self, graph_batch: List[MolGraph], edits: List[Any], edit_atoms: List[Any], bond_vocab: List, atom_vocab: List) -> torch.tensor:
        """ 
        Prepare edit label including atom edits and bond edits.
        """
        bond_vocab_size = self.bond_vocab.size()
        atom_vocab_size = self.atom_vocab.size()
        edit_labels = []
    
        for prod_graph, edit, edit_atom in zip(graph_batch, edits, edit_atoms):
            bond_label = np.zeros((prod_graph.num_bonds, bond_vocab_size))
            atom_label = np.zeros((prod_graph.num_atoms, atom_vocab_size))
            stop_label = np.zeros((1,))
    
            if edit == 'Terminate':
                stop_label[0] = 1.0
    
            elif edit[0] == 'Change Atom' or edit[0] == 'Attaching LG':
                a_map = edit_atom
                a_idx = prod_graph.amap_to_idx[a_map]
                edit_idx = self.atom_vocab.get_index(edit)
                atom_label[a_idx][edit_idx] = 1
    
            else:
                a1, a2 = edit_atom[0], edit_atom[1]
                a_start, a_end = prod_graph.amap_to_idx[a1], prod_graph.amap_to_idx[a2]
                b_idx = prod_graph.mol.GetBondBetweenAtoms(a_start, a_end).GetIdx()
                edit_idx = self.bond_vocab.get_index(edit)
                bond_label[b_idx][edit_idx] = 1
    
            edit_label = np.concatenate(
                (bond_label.flatten(), atom_label.flatten(), stop_label.flatten()))
            edit_label = torch.from_numpy(edit_label)
            edit_labels.append(edit_label)
    
        return edit_labels
    
    def apply_edit_to_mol(self, mol: Chem.Mol, edit: Tuple, edit_atom: Any) -> Chem.Mol:
        """ Apply edits to molecular graph """
    
        if edit[0] == 'Change Atom':
            edit_exe = AtomEditAction(
                edit_atom, *edit[1], action_vocab='Change Atom')
            new_mol = edit_exe.apply(mol)
    
        if edit[0] == 'Delete Bond':
            edit_exe = BondEditAction(
                *edit_atom, *edit[1], action_vocab='Delete Bond')
            new_mol = edit_exe.apply(mol)
    
        if edit[0] == 'Change Bond':
            edit_exe = BondEditAction(
                *edit_atom, *edit[1], action_vocab='Change Bond')
            new_mol = edit_exe.apply(mol)
    
        if edit[0] == 'Add Bond':
            edit_exe = BondEditAction(
                *edit_atom, *edit[1], action_vocab='Add Bond')
            new_mol = edit_exe.apply(mol)
    
        if edit[0] == 'Attaching LG':
            edit_exe = AddGroupAction(
                edit_atom, edit[1], action_vocab='Attaching LG')
            new_mol = edit_exe.apply(mol)
    
        return new_mol
    
    def check_edits(self, edits: List):
        for edit in edits:
            if edit[0] == 'Add Bond':
                return False
        return True
    
    def preprocessing(self, rxns):
        rxns_data = []
        counter = []
        all_edits = {}
    
        for idx, rxn_smi in enumerate(rxns):
            r, p = rxn_smi.split('>>')
            prod_mol = Chem.MolFromSmiles(p)
    
            if (prod_mol is None) or (prod_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):
                print(
                    f'Product has 0 or 1 atom or 1 bond, Skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue
    
            react_mol = Chem.MolFromSmiles(r)
    
            if (react_mol is None) or (react_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):
                print(
                    f'Reactant has 0 or 1 atom or 1 bond, Skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue
    
            try:
                rxn_data = generate_reaction_edits(rxn_smi, kekulize=True)
            except:
                print(f'Failed to extract reaction data, skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue
    
            edits_accepted = self.check_edits(rxn_data.edits)
            if not edits_accepted:
                print(f'Edit: Add new bond. Skipping reaction {idx}')
                print()
                sys.stdout.flush()
                continue
    
            rxns_data.append(rxn_data)
    
        sys.stdout.flush()
    
        for idx, rxn_data in enumerate(rxns_data):
            for edit in rxn_data.edits:
                if edit not in all_edits:
                    all_edits[edit] = 1
                else:
                    all_edits[edit] += 1

        atom_edits = []
        bond_edits = []
        lg_edits = []
        atom_lg_edits = []

        for edit, num in all_edits.items():
            if edit[0] == 'Change Atom':
                atom_edits.append(edit)
                atom_lg_edits.append(edit)
            elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':
                bond_edits.append(edit)
            elif edit[0] == 'Attaching LG':
                lg_edits.append(edit)
        atom_lg_edits.extend(lg_edits)

        filter_rxns_data = []
        for idx, rxn_data in enumerate(rxns_data):
            for edit in rxn_data.edits:
                if edit[0] == 'Attaching LG' and edit not in lg_edits:
                    print(
                        f'The number of {edit} in training set is very small, skipping reaction')
                    rxn_data = None
            if rxn_data is not None:
                counter.append(len(rxn_data.edits))
                filter_rxns_data.append(rxn_data)

        return filter_rxns_data
    
    
    def get_reactants(self, p_smi, sampling=False):
        p_smi = self.canonicalize(p_smi)
        mol = Chem.MolFromSmiles(p_smi)
        for i, atom in enumerate(mol.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
        labeled_smi = Chem.MolToSmiles(mol)
        top_k_results = self.beam_model.run_search(prod_smi=labeled_smi, max_steps=9, rxn_class=False)
        
        all_reactants = []
        all_edits = []
        for res in top_k_results:
            if not res['final_smi'].startswith('final_smi_unmapped'):
                reactant_list = res['final_smi'].split('.')
                all_reactants.append(reactant_list)
                
                edit_seq_with_params = []
                for action, atom_or_atoms in zip(res['edits'], res['edits_atom']):
                    edit_seq_with_params.append((action, atom_or_atoms))
                edit_seq_with_params.append(('Terminate', None))
                all_edits.append(edit_seq_with_params)
        
        if len(all_reactants) == 0:
            raise ValueError(f"Final SMILES for {p_smi} are unmapped.")
        
        # If sampling == True, pick one random route
        if sampling:
            idx = random.choice(range(len(all_reactants)))
            return all_reactants[idx], all_edits[idx]
        else:
            return all_reactants[0], all_edits[0]
    
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

    
    