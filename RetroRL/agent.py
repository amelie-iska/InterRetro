import os
import torch
import torch.nn.functional as F
import numpy as np
import random

from rdkit import Chem

from models import MLPModel, ActionRepresentation, ValueMLP, PriorMLP
from retrorl_utils import smiles2array
from Graph2Edits.models import Graph2Edits, BeamSearch
from Graph2Edits.utils.generate_edits import generate_reaction_edits

from rxnmapper import RXNMapper

from typing import Any, List
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
        
        # Load Graph2Edits
        print(args.graph2edits_model)
        checkpoint = torch.load(args.graph2edits_model)
        config = checkpoint['saveables']
        self.graph2edits = Graph2Edits(**config, device=args.device)
        self.graph2edits.load_state_dict(checkpoint['state'])
        self.graph2edits.to(args.device)
        self.graph2edits.eval()
        
        # Load bream search
        self.beam_model = BeamSearch(model=self.graph2edits, step_beam_size=10,
                                     beam_size=10, use_rxn_class=False)
    
    def fit(self, rxn_batch):
        obs = rxn_batch['obs']
        next_obs = rxn_batch['next_obs']
        rewards = rxn_batch['rewards']
        dones = rxn_batch['dones']
        
        # Automic atom mapping
        print('obs', obs)
        print('next_obs', next_obs)
        reactions = []
        for product, reactants in zip(obs, next_obs):
            reactant_string = '.'.join(reactants)
            reaction = f"{reactant_string}>>{product}"
            reactions.append(reaction)
        print(reactions)
        mapped_reactions = self.rxn_mapper.get_attention_guided_atom_maps(reactions)
        print(mapped_reactions)
        
        mapped_reactions = [rxn['mapped_rxn'] for rxn in mapped_reactions]
        print('\nmapped_reactions', mapped_reactions)
        filter_rxns_data = self.preprocessing(rxns=mapped_reactions)
        print('\nfilter_rxns_data', filter_rxns_data)
        
        
    
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
        for i, atom in enumerate(mol.GetAtoms(), start=1):  # Start numbering from 1
            atom.SetAtomMapNum(i)
        labeled_smi = Chem.MolToSmiles(mol)
        # top_k_results = self.beam_model.run_search(prod_smi=labeled_smi, max_steps=9, rxn_class=False)
        top_k_results = self.beam_model.run_search(prod_smi=labeled_smi, max_steps=9, rxn_class=False)
        
        if sampling:
            all_reactants = []
            for res in top_k_results:
                if not res['final_smi'].startswith('final_smi_unmapped'):       
                    reactants = list(res['final_smi'].split('.'))
                    all_reactants.append(reactants)
            if len(all_reactants) == 0:
                raise ValueError('All final SMILES are invalid.')
            return random.sample(all_reactants, 1)[0]     
        else:
            reactants = None
            for res in top_k_results:
                if not res['final_smi'].startswith('final_smi_unmapped'):       
                    reactants = list(res['final_smi'].split('.'))
                    break
            if reactants == None:
                raise ValueError('All final SMILES are invalid.')
            return reactants
    
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

    
    