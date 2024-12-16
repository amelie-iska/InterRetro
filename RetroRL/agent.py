import os
import torch
import torch.nn.functional as F
import numpy as np

from rdkit import Chem

from models import MLPModel, ActionRepresentation, ValueMLP, PriorMLP
from retrorl_utils import smiles2array
from Graph2Edits.models import Graph2Edits, BeamSearch

class RetroRL:
    def __init__(self, args):
        self.args = args
        self.updates = 0
        
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
    
    def get_reactants(self, p_smi):
        mol = Chem.MolFromSmiles(p_smi)
        for i, atom in enumerate(mol.GetAtoms(), start=1):  # Start numbering from 1
            atom.SetAtomMapNum(i)
        labeled_smi = Chem.MolToSmiles(mol)
        top_k_results = self.beam_model.run_search(prod_smi=labeled_smi, max_steps=9, rxn_class=False)
        
        reactants = None
        for res in top_k_results:
            if not res['final_smi'].startswith('final_smi_unmapped'):       
                reactants = list(res['final_smi'].split('.'))
                break
        if reactants == None:
            raise ValueError('All final SMILES are invalid.')
        return reactants

    
    