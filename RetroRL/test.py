
import os
import sys
import torch

from rdkit import Chem

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from Graph2Edits.utils.rxn_graphs import MolGraph
from Graph2Edits.utils.collate_fn import get_batch_graphs
from Graph2Edits.models.encoder import MPNEncoder

from models import DMPNNValue
from config_parser import get_args

if __name__ == "__main__":
    
    args = get_args()
    
    smiles_list = ["CCO", "CCC", "O=C(O)Cc1ccccc1"]  
    product_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    
    for mol in product_mols:
        Chem.Kekulize(mol)
    
    prod_graphs = [
        MolGraph(mol=Chem.Mol(mol), rxn_class=None, use_rxn_class=False)
        for mol in product_mols
    ]
    prod_tensors, prod_scopes = get_batch_graphs(prod_graphs, use_rxn_class=False)
    
    checkpoint = torch.load(args.graph2edits_model, map_location=torch.device('cpu'), weights_only=False)
    config = checkpoint['saveables']
    value_fn = DMPNNValue(**config, device=args.device)
    value_fn.load_state_dict(checkpoint['state'])
    value_fn.eval()
    value_fn.to(args.device)
    
    
    a_feats = value_fn(prod_tensors, prod_scopes)
    
    