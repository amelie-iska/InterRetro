import pickle
import logging
import json
import random
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from utils import set_seeds, create_exp, smiles2array
from config_parser import get_args
from models import MLPModel, ActionRepresentation
from data_loader import create_molecule_dataloader

args = get_args()
args.seed = set_seeds(args.seed)
exp_folder = create_exp(args.seed, args.group)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(exp_folder + "/experiment_log.log"),
        logging.StreamHandler()
    ]
)
logging.info(f"Created experiment run directory: {exp_folder}")
with open(exp_folder + "/args.json", "w") as f:
    json.dump(args.__dict__, f, indent=4)


routes = pickle.load(open(args.training_routes, 'rb'))
target_mols = []
for route in tqdm(routes[:100]):
    for rxn in route:
        target_mols.append(Chem.CanonSmiles(rxn.split('>')[0]))
target_mols = list(set(target_mols))
logging.info(f"{len(target_mols)} target molecules are preserved after removing duplicates. ")


one_step = MLPModel(state_path=args.state_path, template_path=args.template_path, device=args.device)
ac_rep = ActionRepresentation(state_dim=len(one_step.idx2rules), ac_dim=64, device=args.device)

molecule_loader = create_molecule_dataloader(target_mols, batch_size=1, shuffle=True)
for batch in molecule_loader:
    mol = smiles2array(batch[0], fp_dim=2048)
    obs = torch.Tensor(mol[np.newaxis]).to(device=args.device)
    _, _, idx_topk = one_step.forward_topk(obs, topk=2)
    random.randint(0, len(one_step.idx2rules))
    
    anchor_index = idx_topk[0, 0].item()
    positive_index = idx_topk[0, 1].item()
    negatives_indices = random.sample(range(len(one_step.idx2rules)), 256)
    
    anchor_index = F.one_hot(torch.tensor([anchor_index]).to(device=args.device), num_classes=len(one_step.idx2rules))
    positive_index = F.one_hot(torch.tensor([positive_index]).to(device=args.device), num_classes=len(one_step.idx2rules))
    negative_indices = F.one_hot(torch.tensor(negatives_indices).to(device=args.device), num_classes=len(one_step.idx2rules))

    anchor_rep = ac_rep(anchor_index.to(torch.float32))
    positive_rep = ac_rep(positive_index.to(torch.float32))
    negative_indices = ac_rep(negative_indices.to(torch.float32))
#constrative_trainer = ContrastiveTrainer(one_step, temperature=0.1, device=args.device)








































