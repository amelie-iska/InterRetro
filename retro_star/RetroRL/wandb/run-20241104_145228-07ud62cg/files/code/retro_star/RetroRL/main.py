import pickle
import logging
import json
import random
import time
import wandb
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdchiral.main import rdchiralRunText

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
if args.enable_wandb:
    wandb.init(project=args.project, config=args, group=args.group, name='{}_seed{}'.format(args.group, args.seed))

routes = pickle.load(open(args.training_routes, 'rb'))
target_mols = []
for route in tqdm(routes[:2000]):
    for rxn in route:
        target_mols.append(Chem.CanonSmiles(rxn.split('>')[0]))
target_mols = list(set(target_mols))
logging.info(f"{len(target_mols)} target molecules are preserved after removing duplicates. ")


one_step = MLPModel(state_path=args.state_path, template_path=args.template_path, device=args.device)
ac_rep = ActionRepresentation(state_dim=len(one_step.idx2rules), ac_dim=64, device=args.device)
acrep_optim = torch.optim.Adam(params=ac_rep.parameters(), lr=0.0001)

molecule_loader = create_molecule_dataloader(target_mols, batch_size=1, shuffle=True)
for epoch in range(10):
    for mol_idx, batch in tqdm(enumerate(molecule_loader), total=len(molecule_loader), desc=f"Epoch {epoch+1}", leave=True):
        mol = smiles2array(batch[0], fp_dim=2048)
        obs = torch.Tensor(mol[np.newaxis]).to(device=args.device)
        _, _, idx_topk = one_step.forward_topk(obs, topk=2)
        
        anchor_idx = idx_topk[0, 0].item()
        positive_idx = idx_topk[0, 1].item()
        negatives_idx = random.sample(range(len(one_step.idx2rules)), 128)
        
        anchor_ohidx = F.one_hot(torch.tensor([anchor_idx]).to(device=args.device), num_classes=len(one_step.idx2rules))
        positive_ohidx = F.one_hot(torch.tensor([positive_idx]).to(device=args.device), num_classes=len(one_step.idx2rules))
        negative_ohidx = F.one_hot(torch.tensor(negatives_idx).to(device=args.device), num_classes=len(one_step.idx2rules))
    
        anchor_rep = ac_rep(anchor_ohidx.to(torch.float32)) # torch.Size([1, 64])
        positive_rep = ac_rep(positive_ohidx.to(torch.float32)) # torch.Size([1, 64])
        negative_rep = ac_rep(negative_ohidx.to(torch.float32)) #torch.Size([256, 64])
        
        margin = 0.0
        anchor_positive_dist = F.mse_loss(anchor_rep, positive_rep, reduction='none').sum(dim=1)
        anchor_negative_dists = F.mse_loss(anchor_rep.expand_as(negative_rep), negative_rep, reduction='none').sum(dim=1)
        contrastive_loss = torch.clamp(anchor_positive_dist - anchor_negative_dists + margin, min=-100000).mean()
        
        acrep_optim.zero_grad()
        contrastive_loss.backward()
        acrep_optim.step()
    
        acrep_training_info = {'ActionRep/contrastive loss': contrastive_loss.item}
        
        wandb.log({**acrep_training_info})
    
    logging.info("----------------------------------------")
    
    logging.info(f"Contrastive Loss: {contrastive_loss.item()}")

    start_time = time.time()

    chunk_size = 1024 * 2  # Adjust this to control memory usage
    min_distance = float('inf')
    min_index = None
    with torch.no_grad():
        sampled_indices = random.sample(range(len(one_step.idx2rules)), k=100000)
        for i in range(0, len(sampled_indices), chunk_size):
            end = min(i + chunk_size, len(one_step.idx2rules))
            
            indices = torch.arange(i, end)
            if positive_idx in indices:
                indices = indices[indices != positive_idx]
            
            chunk_rxns = F.one_hot(indices, num_classes=len(one_step.idx2rules)).to(torch.float32).to(device=args.device)
            chunk_reps = ac_rep(chunk_rxns)
            distances = torch.cdist(positive_rep, chunk_reps)
            chunk_min_dist, chunk_min_idx = torch.min(distances, dim=1)
            if chunk_min_dist.item() < min_distance:
                min_distance = chunk_min_dist.item()
                min_index = i + chunk_min_idx.item()
                
        logging.info("Evaluation on the reaction representation")
        logging.info(f"Target molecule {batch[0]}")
        logging.info(f"Positive reaction {positive_idx} with {rdchiralRunText(one_step.idx2rules[positive_idx], batch[0])}")
        logging.info(f"Nearest reaction {min_index} with {rdchiralRunText(one_step.idx2rules[min_index], batch[0])}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Time taken for finding nearest reaction: {elapsed_time:.2f} seconds")

        
if args.enamble_wandb:
    wandb.finished()



















