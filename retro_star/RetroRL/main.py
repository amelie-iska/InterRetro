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

from agent import RetroRL
from envs import RetroEnv
from utils import set_seeds, create_exp, smiles2array, prepare_starting_molecules, minicut
from config_parser import get_args
from models import MLPModel, ActionRepresentation, ValueMLP
from data_loader import create_molecule_dataloader
from buffer import ReplayBuffer
from trajectories import TrajectoryCreator

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
logging.info(f"Experiment run directory created in {exp_folder}")
with open(exp_folder + "/args.json", "w") as f:
    json.dump(args.__dict__, f, indent=4)
if args.enable_wandb:
    wandb.init(project=args.project, config=args, group=args.group, name='{}_seed{}'.format(args.group, args.seed))

routes = minicut(pickle.load(open(args.training_routes, 'rb')), args.minirun)
logging.info(f"{len(routes)} training routes have been loaded.")
target_mols = []
for route in routes:
    for rxn in route:
        target_mols.append(Chem.CanonSmiles(rxn.split('>')[0]))
target_mols = list(set(target_mols))
logging.info(f"{len(target_mols)} target molecules are loaded after removing duplicates. ")

eval_routes = pickle.load(open(args.eval_routes, 'rb'))
logging.info(f"{len(eval_routes)} evaluation routes have been loaded.")
eval_mols = []
for route in eval_routes:
    for rxn in route:
        eval_mols.append(Chem.CanonSmiles(rxn.split('>')[0]))
eval_mols = list(set(eval_mols))
logging.info(f"{len(eval_mols)} evaluation molecules are loaded after removing duplicates. ")

building_blocks = prepare_starting_molecules(args.building_blocks_path)
logging.info(f"{len(building_blocks)} building blocks have been loaded. ")

env = RetroEnv(target_mols=target_mols, blocking_blocks=building_blocks)
eval_env = RetroEnv(target_mols=eval_mols, blocking_blocks=building_blocks)

one_step = MLPModel(state_path=args.state_path, template_path=args.template_path, device=args.device)
agent = RetroRL(one_step)

buffer = ReplayBuffer(capacity=args.buffer_capacity, obs_dim=args.fp_dim, action_dim=len(agent.one_step.rules2idx))

traj_creator = TrajectoryCreator(agent)



for epoch in range(args.epochs):
    for mol_idx in tqdm(range(len(env.target_mols)), mininterval=1.0, desc=f"Epoch {epoch}"):
        root = env.build_tree(agent, mol_idx=mol_idx, exploration_rate=0.05)
        trajs = traj_creator.tree2traj(root)
        translated_trajs = traj_creator.translate_trajs(trajs)
        for tj in translated_trajs:
            for i in range(0, len(tj)-1, 2):
                if len(tj) == i+3:
                    buffer.add(obs=tj[i], next_obs=tj[i+2], action=tj[i+1], reward=1, done=1)
                else:
                    buffer.add(obs=tj[i], next_obs=tj[i+2], action=tj[i+1], reward=0, done=0)
                    
        if mol_idx % args.training_interval == 0 and len(buffer) > 1000:
            batch = buffer.sample(128)
            critic_info = agent.train_critic(batch)
            actor_info = agent.train_actor(batch)
            
        
        if mol_idx % args.evaluation_interval == 0 and len(buffer) > 1000:
            training_info = env.eval(agent, mode='training')
            eval_info = eval_env.eval(agent, mode='evaluation')
            info = {**training_info, **eval_info, **critic_info, **actor_info}
            logging.info(f"Evaluation ep {epoch} mol {mol_idx}\n"
                         + "\n".join([f"{key}: {value:.4f}" for key, value in info.items()]))
            if args.enable_wandb:
                wandb.log(info)

        
        
        
                    
if args.enable_wandb:
    wandb.finished()



















