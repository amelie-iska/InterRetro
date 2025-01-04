import pickle
import logging
import json
import random
import time
import wandb
from collections import deque
import numpy as np
import joblib

import torch
import torch.nn.functional as F

from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdchiral.main import rdchiralRunText

from replay_buffer import ReplayBuffer
from trajectories import TrajectoryCreator
from env import RetroEnv
from alg import online_mcts
from runners import SerialRunner, ParallelRunner
from agent import RetroRL
from retrorl_utils import set_seeds, create_exp, smiles2array, prepare_starting_molecules, minicut, \
    load_model_to_runner
from config_parser import get_args


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
    eval_mols.append(Chem.CanonSmiles(route[0].split('>')[0]))
eval_mols = list(set(eval_mols))
logging.info(f"{len(eval_mols)} evaluation molecules are loaded after removing duplicates. ")

building_blocks = prepare_starting_molecules(args.building_blocks_path)
logging.info(f"{len(building_blocks)} building blocks have been loaded. ")


env = RetroEnv(blocking_blocks=building_blocks)
agent = RetroRL(args)

buffer = ReplayBuffer(capacity=10000)
traj_creator = TrajectoryCreator(agent)

#buffer = joblib.load('buffer_epoch32.pkl')

logging.info("Training starts ...")
archive = set(eval_mols)
for epoch in range(args.epochs):
    
    logs = {}
    eval_info = env.eval(agent, eval_mols)
    logs.update(eval_info)
    
    for mol_idx in tqdm(range(100), mininterval=1.0, 
                        desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True):
        
        mol = archive.pop()
        try:
            tree, succ_subtrees, info = env.build_tree(agent, mol)
        except Exception as e:
            logging.log(f'{e}')
            continue
        
        if len(succ_subtrees) > 0:
            for subtree in succ_subtrees:
               branches = traj_creator.extract_branches(subtree)
               for branch in branches:
                   buffer.add(obs=branch[0], actions=branch[1], next_obs=branch[2], reward=branch[3], done=branch[4])
        
        if not info['success']:
            archive.add(tree.id)
        
        if buffer.size > args.batch_size:
            rxn_batch = buffer.sample(args.batch_size)
            fit_logs = agent.fit(rxn_batch)
            logs.update(fit_logs)
        
        
        
        logs.update({'utils/buffer_size': len(buffer),
                     'utils/archive_size': len(archive)})
        if args.enable_wandb:
            wandb.log({**logs})
        logging.info(f"Training Info ep {epoch} mol {mol_idx}\n" + \
                     "\n".join([f"{key}: {value:.4f}" for key, value in logs.items()]))













