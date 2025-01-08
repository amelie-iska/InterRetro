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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == '__main__':
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
    
    agent = RetroRL(args)
    buffer = ReplayBuffer(capacity=10000, agent=agent)
    traj_creator = TrajectoryCreator(agent)
    
    runner = ParallelRunner(args)
    runner.synchronise_models(agent.graph2edits.state_dict())
    logging.info("Training starts ...")
    for epoch in range(args.epochs):
        archive = set(eval_mols)
        archive_len = len(archive)
        for it in tqdm(range(2 * int(archive_len/args.explore_batch_size)), mininterval=1.0, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True):
            
            logs = {}
            
            mols = []
            for i in range(args.explore_batch_size):
                if len(archive) == 0: break
                mols.append(archive.pop())
            
            try:
                start_time = time.time()
                tree_list, succ_subtrees_list, info_list = runner.run(mols, test=False)
                end_time = time.time()
                logs.update({'explore/elapsed_time': end_time - start_time,
                             'explore/valid_percentage': len(tree_list) / args.explore_batch_size,
                             'explore/success_rate': sum([info['success'] for info in info_list]) / len(info_list)})
            except Exception as e:
                logging.warning(f'{e}')
                continue
            
            for tree, succ_subtrees, info in zip(tree_list, succ_subtrees_list, info_list):
                if len(succ_subtrees) > 0: # Save to buffer
                    for subtree in succ_subtrees:
                       branches = traj_creator.extract_branches(subtree)
                       for branch in branches:
                           buffer.add(**branch)
                if not info['success']: # Save to archive
                    archive.add(tree.id)
                        
            
            logs.update({'utils/buffer_size': len(buffer),
                         'utils/archive_size': len(archive)})
            logging.info(f"Training Info ep {epoch} it {it}\n" + \
                         "\n".join([f"{key}: {value:.4f}" for key, value in logs.items()]))
            if args.enable_wandb:
                wandb.log({**logs})
        
        
        # Training
        logs = {}
        for _ in tqdm(range(1000), mininterval=1.0, desc=f"After-epoch{epoch} training", dynamic_ncols=True, leave=True):
            rxn_batch = buffer.sample(args.training_batch_size)
            fit_logs = agent.fit(rxn_batch)
            logs.update(fit_logs)
        runner.synchronise_models(agent.graph2edits.state_dict())
        
        # Evaluation
        logging.info(f"Evaluation in epoch {epoch} starts ...")
        tree_list, succ_subtrees_list, info_list = runner.run(eval_mols, test=True)
        logs.update({'evaluation/success_rate': sum([info['success'] for info in info_list]) / len(info_list)})
        
        joblib.dump([agent, buffer], f'{exp_folder}/models_epoch{epoch}.pkl')
        logging.info(f"After-epoch Training Info ep {epoch}\n" + \
                     "\n".join([f"{key}: {value:.4f}" for key, value in logs.items()]))
        if args.enable_wandb:
            wandb.log({**logs})
        








