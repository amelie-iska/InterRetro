import pickle
import logging
import json
import random
import time
import wandb
from collections import deque
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdchiral.main import rdchiralRunText

from alg import online_mcts
from runners import SerialRunner, ParallelRunner
from agent import RetroRL
from envs import RetroEnv
from utils import set_seeds, create_exp, smiles2array, prepare_starting_molecules, minicut
from config_parser import get_args
from models import MLPModel, ActionRepresentation, ValueMLP
from data_loader import create_molecule_dataloader
from buffer import ReplayBuffer
from trajectories import TrajectoryCreator

def main():

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
    
    one_step = MLPModel(state_path=args.state_path, template_path=args.template_path, device=args.device)
    agent = RetroRL(one_step, topk=args.topk, discount=args.discount, minibatch_size=args.minibatch_size)
    '''
    env = RetroEnv(target_mols=target_mols, blocking_blocks=building_blocks)
    eval_env = RetroEnv(target_mols=eval_mols, blocking_blocks=building_blocks)
    buffer = ReplayBuffer(capacity=args.buffer_capacity, obs_dim=args.fp_dim, action_dim=len(agent.one_step.rules2idx))
    traj_creator = TrajectoryCreator(agent)
    '''
    
    def load_model_to_runner(model_net, value_net, prior_net, runner, n_gpus=1):
        if n_gpus > 1:
            # load model parameters to subprocesses using different gpus
            model_net.cpu()
            value_net.cpu()
            prior_net.cpu()
            runner.load_model(model_net.state_dict(), prior_net.state_dict(), value_net.state_dict())
            model_net.cuda()
            value_net.cuda()
            prior_net.cuda()
        else:
            runner.load_model(model_net.state_dict(), prior_net.state_dict(), value_net.state_dict())
    
    #runner = SerialRunner(args)
    runner = ParallelRunner(args)
    load_model_to_runner(one_step.net, agent.value_net, agent.syn_net, runner, 1)
    
    logging.info("Training starts ...")
    train_target_mols_idx = list(range(len(target_mols)))
    training_info = {}
    for epoch in range(args.epochs):
    
        random.shuffle(train_target_mols_idx)
        for i in tqdm(range(0, len(target_mols), args.batch_size)):
            target_mols_id_batch = train_target_mols_idx[i: i+args.batch_size]
            target_mols_batch = [target_mols[idx] for idx in target_mols_id_batch]
            
            mcts_start = time.time()
            mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result = \
                runner.mcts(target_mols_batch, target_mols_id_batch, test=False)
            mcts_end = time.time()
            training_info['general/timer'] = mcts_end - mcts_start
            
            syn_info = agent.fit_syn(mols_r_batch, pris_batch, epoch_num=8)
            critic_info = agent.fit_value(mols_v_batch, vals_batch, epoch_num=8)
            actor_info = agent.fit_actor(mols_t_batch, tmps_batch, epoch_num=8)
            training_info = training_info | syn_info | critic_info | actor_info
        
            if i % args.evaluation_interval == 0:
                _, _, _, _, _, _, test_batch_result = runner.mcts(eval_mols[:20], list(range(len(eval_mols[:20]))), test=True)
                training_info['evaluation/success_rate'] = np.array(test_batch_result['succ']).sum()/len(test_batch_result['succ'])
                training_info['evaluation/depth'] = np.array(test_batch_result['route_lens']).sum()/len(test_batch_result['route_lens'])
                training_info['evaluation/model_calls'] = np.array(test_batch_result['iter']).sum()/len(test_batch_result['iter'])
                training_info['evaluation/route_shortest_cost'] = np.array(test_batch_result['route_shortest_cost']).sum()/len(test_batch_result['route_shortest_cost'])
                logging.info(f"Evaluation ep {epoch} mol {i}\n" + "\n".join([f"{key}: {value:.4f}" for key, value in training_info.items()]))
        
            if args.enable_wandb:
                wandb.log({**training_info})
                
        agent.save(save_path=exp_folder, epoch=epoch)
    
    if args.enable_wandb:
        wandb.finished()
    
    
    
if __name__ == '__main__':
    main()   
    














