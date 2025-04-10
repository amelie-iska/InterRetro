import pickle
import logging
import json
import time
import wandb
import torch

from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from replay_buffer import ReplayBuffer
from trajectories import TrajectoryCreator
from runners import ParallelRunner
from retrorl_utils import set_seeds, create_exp, minicut
from config_parser import get_args
from agents import return_agent

if __name__ == '__main__':
    args = get_args()
    args.seed = set_seeds(args.seed)
    exp_folder = create_exp(args.seed, args.group)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(exp_folder + "/build_dataset_log.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Experiment run directory created in {exp_folder}")
    with open(exp_folder + "/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)
    if args.enable_wandb:
        wandb.init(project=args.project, config=args, group=args.group, name='{}_seed{}'.format(args.group, args.seed))
        
    eval_routes = pickle.load(open(args.eval_routes, 'rb'))
    logging.info(f"{len(eval_routes)} evaluation routes have been loaded.")
    eval_mols = []
    for route in eval_routes:
        eval_mols.append(Chem.CanonSmiles(route[0].split('>')[0]))
    del eval_routes
    eval_mols = list(set(eval_mols))
    logging.info(f"{len(eval_mols)} evaluation molecules are loaded after removing duplicates. ")
    
    agent = return_agent(args)
    buffer = ReplayBuffer(capacity=args.buffer_capacity, agent=agent)
    buffer.buffer = torch.load('buffer_epoch1_no772', weights_only=False)
    buffer.size = buffer.capacity
    logging.info("Building the graph cache in the buffer. ")
    buffer.update_graphs()
    traj_creator = TrajectoryCreator(agent)

    runner = ParallelRunner(args)
    runner.synchronise_models(agent.get_state_dict())
    logging.info("Training starts ...")
    for epoch in range(100):
        
        update = 0
        for rxn_batch in tqdm(buffer.sample_without_replacement(batch_size=args.training_batch_size), 
                              total=int(buffer.capacity / args.training_batch_size)):
            
            logs = {}
            
            policy_logs = agent.fit_policy(rxn_batch)
            value_logs = agent.fit_value(rxn_batch)
            logs.update(policy_logs)
            logs.update(value_logs)
    
            # Evaluation
            
            if update % 200 == 0:
                runner.synchronise_models(agent.get_state_dict())
                start_time = time.time()
                tree_list, succ_subtrees_list, info_list = runner.run(eval_mols, test=True)
                end_time = time.time()
                logs.update({'evaluation/elapsed_time': end_time - start_time,
                             'evaluation/success_rate': sum([info['success'] for info in info_list]) / len(eval_mols),
                             'evaluation/depth': sum([info['depth'] for info in info_list]) / len(info_list)})
            
            if update % 100 == 0:
                logging.info(f"Training Info ep {epoch} update {update}\n" + "\n".join([f"{key}: {value:.4f}" for key, value in logs.items()]))
            
            if args.enable_wandb:
                wandb.log({**logs})
            update = update + 1
            
        torch.save(agent.get_state_dict(), f'{exp_folder}/models_epoch{epoch}.pth')
        
    
        
