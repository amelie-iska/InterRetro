import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--project', type=str, default='RSS')
    parser.add_argument('--group', type=str, default='RetroRL')
    parser.add_argument('--enable_wandb',  type=int, choices=[0, 1], default=0)
    parser.add_argument('--minirun',  type=int, choices=[0, 1], default=1)
    
    parser.add_argument('--evaluation_interval', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--runner', type=str, default='serial')
    
    # One-step model
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--realistic_filter', type=int, choices=[0, 1], default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--minibatch_size', type=int, default=128)
    parser.add_argument('--fp_dim', type=int, default=2048)
    
    # MCTS
    parser.add_argument('--expansion_topk', type=int, default=50) # 50
    parser.add_argument('--iterations', type=int, default=50) # 500
    parser.add_argument('--use_value_fn',  type=int, choices=[0, 1], default=1)
    parser.add_argument('--num_simulations', type=int, default=100) # 100
    parser.add_argument('--PUCT_coef', type=float, default=1.0)
    parser.add_argument('--use_single_network', type=int, choices=[0, 1], default=1)
    parser.add_argument('--depth', type=int, default=15)
    
    parser.add_argument('--training_routes', type=str, default='../dataset/routes_train.pkl')
    parser.add_argument('--eval_routes', type=str, default='../dataset/routes_possible_test_hard.pkl')
    parser.add_argument('--building_blocks_path', type=str, default='../dataset/origin_dict_canonical.csv')
    parser.add_argument('--state_path', type=str, default='../one_step_model/saved_rollout_state_1_2048.ckpt')
    parser.add_argument('--template_path', type=str, default='../one_step_model/template_rules_1.dat')
    parser.add_argument('--graph2edits_model', type=str, default='../Graph2Edits/graph2edits_experiments/uspto_50k/without_rxn_class/27-06-2022--10-27-22/epoch_123.pt')
    parser.add_argument('--bond_vocab_file', type=str, default='../Graph2Edits/data/uspto_50k/train/bond_vocab.txt')
    parser.add_argument('--atom_vocab_file', type=str, default='../Graph2Edits/data/uspto_50k/train/atom_lg_vocab.txt')
    
    args = parser.parse_args()
    
    return args