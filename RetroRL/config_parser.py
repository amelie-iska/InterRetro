import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device',  type=str, choices=['cuda', 'mps', 'cpu'], default='cuda')
    parser.add_argument('--n_processes', type=int, default=6)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--project', type=str, default='RSS')
    parser.add_argument('--group', type=str, default='RetroRL')
    parser.add_argument('--enable_wandb',  type=int, choices=[0, 1], default=1)
    parser.add_argument('--minirun_rate',  type=int, choices=[0, 1], default=10)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=250)
    
    # One-step model
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--buffer_capacity', type=int, default=10000)
    parser.add_argument('--training_batch_size', type=int, default=64)
    parser.add_argument('--explore_batch_size', type=int, default=36)
    parser.add_argument('--update_frequency', type=int, default=3)
    
    parser.add_argument('--training_routes', type=str, default='../dataset/routes_train.pkl')
    parser.add_argument('--eval_routes', type=str, default='../dataset/routes_possible_test_hard.pkl')
    parser.add_argument('--building_blocks_path', type=str, default='../dataset/origin_dict_canonical.csv')
    parser.add_argument('--graph2edits_model', type=str, default='../Graph2Edits/graph2edits_experiments/uspto_50k/without_rxn_class/27-06-2022--10-27-22/epoch_123.pt')
    parser.add_argument('--bond_vocab_file', type=str, default='../Graph2Edits/data/uspto_50k/train/bond_vocab.txt')
    parser.add_argument('--atom_vocab_file', type=str, default='../Graph2Edits/data/uspto_50k/train/atom_lg_vocab.txt')
    
    args = parser.parse_args()
    
    return args
