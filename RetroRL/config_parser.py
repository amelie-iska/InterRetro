import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--project', type=str, default='RSS')
    parser.add_argument('--group', type=str, default='RetroRL')
    parser.add_argument('--enable_wandb',  type=int, choices=[0, 1], default=1)
    parser.add_argument('--minirun',  type=int, choices=[0, 1], default=0)

    parser.add_argument('--training_interval', type=int, default=5)
    parser.add_argument('--evaluation_interval', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=5000)
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--pre_collections', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epsilon', type=int, default=0.05)
    parser.add_argument('--fp_dim', type=int, default=2048)
    parser.add_argument('--buffer_capacity', type=int, default=5e6)
    parser.add_argument('--discount', type=float, default=0.95)
    
    # MCTS
    parser.add_argument('--num_simulations', type=int, default=100)
    parser.add_argument('--PUCT_coef', type=float, default=1.0)
    parser.add_argument('--use_single_network', type=int, choices=[0, 1], default=1)
    parser.add_argument('--depth', type=int, default=15)
    
    parser.add_argument('--training_routes', type=str, default='../dataset/routes_train.pkl')
    parser.add_argument('--eval_routes', type=str, default='../dataset/routes_possible_test_hard.pkl')
    parser.add_argument('--building_blocks_path', type=str, default='../dataset/origin_dict_canonical.csv')
    parser.add_argument('--state_path', type=str, default='../one_step_model/saved_rollout_state_1_2048.ckpt')
    parser.add_argument('--template_path', type=str, default='../one_step_model/template_rules_1.dat')
    
    args = parser.parse_args()
    
    return args