
import torch
import torch.nn.functional as F

import numpy as np

from models import MLPModel, ActionRepresentation, ValueMLP
from utils import smiles2array

class RetroRL:
    def __init__(self, one_step, action_mode, discount, topk=10, fp_dim=2048, device='cuda'):
        self.fp_dim = fp_dim
        self.topk = 10
        self.device = device
        self.action_mode = action_mode
        self.discount = discount
        self.one_step = one_step
        self.updates = 0
        
        self.policy_opt = torch.optim.Adam(self.one_step.net.parameters(), lr=0.001)
        self.one_step.net.eval()
        
        self.value_net = ValueMLP(
            n_layers=1, fp_dim=self.fp_dim, rxn_dim=self.one_step.net.n_rules, 
            latent_dim=128, dropout_rate=0.1, device=device
        )
        self.value_target = ValueMLP(
            n_layers=1, fp_dim=self.fp_dim, rxn_dim=self.one_step.net.n_rules, 
            latent_dim=128, dropout_rate=0.1, device=device
        )
        self.value_target.load_state_dict(self.value_net.state_dict())
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=0.001)
        self.value_net.eval()
        self.value_target.eval()
    
    
    def update(self, batch):
        critic_info = self.train_critic(batch)
        self.updates = self.updates + 1
        if self.updates % 2 == 0:
            self.update_value_target()
        return {**critic_info, 
                **{'training/updates': self.updates}}
    
    def train_actor(self, batch):
        obs = torch.Tensor(batch['obs']).to(self.device)
    
        self.one_step.net.train()
    
        preds, _, idx = self.one_step.forward_topk(obs, topk=10)
        candidate_actions = F.one_hot(idx, num_classes=self.one_step.net.n_rules) 
        
        q_values = []
        for i in range(10):
            q_value = self.value_net(obs, candidate_actions[:, i, :]).squeeze()
            q_values.append(q_value)
        q_values = torch.stack(q_values, dim=1) 
    
        target_action_indices = torch.argmax(q_values, dim=1)
    
        actor_loss = F.cross_entropy(preds, target_action_indices)

        self.policy_opt.zero_grad()
        actor_loss.backward()
        self.policy_opt.step()
        
        self.one_step.net.eval()
    
        return {'training/actor_loss': actor_loss.item()}
    
    def train_critic(self, batch):
        obs = torch.Tensor(batch['obs']).to(self.device)
        acs = torch.Tensor(batch['actions']).to(self.device)
        targets = torch.zeros(len(batch['obs']), 1).to(self.device)
        
        with torch.no_grad():
            for i, (next_obs, rwd, dones) in enumerate(zip(batch['next_obs'], batch['rewards'], batch['dones'])):
                rwd = torch.Tensor(rwd).unsqueeze(1).to(self.device)
                next_obs = torch.Tensor(np.array(next_obs)).to(self.device)
                dones = torch.Tensor(dones).unsqueeze(1).to(self.device)
                
                _, _, idx = self.one_step.forward_topk(next_obs, topk=1) 
                next_actions = F.one_hot(idx.squeeze(-1), num_classes=self.one_step.net.n_rules)
                next_values = rwd + (1 - dones) * self.value_target(next_obs, next_actions)
                
                min_values, min_idx = torch.min(next_values, dim=0)
                targets[i] = min_values
        
        self.value_net.train()
        preds = self.value_net(obs, acs)
        loss = F.mse_loss(preds, targets)

        self.value_opt.zero_grad()
        loss.backward()
        self.value_opt.step()

        self.value_net.eval()

        return {'training/critic_loss': loss.item(),
                'training/critic_targets': targets.mean().item(),
                'training/critic_predicts': preds.mean().item()}
    

    def get_action(self, a):
        arr = smiles2array(a, self.fp_dim)
        arr = np.reshape(arr,[-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        arr = arr.to(device='cuda')
        preds, preds_reference, idx = self.one_step.forward_topk(arr, topk=self.topk) 
        if self.action_mode == 'sampling':
            probs = F.softmax(preds, dim=1)
            probs_reference = F.softmax(preds_reference, dim=1)
            action_class = torch.multinomial(probs, num_samples=1).squeeze().cpu().numpy()
            reference_class = torch.multinomial(probs_reference, num_samples=1).squeeze().cpu().numpy()
        elif self.action_mode == 'greedy': 
            action_class = torch.argmax(preds, dim=1).cpu().numpy()
            reference_class = torch.argmax(preds_reference, dim=1).cpu().numpy()
        elif self.action_mode == 'optimiseQ':
            action_candidates_oh = F.one_hot(idx.squeeze(0), num_classes=self.one_step.net.n_rules)
            values = self.value_net(arr.repeat(10, 1), action_candidates_oh)
            best_action = torch.max(values, dim=0)[1]
            reaction = self.one_step.idx2rules[best_action.item()]
            return reaction, None
        action_indices = idx[0][action_class].cpu().item()
        reference_indices = idx[0][reference_class].cpu().item()

        ac_reaction = self.one_step.idx2rules[action_indices]
        ref_reaction = self.one_step.idx2rules[reference_indices]
        return ac_reaction, ref_reaction
    
    def get_random_action(self, a):
        arr = smiles2array(a, self.fp_dim)
        arr = np.reshape(arr,[-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        arr = arr.to(device='cuda')
        preds, preds_reference, idx = self.one_step.forward_topk(arr, topk=20) 

        probs = F.softmax(preds, dim=1)
        probs = torch.ones_like(probs) / 20
        probs_reference = F.softmax(preds_reference, dim=1)
        probs_reference = torch.ones_like(probs_reference) / 20
        action_class = torch.multinomial(probs, num_samples=1).squeeze().cpu().numpy()
        reference_class = torch.multinomial(probs_reference, num_samples=1).squeeze().cpu().numpy()
       
        action_indices = idx[0][action_class].cpu().item()
        reference_indices = idx[0][reference_class].cpu().item()

        ac_reaction = self.one_step.idx2rules[action_indices]
        ref_reaction = self.one_step.idx2rules[reference_indices]
        return ac_reaction, ref_reaction

    def update_value_target(self):
        for param, target_param in zip(self.value_net.parameters(), self.value_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
            
            