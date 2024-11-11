
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
            n_layers=2, fp_dim=self.fp_dim, rxn_dim=self.one_step.net.n_rules, 
            latent_dim=512, dropout_rate=0.1, device=device
        )
        self.value_target = ValueMLP(
            n_layers=2, fp_dim=self.fp_dim, rxn_dim=self.one_step.net.n_rules, 
            latent_dim=512, dropout_rate=0.1, device=device
        )
        self.value_target.load_state_dict(self.value_net.state_dict())
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=0.001)
        self.value_net.eval()
        self.value_target.eval()
    
    
    def update(self, batch):
        critic_info = self.train_critic(batch)
        actor_info = {}#self.train_actor(batch)
        self.updates = self.updates + 1
        if self.updates % 2 == 0:
            self.update_value_target()
        return {**critic_info, 
                **actor_info,
                **{'training/updates': self.updates}}
    
    def train_actor(self, batch):
        obs = torch.Tensor(batch['obs']).to(self.device)
        preds_reference = self.one_step.reference_net(obs).detach()
        preds_reference, idx_topk = torch.topk(preds_reference, k=self.topk)
        candidate_actions = F.one_hot(idx_topk, num_classes=self.one_step.net.n_rules)
        
        q_values = []
        for i in range(self.topk):
            q_value = self.value_net(obs, candidate_actions[:, i, :]).squeeze()
            q_values.append(q_value)
        q_values = torch.stack(q_values, dim=1) 
        targets = torch.argmax(q_values, dim=1)
        # Here is a bug
        
        self.one_step.net.train()
        preds = self.one_step.net(obs)
        main_loss = F.cross_entropy(preds, targets)
        
        preds_topk = torch.gather(F.log_softmax(preds, dim=1), 1, idx_topk)
        reference_topk = F.softmax(preds_reference, dim=1)
        kl_divergence = F.kl_div(preds_topk, reference_topk, reduction='batchmean')
        
        actor_loss = main_loss + kl_divergence

        self.policy_opt.zero_grad()
        actor_loss.backward()
        self.policy_opt.step()
        self.one_step.net.eval()
        
        with torch.no_grad():
            _, pred_classes = torch.max(preds, dim=1)
            accuracy = (pred_classes == targets).float().mean()
    
        return {'training/actor_loss': actor_loss.item(),
                'training/accuracy': accuracy.item(),
                'training/kl_divergence': kl_divergence.item(),
                'training/main_loss': main_loss.item()}
    
    def train_critic(self, batch):
        max_len =  5 # max(len(next_obs) for next_obs in batch['next_obs'])
        batch_size = len(batch['next_obs'])
        obs = torch.Tensor(batch['obs']).to(self.device)
        acs = torch.Tensor(batch['actions']).to(self.device)
        
        padded_next_obs = torch.zeros((batch_size, max_len, self.fp_dim), device=self.device)
        padded_rewards = torch.zeros((batch_size, max_len, 1), device=self.device)
        padded_dones = torch.zeros((batch_size, max_len, 1), device=self.device)
        padded = torch.zeros((batch_size, max_len, 1), device=self.device) + torch.finfo(torch.float32).max
        for i, (next_obs, rewards, dones) in enumerate(zip(batch['next_obs'], batch['rewards'], batch['dones'])):
            stacked_obs = torch.stack([torch.Tensor(obs).to(self.device) for obs in next_obs])
            padded_next_obs[i, :stacked_obs.size(0)] = stacked_obs
            padded_rewards[i, :len(rewards)] = torch.Tensor(rewards).unsqueeze(1).to(self.device)
            padded_dones[i, :len(dones)] = torch.Tensor(dones).unsqueeze(1).to(self.device)
            padded[i, :len(dones)] = 1
        
        with torch.no_grad():
            flat_next_obs = padded_next_obs.view(-1, self.fp_dim)
            preds = self.one_step.reference_net(flat_next_obs)
            _, next_ac_idx = torch.max(preds, dim=1)
            next_actions = F.one_hot(next_ac_idx, num_classes=self.one_step.net.n_rules).float()
            next_actions = next_actions.view(batch_size, max_len, self.one_step.net.n_rules)
            q_values = self.value_target(flat_next_obs, next_actions.view(-1, next_actions.size(-1)))
            q_values = q_values.view(batch_size, max_len, 1)
            
            targets = (padded_rewards + (1 - padded_dones) * q_values) * padded
            targets = targets.min(dim=1).values
            
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
        
        '''
        obs = torch.Tensor(batch['obs']).to(self.device)
        acs = torch.Tensor(batch['actions']).to(self.device)
        targets = torch.zeros(len(batch['obs']), 1).to(self.device)
        
        with torch.no_grad():
            for i, (next_obs, rwd, dones) in enumerate(zip(batch['next_obs'], batch['rewards'], batch['dones'])):
                rwd = torch.Tensor(rwd).unsqueeze(1).to(self.device)
                next_obs = torch.Tensor(np.array(next_obs)).to(self.device)
                dones = torch.Tensor(dones).unsqueeze(1).to(self.device)

                #preds = self.one_step.net(next_obs)
                preds = self.one_step.reference_net(next_obs)
                _, next_ac_idx = torch.max(preds, dim=1)
                next_actions = F.one_hot(next_ac_idx, num_classes=self.one_step.net.n_rules)
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
        '''
    
    def get_action(self, a, exploration=False):
        arr = smiles2array(a, self.fp_dim)
        obs = torch.Tensor(arr[np.newaxis]).to(self.device)
        preds_reference = self.one_step.reference_net(obs).detach()
        preds_reference, idx_topk = torch.topk(preds_reference, k=self.topk)
        idx_topk = idx_topk.squeeze(dim=0)

        if exploration:
            #probs = F.softmax(preds_reference, dim=1)
            probs = torch.ones_like(preds_reference) / self.topk
            idx = torch.multinomial(probs, num_samples=1).cpu().item()
        else:
            candidate_actions = F.one_hot(idx_topk, num_classes=self.one_step.net.n_rules)
            q_values = self.value_net(obs.repeat(self.topk, 1), candidate_actions)
            idx = torch.argmax(q_values, dim=0).cpu().item()

        rxn_idx = idx_topk[idx].cpu().item()
        rxn = self.one_step.idx2rules[rxn_idx]
        return rxn

    def update_value_target(self):
        for param, target_param in zip(self.value_net.parameters(), self.value_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
            
            