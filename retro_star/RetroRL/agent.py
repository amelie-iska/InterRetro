
import torch
import torch.nn.functional as F

import numpy as np

from models import MLPModel, ActionRepresentation, ValueMLP
from utils import smiles2array

class RetroRL:
    def __init__(self, one_step, topk=10, fp_dim=2048, sampling=False, device='cuda'):
        self.fp_dim = fp_dim
        self.topk = 10
        self.sampling = sampling
        self.device = device
        self.one_step = one_step
        
        self.policy_opt = torch.optim.Adam(self.one_step.net.parameters(), lr=0.001)
        self.one_step.net.eval()
        self.value_net = ValueMLP(
            n_layers=1, fp_dim=self.fp_dim, rxn_dim=self.one_step.net.n_rules, 
            latent_dim=128, dropout_rate=0.1, device=device
        )
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=0.001)
        self.value_net.eval()
        
    def train_actor(self, batch):
        obs = torch.Tensor(batch['obs']).to(self.device)
    
        self.one_step.net.train()
    
        preds, _, idx = self.one_step.forward_topk(obs, topk=10)
        candidate_actions = F.one_hot(idx, num_classes=self.one_step.net.n_rules).float()  
        
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
    
        return {'actor/loss': actor_loss.item()}
    
    def train_critic(self, batch):
        obs = torch.Tensor(batch['obs']).to(self.device)
        acs = torch.Tensor(batch['actions']).to(self.device)
        next_obs = torch.Tensor(batch['next_obs']).to(self.device)
        rwd = torch.Tensor(batch['rewards']).to(self.device)
        done = torch.Tensor(batch['dones']).to(self.device)

        with torch.no_grad():
            preds, _, _ = self.one_step.forward_topk(next_obs, topk=1) 
            probs = F.softmax(preds, dim=1)
            next_actions = torch.multinomial(probs, num_samples=1)
            next_actions = F.one_hot(next_actions.squeeze(-1), num_classes=self.one_step.net.n_rules)
            target_q_values = self.value_net(next_obs, next_actions).squeeze()
            target = rwd + (1 - done) * target_q_values
        

        self.value_net.train()
        pred = self.value_net(obs, acs).squeeze()
        loss = F.mse_loss(pred, target)

        print(target, pred)

        self.value_opt.zero_grad()
        loss.backward()
        self.value_opt.step()

        self.value_net.eval()

        return {'critic/loss': loss.item()}
    
    def get_action(self, a):
        arr = smiles2array(a, self.fp_dim)
        arr = np.reshape(arr,[-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        arr = arr.to(device='cuda')
        preds, preds_reference, idx = self.one_step.forward_topk(arr, topk=self.topk) 
        if self.sampling:
            probs = F.softmax(preds, dim=1)
            probs_reference = F.softmax(preds_reference, dim=1)
            action_class = torch.multinomial(probs, num_samples=1).squeeze().cpu().numpy()
            reference_class = torch.multinomial(probs_reference, num_samples=1).squeeze().cpu().numpy()
        else: 
            action_class = torch.argmax(preds, dim=1).cpu().numpy()
            reference_class = torch.argmax(preds_reference, dim=1).cpu().numpy()
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
    
    
        
        
        