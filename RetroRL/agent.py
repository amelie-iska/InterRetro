
import torch
import torch.nn.functional as F

import numpy as np

from models import MLPModel, ActionRepresentation, ValueMLP, PriorMLP
from utils import smiles2array

class RetroRL:
    def __init__(self, one_step, discount, topk=10, fp_dim=2048, minibatch_size=1024, device='cuda'):
        self.fp_dim = fp_dim
        self.topk = topk
        self.device = device
        self.discount = discount
        self.one_step = one_step
        self.minibatch_size = minibatch_size
        self.updates = 0
        
        self.actor_optim = torch.optim.Adam(self.one_step.net.parameters(), lr=0.001)
        self.one_step.net.eval()
        
        self.syn_net = PriorMLP(
            n_layers=1, fp_dim=self.fp_dim, latent_dim=128, 
            dropout_rate=0.1, device=device)
        self.syn_net.eval()
        self.syn_net_optim = torch.optim.Adam(list(self.syn_net.parameters()), lr=0.001)
        
        self.value_net = ValueMLP(
            n_layers=1, fp_dim=self.fp_dim, latent_dim=128, 
            dropout_rate=0.1, device=device)
        self.value_net.load_state_dict(torch.load('../saved_models/best_epoch_final_4.pt'))
        self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=0.001)
        self.value_net.eval()
    
    
    def fit_syn(self, mols, pris, epoch_num=1, once=True):
        fps = np.array([smiles2array(mol, fp_dim=self.fp_dim) for mol in mols])
        pris = np.array(pris)
        loss = [] # Append all batches in all epochs. 
        self.syn_net.train()

        if once: # move all collected mols and pris from cpu to gpu once. 
            fps = torch.FloatTensor(fps).to(self.device)
            pris = torch.FloatTensor(pris).to(self.device)

        for _ in range(epoch_num):
            fps, pris = self._shuffle_data(fps, pris)
            for i in range(0, len(mols), self.minibatch_size):
                fps_batch = fps[i:i+self.minibatch_size]
                pris_batch = pris[i:i+self.minibatch_size].unsqueeze(-1)
                if not once: # move current batch from cpu to gpu. 
                    fps_batch = torch.FloatTensor(fps_batch).to(self.device)
                    pris_batch = torch.FloatTensor(pris_batch).to(self.device)

                pris_batch_pred = self.syn_net(fps_batch)
                loss_minibatch = F.binary_cross_entropy(pris_batch_pred, pris_batch)
                loss.append(loss_minibatch.item())
                
                self.syn_net_optim.zero_grad()
                loss_minibatch.backward()
                torch.nn.utils.clip_grad_norm_(self.syn_net.parameters(), 0.5)
                self.syn_net_optim.step()
        self.syn_net.eval()
        return {'syn/loss': np.mean(loss).item()}
    
    def fit_value(self, mols, vals, epoch_num=1, once=True):
        fps = np.array([smiles2array(mol, fp_dim=self.fp_dim) for mol in mols])
        vals = np.array(vals)
        loss = [] # Append all batches in all epochs. 
        self.value_net.train()

        if once: # move all collected mols and pris from cpu to gpu once. 
            fps = torch.FloatTensor(fps).to(self.device)
            vals = torch.FloatTensor(vals).to(self.device)

        for _ in range(epoch_num):
            fps, vals = self._shuffle_data(fps, vals)
            for i in range(0, len(mols), self.minibatch_size):
                fps_batch = fps[i:i+self.minibatch_size]
                vals_batch = vals[i:i+self.minibatch_size].unsqueeze(-1)
                if not once: # move current batch from cpu to gpu. 
                    fps_batch = torch.FloatTensor(fps_batch).to(self.device)
                    vals_batch = torch.FloatTensor(vals_batch).to(self.device)

                vals_batch_pred = self.value_net(fps_batch)
                loss_minibatch = F.mse_loss(vals_batch_pred, vals_batch)
                loss.append(loss_minibatch.item())
                self.value_optim.zero_grad()
                loss_minibatch.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optim.step()

        self.value_net.eval()
        return {'value/loss': np.mean(loss).item()}
    
    def fit_actor(self, mols, tmps, epoch_num=1, once=True):
        fps = np.array([smiles2array(mol, fp_dim=self.fp_dim) for mol in mols])
        tmps = np.array(tmps)
        loss = []
        self.one_step.net.train()
        for m in self.one_step.net.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()

        if once: # move all collected mols and pris from cpu to gpu once. 
            fps = torch.FloatTensor(fps).to(self.device)
            tmps = torch.LongTensor(tmps).to(self.device)
 
        for _ in range(epoch_num):
            fps, tmps = self._shuffle_data(fps, tmps)
            for i in range(0, len(mols), self.minibatch_size):
 
                fps_batch = fps[i:i+self.minibatch_size]
                tmps_batch = tmps[i:i+self.minibatch_size]
                if not once: # move current batch from cpu to gpu. 
                    fps_batch = torch.FloatTensor(fps_batch).to(self.device)
                    tmps_batch = torch.LongTensor(tmps_batch).to(self.device)
 
                action_logits = self.one_step.net(fps_batch)
                loss_minibatch = F.cross_entropy(action_logits, tmps_batch)
                print(loss_minibatch)
                loss.append(loss_minibatch.item())
                self.actor_optim.zero_grad()
                loss_minibatch.backward()
                torch.nn.utils.clip_grad_norm_(self.one_step.net.parameters(), 0.5)
                self.actor_optim.step()
 
        self.one_step.net.eval()
        return {'actor/loss': np.mean(loss).item()}
    
    def _shuffle_data(self, features, labels):
        shuffled_idx = np.arange(len(features))
        np.random.shuffle(shuffled_idx)
        features = features[shuffled_idx]
        labels = labels[shuffled_idx]
        return features, labels
            
            