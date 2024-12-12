
import os
import torch
import torch.nn.functional as F

import numpy as np

from models import MLPModel, ActionRepresentation, ValueMLP, PriorMLP
from utils import smiles2array

class RetroRL:
    def __init__(self, args):
        self.args = args
        self.updates = 0
        
        self.actor = MLPModel(state_path=args.state_path, template_path=args.template_path, 
                            device=args.device, realistic_filter=args.realistic_filter)
        self.actor_optim = torch.optim.Adam(self.actor.net.parameters(), lr=0.001)
        self.actor.net.eval()
        
        self.syn_net = PriorMLP(n_layers=1, fp_dim=args.fp_dim, latent_dim=128, 
                                dropout_rate=0.1, device=args.device)
        self.syn_net_optim = torch.optim.Adam(list(self.syn_net.parameters()), lr=0.001)
        self.syn_net.eval()
        
        self.value_net = ValueMLP(n_layers=1, fp_dim=args.fp_dim, latent_dim=128, 
                                  dropout_rate=0.1, device=args.device)
        # self.value_net.load_state_dict(torch.load('../saved_models/best_epoch_final_4.pt'))
        self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=0.001)
        self.value_net.eval()
    
    
    def fit_syn(self, mols, pris, epoch_num=1, once=True):
        fps = np.array([smiles2array(mol, fp_dim=self.args.fp_dim) for mol in mols])
        pris = np.array(pris)
        loss = [] # Append all batches in all epochs. 
        self.syn_net.train()

        if once: # move all collected mols and pris from cpu to gpu once. 
            fps = torch.FloatTensor(fps).to(self.args.device)
            pris = torch.FloatTensor(pris).to(self.args.device)

        for _ in range(epoch_num):
            fps, pris = self._shuffle_data(fps, pris)
            for i in range(0, len(mols), self.args.minibatch_size):
                fps_batch = fps[i:i+self.args.minibatch_size]
                pris_batch = pris[i:i+self.args.minibatch_size].unsqueeze(-1)
                if not once: # move current batch from cpu to gpu. 
                    fps_batch = torch.FloatTensor(fps_batch).to(self.args.device)
                    pris_batch = torch.FloatTensor(pris_batch).to(self.args.device)

                pris_batch_pred = self.syn_net(fps_batch)
                loss_minibatch = F.binary_cross_entropy(pris_batch_pred, pris_batch)
                loss.append(loss_minibatch.item())
                
                self.syn_net_optim.zero_grad()
                loss_minibatch.backward()
                torch.nn.utils.clip_grad_norm_(self.syn_net.parameters(), 0.5)
                self.syn_net_optim.step()
        self.syn_net.eval()
        return {'training_info/syn_loss': np.mean(loss).item()}
    
    def fit_value(self, mols, vals, epoch_num=1, once=True):
        fps = np.array([smiles2array(mol, fp_dim=self.args.fp_dim) for mol in mols])
        vals = np.array(vals)
        loss = [] # Append all batches in all epochs. 
        self.value_net.train()

        if once: # move all collected mols and pris from cpu to gpu once. 
            fps = torch.FloatTensor(fps).to(self.args.device)
            vals = torch.FloatTensor(vals).to(self.args.device)

        for _ in range(epoch_num):
            fps, vals = self._shuffle_data(fps, vals)
            for i in range(0, len(mols), self.args.minibatch_size):
                fps_batch = fps[i:i+self.args.minibatch_size]
                vals_batch = vals[i:i+self.args.minibatch_size].unsqueeze(-1)
                if not once: # move current batch from cpu to gpu. 
                    fps_batch = torch.FloatTensor(fps_batch).to(self.args.device)
                    vals_batch = torch.FloatTensor(vals_batch).to(self.args.device)

                vals_batch_pred = self.value_net(fps_batch)
                loss_minibatch = F.mse_loss(vals_batch_pred, vals_batch)
                loss.append(loss_minibatch.item())
                self.value_optim.zero_grad()
                loss_minibatch.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optim.step()

        self.value_net.eval()
        return {'training_info/value_loss': np.mean(loss).item()}
    
    def fit_actor(self, mols, tmps, epoch_num=1, once=True):
        fps = np.array([smiles2array(mol, fp_dim=self.args.fp_dim) for mol in mols])
        tmps = np.array(tmps)
        loss = []
        self.actor.net.train()
        for m in self.actor.net.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()

        if once: # move all collected mols and pris from cpu to gpu once. 
            fps = torch.FloatTensor(fps).to(self.args.device)
            tmps = torch.LongTensor(tmps).to(self.args.device)
 
        for _ in range(epoch_num):
            fps, tmps = self._shuffle_data(fps, tmps)
            for i in range(0, len(mols), self.args.minibatch_size):
 
                fps_batch = fps[i:i+self.args.minibatch_size]
                tmps_batch = tmps[i:i+self.args.minibatch_size]
                if not once: # move current batch from cpu to gpu. 
                    fps_batch = torch.FloatTensor(fps_batch).to(self.args.device)
                    tmps_batch = torch.LongTensor(tmps_batch).to(self.args.device)
 
                if self.args.realistic_filter:
                   action_logits, _, _ = self.actor.forward_topk(fps_batch, topk=self.args.expansion_topk)
                else:
                   action_logits = self.actor.net(fps_batch)
                loss_minibatch = F.cross_entropy(action_logits, tmps_batch)
                loss.append(loss_minibatch.item())
                self.actor_optim.zero_grad()
                loss_minibatch.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), 0.5)
                self.actor_optim.step()
 
        self.actor.net.eval()
        return {'training_info/actor_loss': np.mean(loss).item()}
    
    def _shuffle_data(self, features, labels):
        shuffled_idx = np.arange(len(features))
        np.random.shuffle(shuffled_idx)
        features = features[shuffled_idx]
        labels = labels[shuffled_idx]
        return features, labels
            

    def save(self, save_path, epoch):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_data = {
            'actor_net_state_dict': self.actor.net.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'value_optim_state_dict': self.value_optim.state_dict(),
            'syn_net_state_dict': self.syn_net.state_dict(),
            'syn_net_optim_state_dict': self.syn_net_optim.state_dict(),
            'epoch': epoch
        }
        
        save_file = os.path.join(save_path, f'models_epoch{epoch}.pt')
        torch.save(save_data, save_file)
        print(f"Models and optimisers saved at {save_file}")
        return save_data

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"The specified file {load_path} does not exist.")
        
        checkpoint = torch.load(load_path, map_location=self.args.device)
        
        self.actor.net.load_state_dict(checkpoint['actor_net_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.value_optim.load_state_dict(checkpoint['value_optim_state_dict'])
        self.syn_net.load_state_dict(checkpoint['syn_net_state_dict'])
        self.syn_net_optim.load_state_dict(checkpoint['syn_net_optim_state_dict'])
        epoch = checkpoint.get('epoch', -1)
        
        self.actor.net.to(self.args.device)
        self.value_net.to(self.args.device)
        self.syn_net.to(self.args.device)

        print(f"Models and optimisers loaded from {load_path}. Resuming from epoch {epoch}")
        return epoch
