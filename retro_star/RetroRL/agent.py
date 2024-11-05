
import torch
import torch.nn.functional as F

import numpy as np


from utils import smiles2array

class RetroRL:
    def __init__(self, one_step, topk=10, fp_dim=2048, sampling=False):
        self.fp_dim = fp_dim
        self.topk = 10
        self.sampling = sampling
        self.one_step = one_step
        
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