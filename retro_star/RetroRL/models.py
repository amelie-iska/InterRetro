
import torch
import torch.nn as nn
import torch.nn.functional as F


class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, fp_dim=2048, dim=512, dropout_rate=0.3):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim,dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        # self.fc2 = nn.Linear(dim,dim)
        # self.bn2 = nn.BatchNorm1d(dim)
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim,n_rules)

    def forward(self,x, y=None, loss_fn =nn.CrossEntropyLoss()):
        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        # x = self.dropout1(F.elu(self.fc1(x)))
        # x = self.dropout2(F.elu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        if y is not None :
            return loss_fn(x, y)
        else :
            return x
        return x

def load_model(state_path, template_rule_path,fp_dim=2048):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in enumerate(f):
            rule= l.strip()
            template_rules[rule] = i
    idx2rule = {}
    for rule, idx in template_rules.items():
        idx2rule[idx] = rule
    rollout = RolloutPolicyNet(len(template_rules),fp_dim=fp_dim)
    
    state_dict = torch.load(state_path, map_location='cpu')
    # Remove the 'module.' prefix if present (i.e., if it was saved using DataParallel)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = state_dict[key]
    rollout.load_state_dict(new_state_dict)
    return rollout, idx2rule, template_rules
    
class MLPModel(object):
    def __init__(self, state_path, template_path, device=-1, fp_dim=2048, realistic_filter=False):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules, self.rules2idx = load_model(state_path, template_path, fp_dim)
        self.net.eval()
        self.device = device
        if device >= 0:
            self.net.to(device)

        self.realistic_filter = realistic_filter
        
        self.reference_net, _, _ = load_model(state_path, template_path, fp_dim)
        self.reference_net.eval()
        if device >= 0:
            self.reference_net.to(device)

    def forward_topk(self, arr, topk=10):
        preds = self.net(arr)
        preds_reference = self.reference_net(arr).detach()

        if self.realistic_filter:
            preds_reference, idx_topk = torch.topk(preds_reference, k=topk)
            preds = preds.gather(1, idx_topk)
        else:
            preds, idx_topk = torch.topk(preds, k=topk)
            preds_reference = preds_reference.gather(1, idx_topk)

        return preds, preds_reference, idx_topk


class ActionRepresentation(nn.Module):
    def __init__(self, state_dim, ac_dim, device):
        super(ActionRepresentation, self).__init__()
        self.model = nn.Sequential(nn.Linear(state_dim, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, ac_dim))
        self.model.to(device=device)
        
    def forward(self, rxn):
        return self.model(rxn)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        