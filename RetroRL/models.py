import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
# or GCNConv, GraphConv, etc.

class ValueGNN(nn.Module):
    """
    A model that handles a *batch* of SMILES at once.
    1) Each SMILES -> subgraph
    2) We combine them into a single PyG Batch
    3) GNN forward pass
    4) Global pooling => [batch_size, hidden_dim]
    5) MLP => [batch_size, 1] regression output
    """

    def __init__(self,
                 node_dim: int = 21,   # dimension from featurize_atom
                 edge_dim: int = 4,    # dimension from featurize_bond
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 mlp_hidden_dim: int = 128,
                 dropout: float = 0.1):
        super(ValueGNN, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build GIN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # GIN needs an MLP for "update"
            gin_mlp = nn.Sequential(
                nn.Linear(hidden_dim if i>0 else node_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINConv(nn=gin_mlp)
            self.convs.append(conv)

        # Final feed-forward
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)  # single scalar
        )

    def forward(self, smiles_list):
        """
        smiles_list: List[str] of length B (batch size)
        Returns: tensor of shape [B, 1] with regression values.
        """
        # 1) Convert SMILES -> PyG Batch
        batch_data = smiles_to_pyg_batch(smiles_list)  # merges them
        device = next(self.parameters()).device
        batch_data = batch_data.to(device)

        # 2) GNN forward
        x, edge_index, edge_attr, batch = (batch_data.x,
                                           batch_data.edge_index,
                                           batch_data.edge_attr,
                                           batch_data.batch)

        # x: [total_nodes, node_dim]
        # batch: [total_nodes], each entry is which graph in the batch the node belongs to

        h = x
        for i, conv_layer in enumerate(self.convs):
            h = conv_layer(h, edge_index)  # ignoring edge_attr for simplicity
            h = F.relu(h)

        # 3) Global Pooling => [batch_size, hidden_dim]
        graph_embed = global_mean_pool(h, batch)

        # 4) MLP => [batch_size, 1]
        out = self.head(graph_embed)
        return out  # shape [batch_size, 1]


def featurize_atom(atom: Chem.Atom) -> torch.Tensor:
    """
    Example: one-hot atomic number up to 20 + is_aromatic => 21D (or 22D).
    Adjust as you see fit.
    """
    max_num = 20
    atomic_num = atom.GetAtomicNum()
    atomic_num = min(atomic_num, max_num)
    one_hot = [0]*max_num
    if atomic_num > 0:
        one_hot[atomic_num-1] = 1
    is_aromatic = 1 if atom.GetIsAromatic() else 0
    feats = torch.tensor(one_hot + [is_aromatic], dtype=torch.float)
    return feats

def featurize_bond(bond: Chem.Bond) -> torch.Tensor:
    """
    4D bond feature: single/double/triple/aromatic.
    """
    from rdkit import Chem
    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return torch.tensor([1, 0, 0, 0], dtype=torch.float)
    elif bt == Chem.BondType.DOUBLE:
        return torch.tensor([0, 1, 0, 0], dtype=torch.float)
    elif bt == Chem.BondType.TRIPLE:
        return torch.tensor([0, 0, 1, 0], dtype=torch.float)
    elif bond.GetIsAromatic():
        return torch.tensor([0, 0, 0, 1], dtype=torch.float)
    else:
        return torch.tensor([1, 0, 0, 0], dtype=torch.float)  # fallback

def mol_to_pyg_data(mol: Chem.Mol) -> Data:
    """
    Convert an RDKit Mol into a PyG Data object with:
      - x: node features (shape [num_atoms, node_dim])
      - edge_index: [2, num_edges]
      - edge_attr: [num_edges, edge_dim]
    """
    # Build node features
    atom_feats_list = [featurize_atom(atom) for atom in mol.GetAtoms()]
    if len(atom_feats_list) == 0:
        # Empty => return dummy
        return Data(x=torch.zeros((0,)), edge_index=torch.zeros((2,0),dtype=torch.long))
    x = torch.stack(atom_feats_list, dim=0)  # [num_atoms, node_dim]

    # Build edges
    edges = []
    edge_feats_list = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        edges.append((a1, a2))
        edges.append((a2, a1))
        bf = featurize_bond(bond)
        edge_feats_list.append(bf)
        edge_feats_list.append(bf)
    if len(edges) == 0:
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_attr = torch.zeros((0,4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t()  # shape [2, E]
        edge_attr = torch.stack(edge_feats_list, dim=0)         # shape [E, 4]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def smiles_to_pyg_batch(smiles_list):
    """
    Takes a list of SMILES and returns a PyG Batch object containing
    all their graphs. Each element in the list becomes one subgraph
    in the batch.
    """
    from rdkit import Chem

    data_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Invalid => produce empty graph or skip
            data = Data(x=torch.zeros((0,)), edge_index=torch.zeros((2,0),dtype=torch.long))
        else:
            data = mol_to_pyg_data(mol)
        data_list.append(data)

    # Merge into a single Batch
    batch = Batch.from_data_list(data_list)
    return batch
