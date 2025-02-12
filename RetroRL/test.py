
from models import ValueGNN
import torch

if __name__ == "__main__":
    model = ValueGNN(node_dim=21, edge_dim=4, hidden_dim=64, num_layers=3)
    model.eval()

    # Suppose we have a small batch of SMILES
    smiles_batch = ["CCO", "CCN", "CCC", "CC(=O)O", "c1ccccc1"]

    with torch.no_grad():
        values = model(smiles_batch)  # shape [5, 1]
    print("Predicted values:", values.squeeze(-1))
