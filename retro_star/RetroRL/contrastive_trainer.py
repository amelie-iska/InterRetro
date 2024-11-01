import torch
import torch.nn.functional as F

class ContrastiveTrainer:
    def __init__(self, model, temperature=0.1, device='cpu'):
        """
        Initializes the trainer for contrastive learning.
        
        Args:
        - model (MLPModel): The one-step model instance.
        - temperature (float): Temperature parameter for InfoNCE loss.
        - device (str): Device to use for training (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.temperature = temperature
        self.device = device
        self.model.net.to(self.device)

    def compute_similarity(self, z_i, z_j):
        """Calculates cosine similarity between two tensors."""
        return F.cosine_similarity(z_i, z_j)

    def sample_negatives(self, all_templates, N):
        """Samples N random templates for negative examples."""
        neg_idxs = torch.randint(len(all_templates), (N,))
        negatives = [all_templates[i] for i in neg_idxs]
        return negatives

    def info_nce_loss(self, anchor, positive, negatives):
        """Calculates InfoNCE loss using anchor, positive, and negative samples."""
        # Calculate similarities
        pos_sim = self.compute_similarity(anchor, positive) / self.temperature
        neg_sims = torch.cat([
            self.compute_similarity(anchor, neg).unsqueeze(0)
            for neg in negatives
        ]) / self.temperature

        # Compute the InfoNCE loss
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sims).sum()))
        return loss

    def train_step(self, molecule, all_templates, N=10):
        """
        Performs a single training step for contrastive learning.
        
        Args:
        - molecule: Molecule representation (e.g., as a tensor).
        - all_templates: List of all reaction templates in the dataset.
        - N (int): Number of negatives to sample.
        
        Returns:
        - loss: Computed InfoNCE loss for the step.
        """
        # Generate positive pair
        anchor_template, positive_template = self.sample_positive_pair(molecule)
        
        # Sample negative examples
        negatives = self.sample_negatives(all_templates, N)

        # Pass through the model to get latent representations
        anchor = self.model.net(anchor_template)
        positive = self.model.net(positive_template)
        negatives_latent = [self.model.net(neg) for neg in negatives]

        # Compute InfoNCE loss
        loss = self.info_nce_loss(anchor, positive, negatives_latent)
        return loss

    def sample_positive_pair(self, molecule):
        """
        Placeholder: Given a molecule, sample two feasible reaction templates as anchor and positive.
        
        Replace with actual logic to sample based on your data structure.
        """
        # Assuming you have a way to sample two templates that work for `molecule`
        anchor_template = torch.rand((self.model.fp_dim,)).to(self.device)
        positive_template = torch.rand((self.model.fp_dim,)).to(self.device)
        return anchor_template, positive_template


