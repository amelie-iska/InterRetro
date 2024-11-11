from torch.utils.data import Dataset, DataLoader


def create_molecule_dataloader(molecule_dataset, batch_size, shuffle):
    molecule_dataset = MoleculeDataset(molecule_dataset)
    molecule_loader = DataLoader(molecule_dataset, batch_size=batch_size, shuffle=shuffle)
    return molecule_loader

class MoleculeDataset(Dataset):
    def __init__(self, molecules):
        self.molecules = molecules

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        return self.molecules[idx]