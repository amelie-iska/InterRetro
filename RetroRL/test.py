from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel

test_mol = Molecule("Cc1ccc(-c2ccc(C)cc2)cc1")
model = LocalRetroModel()

def mols_to_str(mols) -> str:
    return " + ".join([mol.smiles for mol in mols])

def print_results(results) -> None:
    for idx, prediction in enumerate(results):
        print(f"{idx + 1}: " + mols_to_str(prediction.reactants))

from syntheseus.reaction_prediction.inference import *

models = [
    ChemformerModel(),
    Graph2EditsModel(),
    LocalRetroModel(),
    MEGANModel(),
    MHNreactModel(),
    RetroKNNModel(),
    RootAlignedModel(),
]

for model in models:
    # When interested in very few predictions (e.g. one), it may be
    # useful to set `num_results > 1`, as this will cause e.g.
    # larger beam size for models based on beam search.
    [results] = model([test_mol], num_results=5)

    top_prediction = results[0].reactants
    print(f"{model.name + ':':12} {mols_to_str(top_prediction)}")