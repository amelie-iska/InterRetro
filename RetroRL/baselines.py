
import time
import pickle

from rdkit import Chem

from config_parser import get_args

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import LocalRetroModel

from runners import ParallelRunner, SerialRunner

class ModelWrapper:
    def __init__(self, model):
        self.model = model
    
    def get_reactants(self, obs, **kwargs):
        mol = Molecule(obs)
        [results] = self.model([mol], num_results=1)
        for idx, prediction in enumerate(results):
            results[idx] = self.mols_to_str(prediction.reactants)
        reactants = results[0].split('.')
        return reactants, []
        
    def mols_to_str(self, mols) -> str:
        return ".".join([mol.smiles for mol in mols])


if __name__ == '__main__':
    args = get_args()
    
    eval_routes = pickle.load(open(args.eval_routes, 'rb'))
    eval_mols = []
    for route in eval_routes:
        eval_mols.append(Chem.CanonSmiles(route[0].split('>')[0]))
    eval_mols = list(set(eval_mols))
    print(f"{len(eval_mols)} evaluation molecules are loaded after removing duplicates. ")
    
    runner = SerialRunner(args)
    runner.agent = ModelWrapper(model=LocalRetroModel())
    
    start_time = time.time()
    tree_list, succ_subtrees_list, info_list = runner.run(eval_mols, test=True)
    end_time = time.time()
    print({'evaluation/elapsed_time': end_time - start_time,
           'evaluation/success_rate': sum([info['success'] for info in info_list]) / len(eval_mols)})
