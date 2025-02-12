
import time
import pickle
import logging

from tqdm import tqdm
from rdkit import Chem

from env import RetroEnv
from config_parser import get_args
from retrorl_utils import set_seeds
from retrorl_utils import prepare_starting_molecules

from syntheseus import Molecule
from syntheseus.reaction_prediction.inference import *
from syntheseus.search.mol_inventory import SmilesListInventory

from syntheseus_paroutes import PaRoutesInventory

class ModelWrapper:
    def __init__(self, model):
        self.model = model
    
    def get_reactants(self, obs, **kwargs):
        mol = Molecule(obs)
        [results] = self.model([mol], num_results=5)
        for idx, prediction in enumerate(results):
            results[idx] = self.mols_to_str(prediction.reactants)
        reactants = results[0].split('.')
        return reactants, []
        
    def mols_to_str(self, mols) -> str:
        return ".".join([mol.smiles for mol in mols])

class SerialRunner:
    def __init__(self, args):
        self.args = args
        self.building_blocks = prepare_starting_molecules(args.building_blocks_path)
        self.agent = None
        self.env = RetroEnv(building_blocks=self.building_blocks)

    
    def run(self, target_mols, test):
        tree_list = []
        succ_subtrees_list = []
        info_list = []
        for mol in target_mols:
            try:
                tree, succ_subtrees, info = self.env.build_tree(self.agent, mol, test=test)
            except Exception as e:
                logging.warning(f'{e}')
                continue
            tree_list.append(tree)
            succ_subtrees_list.append(succ_subtrees)
            info_list.append(info)
        return tree_list, succ_subtrees_list, info_list
            
    def close(self):
        pass

    def synchronise_models(self, model_state_dict):
        self.agent.load_state_dict(model_state_dict)
        

if __name__ == '__main__':
    args = get_args()
    args.seed = set_seeds(args.seed)
    
    eval_routes = pickle.load(open(args.eval_routes, 'rb'))
    eval_mols = []
    for route in eval_routes:
        eval_mols.append(Chem.CanonSmiles(route[0].split('>')[0]))
    eval_mols = list(set(eval_mols))
    print(f"{len(eval_mols)} evaluation molecules are loaded after removing duplicates. ")
    
    runner = SerialRunner(args)
    models = {
        #'Chemformer': ChemformerModel(),
        'Graph2Edits': Graph2EditsModel(),
        #'LocalRetro': LocalRetroModel(),
        #'MEGAN': MEGANModel(),
        #'MHNreact': MHNreactModel(),
        #'RetroKNN': RetroKNNModel(),
        #'RetroAligned': RootAlignedModel(),
    }

    # Evaluate pretrained models
    for algo, model in models.items():
        runner.agent = ModelWrapper(model=model)
        start_time = time.time()
        successes = 0
        for mol in tqdm(eval_mols):
            tree_list, succ_subtrees_list, info_list = runner.run([mol], test=True)
            successes += sum([info['success'] for info in info_list])
        end_time = time.time()
        print(f'\nResults of {algo}')
        print(f'evaluation/elapsed_time: {end_time - start_time:.4f}')
        print(f'evaluation/success_rate: {successes / len(eval_mols):.4f}')
    






















