
import pickle
from rdkit import Chem, RDLogger

from retrorl_utils import prepare_starting_molecules

from tqdm import tqdm

from syntheseus import Molecule

from syntheseus.search.algorithms.best_first import retro_star
from syntheseus.search.graph.and_or import AndNode, OrNode, AndOrGraph
from syntheseus.search.node_evaluation.common import ReactionModelLogProbCost

from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.reaction_prediction.inference import (
    LocalRetroModel, ChemformerModel, Graph2EditsModel
)


from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.search.algorithms.breadth_first import (
    AndOr_BreadthFirstSearch
)
from syntheseus.search.analysis.route_extraction import (
    iter_routes_time_order,
)


RXN_MODEL_CALL_LIMIT = 100
TIME_LIMIT_S = 500
building_blocks_path = "../dataset/origin_dict_canonical.csv"
eval_path = '../dataset/routes_possible_test_hard.pkl'


eval_routes = pickle.load(open(eval_path, 'rb'))
eval_mols = []
for route in eval_routes:
    eval_mols.append(Chem.CanonSmiles(route[0].split('>')[0]))
del eval_routes
eval_mols = list(set(eval_mols))

inventory_list = list(prepare_starting_molecules(building_blocks_path))
inventory = SmilesListInventory(inventory_list, canonicalize=False)
del inventory_list

reaction_model = Graph2EditsModel(use_cache=False, default_num_results=50)

or_node_cost_fn = retro_star.MolIsPurchasableCost()
and_node_cost_fn = ReactionModelLogProbCost(normalize=False)
retro_star_value_function = ConstantNodeEvaluator(0.0)

retro_star_alg = retro_star.RetroStarSearch(
    reaction_model=reaction_model,
    mol_inventory=inventory,
    or_node_cost_fn=or_node_cost_fn,
    and_node_cost_fn=and_node_cost_fn,
    value_function=retro_star_value_function,
    limit_reaction_model_calls=RXN_MODEL_CALL_LIMIT,
    time_limit_s=TIME_LIMIT_S,
)

successes = 0
for mol in tqdm(eval_mols):
    target_mol = Molecule(mol)
    retro_star_alg.reset()
    output_graph, b = retro_star_alg.run_from_mol(target_mol)
    
    routes = list(iter_routes_time_order(output_graph, max_routes=1))
    if len(routes) > 0:
        successes = successes + 1
        print(successes)


