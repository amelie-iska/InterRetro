
def eval(envs, agent):
    eval_info = {}
    for env in envs:
        rewards = np.zeros((len(env.target_mols), 1))
        for mol_idx in range(min(len(env.target_mols), 200)):
            obs = env.reset(mol_idx)
            for t in range(100):
                a, _ = agent.get_action(obs)
                next_obs, reward, done, info = env.step(a)
                rewards[mol_idx] = reward
                if done:
                    break
                obs = next_obs
        eval_info[env.mode+'/success rate'] = rewards.mean()
    return eval_info


class RetroEnv:
    def __init__(self, mode='training'):
        self.mode = mode
        if mode == 'training':
            self.routes = pickle.load(open('dataset/routes_train.pkl', 'rb'))
            self.target_mols = [Chem.CanonSmiles(self.route[0].split('>')[0]) for self.route in tqdm(self.routes[:200], mininterval=2, desc='Load target molecules')]
        elif mode == 'evaluation':
            self.routes = pickle.load(open('dataset/routes_possible_test_hard.pkl', 'rb'))
            self.target_mols = [Chem.CanonSmiles(self.route[0].split('>')[0]) for self.route in tqdm(self.routes, mininterval=2, desc='Load target molecules')]
        self.building_blocks = prepare_starting_molecules('dataset/origin_dict_canonical.csv')
    
    def reset(self, mol_idx=None):
        self.tree_mdp = None
        self.tobevisited_nodes = []

        if mol_idx == None:
            self.obs = random.choice(self.target_mols)
        else:
            self.obs = self.target_mols[mol_idx]
            
        self.obs_node = AnyNode(id=self.obs, action=None, inbb=int(self.obs in self.building_blocks))
        self.tree_mdp = self.obs_node
        return self.obs

    def step(self, a):
        self.obs_node.action = a
        reactants = rdchiralRunText(a, self.obs)
        if len(reactants) == 0:
            return None, 0, 1, None
        else:
            reactants = list(set(reactants[0].split('.')))
        
        for r in reactants:
            r_node = AnyNode(id=r, parent=self.obs_node, action=None, inbb=int(r in self.building_blocks))
            if not r in self.building_blocks:
                self.tobevisited_nodes.append(r_node)
        
        if self.tobevisited_nodes:
            self.obs = self.tobevisited_nodes[-1].id
            self.obs_node = self.tobevisited_nodes[-1]
            self.tobevisited_nodes.remove(self.tobevisited_nodes[-1])
            return self.obs, 0, 0, None
        else:
            self.obs = None
            self.obs_node = None
            return None, 1, 1, None