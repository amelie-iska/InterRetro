import os
import numpy as np

from alg import online_mcts
from models import MLPModel, ValueMLP, PriorMLP
from retrorl_utils import prepare_starting_molecules, smiles2array

import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
from multiprocessing import Queue
from time import sleep
from viz_tree.mol_tree_to_paroutes_dict import mol_tree_to_paroutes_dict

from runners.serial_runner import SerialRunner

mp.set_start_method('spawn', force=True)



def worker(q1, remote, args, gpu=-1):
    if gpu > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    local_runner = SerialRunner(args)
    while True:
        cmd = remote.recv()
        if type(cmd) != str: 
            assert len(cmd) == 2  # for load model mode.
            cmd, data = cmd[0], cmd[1]

        if cmd == "load":
            local_runner.load_model(*data)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd in ["sample", "test", "plan", "retro", "mcts", "mcts_test"]:
            results = []
            # while not q1.empty(): # sometimes get stuck here.
            while True:
                element = q1.get()
                if element == 'DONE': # stop element
                    break
                target_mols, target_mol_ids = [element[0]], [element[1]]
                if cmd == 'sample':
                    mols_batch, vals_batch, tmps_batch, batch_result = local_runner.run(target_mols, target_mol_ids)
                    results.append([mols_batch, vals_batch, tmps_batch, batch_result])
                elif cmd == 'test':
                    mols_batch, vals_batch, tmps_batch, batch_result = local_runner.run(target_mols, target_mol_ids, test=True)
                    results.append([mols_batch, vals_batch, tmps_batch, batch_result])
                elif cmd == 'retro':
                    batch_result = local_runner.retro(target_mols, target_mol_ids)
                    results.append([batch_result])
                elif cmd == 'plan':
                    mols_batch, vals_batch, tmps_batch, batch_result = local_runner.plan(target_mols, target_mol_ids)
                    results.append([mols_batch, vals_batch, tmps_batch, batch_result])
                elif cmd == 'mcts' or cmd == 'mcts_test':
                    mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result = local_runner.mcts(target_mols, target_mol_ids, test=(cmd=='mcts_test'))
                    results.append([mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result])
            remote.send(results) # send data to continue main process. 
        else:
            raise NotImplementedError


class ParallelRunner:
    def __init__(self, args):
        self.args = args
        self.n_gpus = args.n_gpus
        self.n_processes = args.n_processes
        self.parent_conns, self.worker_conns = zip(*[mp.Pipe() for _ in range(self.n_processes)])

        self.manager = Manager()
        self.q1 = self.manager.Queue() # use q1 to pass the batch target mols from main process to workers. 
        # directly use Pipe to return batch results from workers to main process.
        self.ps = [mp.Process(target=worker, args=(self.q1, self.worker_conns[i], args, i%self.n_gpus if self.n_gpus else -1,)) for i in range(self.n_processes)]
        
        for p in self.ps:
            p.daemon = True
            p.start()

    def run(self, target_mols, target_mol_ids, test=False):
        # 1. put all the target mols and their id into the shared queue.
        for target_mol, target_mol_id in zip(target_mols, target_mol_ids):
            self.q1.put((target_mol, target_mol_id))

        for i in range(0, self.n_processes):
            self.q1.put('DONE')

        # assert self.q1.qsize() == (self.len(target_mols) + self.n_processes)  # from exp, sometimes will violate. delay in mp.Queue.put() in main process.

        # 2. send message to each worker process to let them start working.
        for i in range(self.n_processes):
            if test:
                self.parent_conns[i].send("test")
            else:
                self.parent_conns[i].send("sample")

        # 3. prepare container to save the results.
        batch_result = {
            'ids': [],
            'succ': [],
            'iter': [],
            'route_costs': [],
            'route_lens': []
        }
        mols_batch, vals_batch, tmps_batch = [], [], []

        # 4. wait until each worker process passes data.
        for i in range(self.n_processes):
            results = self.parent_conns[i].recv()
            for result in results:
                mols_batch += result[0]
                vals_batch += result[1]
                tmps_batch += result[2]        
                for key in batch_result.keys():
                    batch_result[key] += (result[3][key])
                    
        return mols_batch, vals_batch, tmps_batch, batch_result

    def mcts(self, target_mols, target_mol_ids, test=False):
        # 1. put all the target mols and their id into the shared queue.
        for target_mol, target_mol_id in zip(target_mols, target_mol_ids):
            self.q1.put((target_mol, target_mol_id))

        for i in range(0, self.n_processes):
            self.q1.put('DONE')

        assert self.q1.qsize() == (len(target_mols) + self.n_processes)

        # 2. send message to each worker process to let them start working.
        for i in range(self.n_processes):
            if test:
                self.parent_conns[i].send("mcts_test")
            else:
                self.parent_conns[i].send("mcts")

        # 3. prepare container to save the results.
        batch_result = {
            "ids": [],
            "succ": [],
            "simulate_succ": [],
            "iter": [],
            "route_lens": [],
            "num_succ_before_take": 0,
            "num_take_unsucc_when_succ": 0,
            "num_deadends": 0,
            "num_building_blocks": 0,
            "num_unexpanded": 0,
            "route_shortest_cost": [],
            "mol_nodes": [],
            "reaction_nodes": [],
            "nll_reference": [],
            "nll": [],
            "succ_len": [],
        }
        mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch = [], [], [], [], [], []

        # 4. wait until each worker process passes data.
        for i in range(self.n_processes):
            results = self.parent_conns[i].recv()
            for result in results:
                mols_r_batch += result[0]
                pris_batch += result[1]
                mols_v_batch += result[2]
                vals_batch += result[3]
                mols_t_batch += result[4]
                tmps_batch += result[5]
                for key in batch_result.keys():
                    batch_result[key] += (result[-1][key])

        return mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result

    def retro(self, target_mols, target_mol_ids):
        # 1. put all the target mols and their id into the shared queue.
        for target_mol, target_mol_id in zip(target_mols, target_mol_ids):
            self.q1.put((target_mol, target_mol_id))

        for i in range(0, self.n_processes):
            self.q1.put('DONE')

        assert self.q1.qsize() == (len(target_mols) + self.n_processes)

        # 2. send message to each worker process to let them start working.
        for i in range(self.n_processes):
            self.parent_conns[i].send("retro")

        # 3. prepare container to save the results.
        batch_result = {
            "ids": [],
            "succ": [],
            "iter": [],
            "mol_nodes": [],
            "reaction_nodes": [],
            "nll_reference": [], # negative log likelihood given by the reference model
            "nll": [], # negative log likelihood given by the trained model
            "succ_len": [],
        }

        # 4. wait until each worker process passes data.
        for i in range(self.n_processes):
            results = self.parent_conns[i].recv()
            for result in results:
                for key in batch_result.keys():
                    batch_result[key] += (result[-1][key])

        return batch_result

    def plan(self, target_mols, target_mol_ids):
        # 1. put all the target mols and their id into the shared queue.
        for target_mol, target_mol_id in zip(target_mols, target_mol_ids):
            self.q1.put((target_mol, target_mol_id))

        for i in range(0, self.n_processes):
            self.q1.put('DONE')

        assert self.q1.qsize() == (self.len(target_mols) + self.n_processes)

        # 2. send message to each worker process to let them start working.
        for i in range(self.n_processes):
            self.parent_conns[i].send("plan")

        batch_result = {
            'ids': [],
            'succ': [],
            'simulate_succ': [],
        }
        mols_batch, vals_batch, tmps_batch = [], [], []

        # 4. wait until each worker process passes data.
        for i in range(self.n_processes):
            results = self.parent_conns[i].recv()
            for result in results:
                mols_batch += result[0]
                vals_batch += result[1]
                tmps_batch += result[2]        
                for key in batch_result.keys():
                    batch_result[key] += (result[3][key])

        return mols_batch, vals_batch, tmps_batch, batch_result

    def load_model(self, model_state_dict, prior_state_dict, value_state_dict=None): # TODO
        for i in range(self.n_processes):
            if value_state_dict:
                self.parent_conns[i].send(("load", (model_state_dict, prior_state_dict, value_state_dict)))
            else:
                self.parent_conns[i].send(("load", (model_state_dict, )))

    def close(self):
        for i in range(self.n_processes):
            # self.parent_conns[i].send(('close', None))
            self.parent_conns[i].send('close')