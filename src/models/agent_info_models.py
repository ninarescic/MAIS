import json
import numpy as np
import pandas as pd

import time
import logging

from models.simulation_engine import SimulationEngine
from utils.random_utils import RandomDuration
from utils.random_utils import gen_tuple
from utils.history_utils import TimeSeries, TransitionHistory, ShortListSeries

from utils.global_configs import monitor
import utils.global_configs as global_configs



class STATES():
    S = 0
    I = 1 
    R = 2
    EXT = 10


class InfoSIRModel(SimulationEngine):
    states = [
        STATES.S,
        STATES.I,
        STATES.R,
        STATES.EXT
    ]

    num_states = len(states)
    state_str_dict = {
        STATES.S: "S",
        STATES.I: "I",
        STATES.R: "R",
        STATES.EXT: "EXT"
    }
    ext_code = STATES.EXT

    
    transitions = [
        (STATES.S, STATES.I),
        (STATES.I, STATES.R)
    ]

    num_transitions = len(transitions)

    final_states = [
        STATES.R
    ]

    invisible_states = [
        STATES.EXT
    ]

    unstable_states = [
        STATES.I
    ]

    fixed_model_parameters = {
        "I_duration": (1, "time in the I state"),
    }

    model_parameters = {
                "beta": (0,  "rate of transmission (exposure)"),    
    }

    def inicialization(self):
        super().inicialization()
        

    def setup_series_and_time_keeping(self):

        super().setup_series_and_time_keeping()


    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        super().states_and_counts_init(ext_nodes, ext_code)

        # need_check - state that needs regular checkup
        self.need_check = self.memberships[STATES.S]

        self.update_plan(np.ones(self.num_nodes, dtype=bool))

    def prob_of_contact(self, source_state, dest_state, beta):
        """
        Evaluates if transition happens.
        Edge goes from source to dest, dest is the infected node, source is the one that can become exposed. 
        """ 

        # source_states - states that can be infected
        # dest_states - states that are infectious

        main_s = time.time()

        edges_probs = self.graph.get_all_edges_probs()
        num_edges = len(edges_probs)

        r = np.random.rand(num_edges)
        active_edges = (r < edges_probs).nonzero()[0]
        logging.info(f"active_edges {len(active_edges)}")

        source_nodes = self.graph.e_source[active_edges]
        dest_nodes = self.graph.e_dest[active_edges]
        types = self.graph.e_types[active_edges]
        # contact_info = (
        #     np.concatenate([source_nodes, dest_nodes]),
        #     np.concatenate([dest_nodes, source_nodes]),
        #     np.concatenate([types, types])
        # )
        # self.contact_history.append(contact_info)

        # take them in both directions
        n = len(active_edges)
        active_edges = np.concatenate([active_edges, active_edges])
        active_edges_dirs = np.ones(2*n, dtype=bool)
        active_edges_dirs[n:] = False

        source_nodes, dest_nodes = self.graph.get_edges_nodes(
            active_edges,
            active_edges_dirs
        )

        # is source in feasible state?
        is_relevant_source = self.memberships[source_state, source_nodes, 0]
        
        # is dest in feasible state?
        is_relevant_dest = self.memberships[dest_state, dest_nodes, 0]
        
        is_relevant_edge = np.logical_and(
            is_relevant_source,
            is_relevant_dest
        )

        ##########################
        relevant_edges = active_edges[is_relevant_edge]

        intensities = self.graph.get_edges_intensities(
            relevant_edges).reshape(-1, 1)
        relevant_sources, relevant_dests = self.graph.get_edges_nodes(
            relevant_edges, active_edges_dirs[is_relevant_edge])

        b_intensities = beta[relevant_sources]

        r = np.random.rand(intensities.ravel().shape[0]).reshape(-1, 1)
        is_exposed = r < (b_intensities * intensities)

        
        if np.all(is_exposed == False):
            return np.zeros((self.num_nodes, 1))

        is_exposed = is_exposed.ravel()
        
        exposed_nodes = relevant_sources[is_exposed]
        
        print(exposed_nodes)
        
        ret = np.zeros((self.num_nodes, 1))
        ret[exposed_nodes] = 1

        main_e = time.time()
        logging.info(f"PROBS OF CONTACT {main_e - main_s}")
        return ret


    def daily_update(self, nodes):
        """
        Everyday checkup
        """

        # S
        target_nodes = self._get_target_nodes(nodes, STATES.S)      

        # if we have external nodes
        if self.num_ext_nodes > 0:
            raise NotImplementedError("External nodes not implemented yet.")    

        # try infection 
        P_infection = self.prob_of_contact(
                                      STATES.S,
                                      STATES.I,  
                                      self.beta
                                      ).flatten()

        exposed = P_infection[target_nodes]
        
        exposed_mask = np.zeros(self.num_nodes, dtype=bool)
        exposed_mask[target_nodes] = (exposed == 1)

        print("Počet nakažených",  exposed_mask.sum())
        if exposed_mask.sum() > 0:
            print(exposed_mask)
            
        self.time_to_go[exposed_mask] = 1
        self.state_to_go[exposed_mask] = STATES.I


    def update_plan(self, nodes):
        """ This is done for nodes that  just changed their states.
        New plans are generated according the state."""

        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, STATES.S)
        
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = STATES.S
        self.need_check[target_nodes] = True

        # STATES.I:   "I"
        target_nodes = self._get_target_nodes(nodes, STATES.I)
        self.time_to_go[target_nodes] = self.I_duration 
        self.state_to_go[target_nodes] = STATES.R
        self.need_check[target_nodes] = False

        # STATES.R:   "R",
        target_nodes = self._get_target_nodes(nodes, STATES.R)
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False



    def run_iteration(self):
        super().run_iteration()

 
              
  