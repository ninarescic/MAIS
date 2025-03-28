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
    }

    def inicialization(self):
        super().inicialization()
        

    def setup_series_and_time_keeping(self):

        super().setup_series_and_time_keeping()


    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        super().states_and_counts_init(ext_nodes, ext_code)

        # need_check - state that needs regular checkup
        self.need_check = self.memberships[STATES.S],

        self.update_plan(np.ones(self.num_nodes, dtype=bool))

    def prob_of_contact(self, state_from, state_to, beta):
        ...

    def daily_update(self, nodes):
        """
        Everyday checkup
        """

        # S
        target_nodes = self._get_target_nodes(nodes, STATES.S)

        # if we have external nodes
        if self.num_ext_nodes > 0:
            raise NotImplementedError("External nodes not implemented yet.")    

        # try infection (may rewrite S/Ss moves)
        P_infection = prob_of_contact(self,
                                      STATE.S,
                                      STATE.I,  
                                      self.beta
                                      ).flatten()

        exposed = P_infection[target_nodes]
        
        exposed_mask = np.zeros(self.num_nodes, dtype=bool)
        exposed_mask[target_nodes] = exposed

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

                
  