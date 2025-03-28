import numpy as np
import graph_tool.all as gt 
import logging


from policies.policy import Policy
from models.agent_info_models import STATES

class Spreader(Policy):

    """
    Info Spreader Policy.
    Seeds one node into the state I. 
    """

    def __init__(self, graph, model, quantile=0.9):
        super().__init__(graph, model)
        self.quantile = quantile

    def first_day_setup(self):

        print("Spreader policy: first day setup")

        # create graph_tool graph
        g = gt.Graph(directed=False)
        e_weight = g.new_edge_property("float")
        g.add_vertex(self.graph.num_nodes)
        for i, (source, dest) in enumerate(zip(self.graph.e_source, self.graph.e_dest)):
            e = g.add_edge(source, dest)
            e_weight[e] = self.graph.e_probs[i] * self.graph.e_intensities[i] * self.graph.layer_weights[self.graph.e_types[i]]

        # get nodes centralities 
        nodes_centralities = gt.pagerank(g, weight=e_weight)
        nodes_centralities = nodes_centralities.get_array()
        print(nodes_centralities.shape)
        print(len(nodes_centralities))
        indexes = sorted(range(len(nodes_centralities)), key=lambda k: nodes_centralities[k])        
        idx = indexes[int(len(nodes_centralities) * self.quantile) - 1]

        logging.info(f"Spreader policy: first day setup, node {idx} is selected with centrality {nodes_centralities[idx]}")
        
        self.model.change_states(self.model.nodes == idx, target_state=STATES.I)
 
    
    