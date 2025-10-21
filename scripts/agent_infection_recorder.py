# scripts/agent_infection_recorder.py
from pathlib import Path

class AgentInfectionRecorder:
    """
    Tracks the first time each node enters state 'I' and writes a NetRate-ready CSV:
      cascade_id,node_id,infection_time
    Works with two common patterns:
      - model.get_nodes_in_state('I')
      - networkx graph node attribute 'state' == 'I'
    """

    def __init__(self, cascade_id: str = "0"):
        self.cascade_id = str(cascade_id)
        self.first_time_I = {}   # node_id -> time
        self._prev_I = set()

    def _current_I(self, model):
        # Try a direct API if the model exposes it
        if hasattr(model, "get_nodes_in_state"):
            try:
                return set(map(str, model.get_nodes_in_state("I")))
            except Exception:
                pass

        # Fallback: try reading node attributes from model.G or model.graph
        for g_attr in ("G", "graph", "network"):
            G = getattr(model, g_attr, None)
            if G is not None and hasattr(G, "nodes"):
                try:
                    return {str(n) for n, data in G.nodes(data=True)
                            if str(data.get("state", "")).upper() == "I"}
                except Exception:
                    continue

        # As a last resort, if the model exposes a dataframe with states at time t,
        # the caller can pass that list directly via update(..., infected_ids=...)
        return None

    def update(self, model, t: float, infected_ids=None):
        """Call every step. If you already know infected ids, pass infected_ids=list_of_ids."""
        I_now = set(map(str, infected_ids)) if infected_ids is not None else self._current_I(model)
        if I_now is None:
            # Could not read states this step; do nothing
            return

        # newly infected = currently I minus those that were I at previous step
        new_I = I_now - self._prev_I
        for v in new_I:
            self.first_time_I.setdefault(str(v), float(t))
        self._prev_I = I_now

    def dump_netrate_csv(self, out_csv: Path):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("cascade_id,node_id,infection_time\n")
            for v, tt in self.first_time_I.items():
                f.write(f"{self.cascade_id},{v},{tt}\n")
        return out_csv
