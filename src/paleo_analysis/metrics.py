import numpy as np
import pandas as pd


def compute_precision_recall(synapses_by_state: dict):
    states = list(synapses_by_state.keys())
    final_synapse_map = synapses_by_state[states[-1]]
    final_synapse_ids = np.unique(list(final_synapse_map.keys()))
    rows = []
    for state_id, synapse_map in synapses_by_state.items():
        synapse_ids = np.unique(list(synapse_map.keys()))
        n_intersection = len(np.intersect1d(final_synapse_ids, synapse_ids))
        if len(synapse_ids) == 0:
            precision = np.nan
        else:
            precision = n_intersection / len(synapse_ids)
        if len(final_synapse_ids) == 0:
            recall = np.nan
        else:
            recall = n_intersection / len(final_synapse_ids)
        rows.append({"state_id": state_id, "precision": precision, "recall": recall})

    results = pd.DataFrame(rows).set_index("state_id")
    return results
