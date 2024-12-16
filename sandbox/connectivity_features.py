# %%

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from cloudfiles import CloudFiles
from paleo import (
    apply_edit_sequence,
    get_metaedit_counts,
    get_metaedits,
    get_node_aliases,
    get_nucleus_supervoxel,
    map_synapses_to_sequence,
    skeletonize_sequence,
)
from paleo.io import json_to_edits, json_to_graph
from paleo_analysis import decode_synapses, subset_dict

VERSION = int(os.getenv("VERSION", 1181))
BUCKET = os.getenv("BUCKET", "allen-minnie-phase3/paleo_edits")
BUCKET = f"gs://{BUCKET}"
BUCKET = BUCKET + "/v" + str(VERSION)

client = CAVEclient("minnie65_phase3_v1", version=VERSION)
cf = CloudFiles(BUCKET)
# %%
edit_files = list(cf.list("edits"))
print(len(edit_files))
initial_graph_files = list(cf.list("initial_graphs"))
print(len(initial_graph_files))
pre_synapse_files = list(cf.list("pre_synapses"))
print(len(pre_synapse_files))
post_synapse_files = list(cf.list("post_synapses"))
print(len(post_synapse_files))

# %%

all_roots = set()
for group in [edit_files, initial_graph_files, pre_synapse_files, post_synapse_files]:
    for file_name in group:
        root_id = int(file_name.split("/")[1].split("_")[0])
        all_roots.add(root_id)

finished_roots = []
for root in all_roots:
    finished = f"post_synapses/{root}_post_synapses.csv.gz" in post_synapse_files
    finished = (
        finished and f"pre_synapses/{root}_pre_synapses.csv.gz" in pre_synapse_files
    )
    finished = (
        finished and f"initial_graphs/{root}_initial_graph.json" in initial_graph_files
    )
    finished = finished and f"edits/{root}_edits.json" in edit_files
    if finished:
        finished_roots.append(root)

print(len(finished_roots))

# %%
proofreading_table = client.materialize.query_table("proofreading_status_and_strategy")

# %%

proofreading_table.query("pt_root_id.isin(@finished_roots)", inplace=True)

# %%
proofreading_table.query(
    "strategy_axon.isin(['axon_fully_extended', 'axon_partially_extended'])",
    inplace=True,
)

# %%
root_ids = proofreading_table["pt_root_id"].unique()

cell_types = client.materialize.query_table("aibs_metamodel_celltypes_v661")
cell_types.query("pt_root_id.isin(@root_ids)", inplace=True)

root_ids = cell_types["pt_root_id"].unique()
root_ids = np.sort(root_ids)

root_id = root_ids[3]
cell_type = cell_types.set_index("pt_root_id").loc[root_id, "cell_type"]
print(cell_type)
# %%
currtime = time.time()
initial_graph = json_to_graph(
    cf.get_json(f"initial_graphs/{root_id}_initial_graph.json")
)
edits = json_to_edits(cf.get_json(f"edits/{root_id}_edits.json"))
pre_synapses = decode_synapses(
    cf.get(f"pre_synapses/{root_id}_pre_synapses.csv.gz"), side="pre"
)
post_synapses = decode_synapses(
    cf.get(f"post_synapses/{root_id}_post_synapses.csv.gz"), side="post"
)
client = CAVEclient("minnie65_public", version=VERSION)

metaedits, edit_map = get_metaedits(edits)
metaedit_counts = get_metaedit_counts(edit_map)

nuc_supervoxel_id = get_nucleus_supervoxel(root_id, client)
anchor_nodes = get_node_aliases(nuc_supervoxel_id, client, stop_layer=2)

graphs_by_sequence = {}
skeletons_by_sequence = {}
dfs = []

sequence_name = "idealized"
sequence_edits = metaedits

graphs_by_state = apply_edit_sequence(
    initial_graph,
    sequence_edits,
    anchor_nodes,
    return_graphs=True,
    remove_unchanged=True,
    include_initial=True,
)

# %%
print(len(graphs_by_state))
skeletons_by_state, mappings_by_state = skeletonize_sequence(
    graphs_by_state, root_id=root_id, client=client, remove_unchanged=True
)
print(len(skeletons_by_state))
#%%

skeletons_by_sequence[sequence_name] = skeletons_by_state

synapses_by_state = map_synapses_to_sequence(pre_synapses, graphs_by_state)

synapses_by_state = subset_dict(synapses_by_state, list(skeletons_by_state.keys()))

synapse_ids_by_state = {k: list(v.keys()) for k, v in synapses_by_state.items()}

#%%
used_metaedit_ids = list(graphs_by_state.keys())
metaedit_info = pd.DataFrame(index=used_metaedit_ids)
metaedit_info["n_edits"] = (
    metaedit_info.index.map(metaedit_counts).fillna(0).astype(int)
)
metaedit_info["cumulative_n_edits"] = metaedit_info["n_edits"].cumsum()


cf.put_json(
    f"synapse_sequences/{root_id}_{sequence_name}_sequence.json", synapse_ids_by_state
)

print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%

states = list(synapses_by_state.keys())
rows = []
for i in range(1, len(states)):
    last_state = states[i - 1]
    this_state = states[i]
    last_synapse_ids = set(synapses_by_state[last_state].keys())
    this_synapse_ids = set(synapses_by_state[this_state].keys())
    intersection_ids = last_synapse_ids.intersection(this_synapse_ids)
    added_ids = this_synapse_ids - last_synapse_ids
    removed_ids = last_synapse_ids - this_synapse_ids
    n_added = len(added_ids)
    n_removed = len(removed_ids)
    n_total = len(this_synapse_ids)

    rows.append(
        {
            "state_id": this_state,
            "n_added": n_added,
            "n_removed": n_removed,
            "n_total": n_total,
            "order": i,
        }
    )

synapse_changes = pd.DataFrame(rows).set_index("state_id")

# %%

fig, ax = plt.subplots()
sns.lineplot(data=synapse_changes, x="order", y="n_added", ax=ax)
sns.scatterplot(data=synapse_changes, x="order", y="n_added", ax=ax)
sns.lineplot(data=synapse_changes, x="order", y="n_removed", ax=ax)
sns.scatterplot(data=synapse_changes, x="order", y="n_removed", ax=ax)

# %%
cell_types.groupby("cell_type").size()

# %%
