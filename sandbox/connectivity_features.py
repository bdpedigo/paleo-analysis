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
    check_graph_changes,
    get_level2_data,
    get_level2_spatial_graphs,
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
level2_data = get_level2_data(graphs_by_state, client)
spatial_graphs_by_state = get_level2_spatial_graphs(graphs_by_state, client)
is_new_level2_state = check_graph_changes(spatial_graphs_by_state)

skeletons_by_state, mappings_by_state = skeletonize_sequence(
    graphs_by_state,
    client=client,
    root_id=root_id,
    remove_unchanged=False,
    level2_data=level2_data,
)


#  73: True,
#  74: False,
#  75: False,

# %%
from morphsync import MorphSync


# %%
def graph_to_arrays(graph):
    graph = graph.copy()
    nodes = np.array(list(graph.nodes()))
    nodes = level2_data.loc[
        nodes, ["rep_coord_nm_x", "rep_coord_nm_y", "rep_coord_nm_z"]
    ]
    edges = np.array(list(graph.edges()))
    edges = pd.DataFrame(edges, columns=["source", "target"])
    return nodes, edges


graph_to_arrays(graphs_by_state[73])

morphology1 = MorphSync()
morphology1.add_graph(graph_to_arrays(graphs_by_state[73]), "level2")
level2_poly1 = morphology1.level2.to_pyvista()

morphology1.add_graph(skeletons_by_state[43], "skeleton")
skeleton_poly1 = morphology1.skeleton.to_pyvista()

morphology2 = MorphSync()
morphology2.add_graph(graph_to_arrays(graphs_by_state[74]), "level2")
level2_poly2 = morphology2.level2.to_pyvista()

morphology2.add_graph(skeletons_by_state[44], "skeleton")
skeleton_poly2 = morphology2.skeleton.to_pyvista()

# %%
metaedit_counts = pd.Series(edit_map).value_counts()

# %%
member_edits = edit_map_series[edit_map_series == 74].index

# %%
from paleo import get_detailed_change_log

change_log = get_detailed_change_log(root_id, client, filtered=False)

# %%
coords = np.stack(change_log.loc[member_edits]["source_coords"].values).reshape(-1, 3)
edit_centroid = coords.mean(axis=0)
edit_centroid = edit_centroid * np.array([8, 8, 40])
print(edit_centroid)

# %%
import pyvista as pv

pv.set_jupyter_backend("client")

plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
plotter.add_mesh(level2_poly1, color="lightgrey")
plotter.add_mesh(skeleton_poly1, color="red", line_width=5)

plotter.subplot(0, 1)
plotter.add_mesh(level2_poly2, color="lightgrey")
plotter.add_mesh(skeleton_poly2, color="red", line_width=5)

plotter.link_views()

plotter.camera.focal_point = edit_centroid
plotter.camera.position = edit_centroid + np.array([10000, 0, 0])
plotter.camera.view_up = [0, -1, 0]

plotter.show()
# %%

# skeletons_by_sequence[sequence_name] = skeletons_by_state

relevant_states = [k for k, v in is_new_level2_state.items() if v]

# %%

synapses_by_state = map_synapses_to_sequence(pre_synapses, graphs_by_state)

synapses_by_state = subset_dict(synapses_by_state, relevant_states)

synapse_ids_by_state = {k: list(v.keys()) for k, v in synapses_by_state.items()}

# %%

state_info = pd.DataFrame(index=relevant_states)
if sequence_name == "idealized":
    state_info["n_edits"] = state_info.index.map(metaedit_counts).fillna(0).astype(int)
elif sequence_name == "historical":
    state_info["n_edits"] = np.ones(len(state_info), dtype=int)
state_info["cumulative_n_edits"] = state_info["n_edits"].cumsum()


# %%


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

pre_synapse_sequence_files = list(cf.list("pre_synapse_sequences"))
pre_synapse_sequence_files = [f for f in pre_synapse_sequence_files if "idealized" in f]
pre_root_ids = [int(f.split("/")[1].split("_")[0]) for f in pre_synapse_sequence_files]

post_synapse_sequence_files = list(cf.list("post_synapse_sequences"))
post_synapse_sequence_files = [
    f for f in post_synapse_sequence_files if "idealized" in f
]
post_root_ids = [
    int(f.split("/")[1].split("_")[0]) for f in post_synapse_sequence_files
]

info_files = list(cf.list("state_info"))
info_files = [f for f in info_files if "idealized" in f]
info_root_ids = [int(f.split("/")[1].split("_")[0]) for f in info_files]

root_ids = np.intersect1d(pre_root_ids, post_root_ids)
root_ids = np.intersect1d(root_ids, info_root_ids)


def key_by_int(dict_):
    return {int(k): v for k, v in dict_.items()}


rows = []
for root_id in root_ids:
    for side in ["pre", "post"]:
        file = f"{side}_synapse_sequences/{root_id}_idealized_sequence.json"
        synapse_ids_by_state = key_by_int(cf.get_json(file))
        states = list(synapse_ids_by_state.keys())
        for i in range(1, len(states)):
            last_state = states[i - 1]
            this_state = states[i]
            last_synapse_ids = set(synapse_ids_by_state[last_state])
            this_synapse_ids = set(synapse_ids_by_state[this_state])
            intersection_ids = last_synapse_ids.intersection(this_synapse_ids)
            added_ids = this_synapse_ids - last_synapse_ids
            removed_ids = last_synapse_ids - this_synapse_ids
            n_added = len(added_ids)
            n_removed = len(removed_ids)
            n_total = len(this_synapse_ids)

            rows.append(
                {
                    "root_id": root_id,
                    "state_id": this_state,
                    "n_added": n_added,
                    "n_removed": n_removed,
                    "n_total": n_total,
                    "side": side,
                    "order": i,
                }
            )

synapse_changes = pd.DataFrame(rows).set_index(["root_id", "state_id", "side"])

# %%
fig, ax = plt.subplots()

sns.scatterplot(
    data=synapse_changes.query("side=='post'"), x="order", y="n_added", ax=ax
)
sns.scatterplot(
    data=synapse_changes.query("side=='post'"), x="order", y="n_removed", ax=ax
)
sns.scatterplot(
    data=synapse_changes.query("side=='pre'"), x="order", y="n_added", ax=ax
)
sns.scatterplot(
    data=synapse_changes.query("side=='pre'"), x="order", y="n_removed", ax=ax
)

# %%
