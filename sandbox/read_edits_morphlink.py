# %%

import os
import time
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from caveclient import CAVEclient
from cloudfiles import CloudFiles
from morphsync import MorphSync
from paleo import (
    apply_edit_sequence,
    check_skeleton_changes,
    compare_skeletons,
    get_metaedits,
    get_node_aliases,
    get_nucleus_supervoxel,
    map_synapses_to_sequence,
    skeletonize_sequence,
)
from paleo.io import json_to_edits, json_to_graph
from tqdm.auto import tqdm

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


def bytes_to_df(bytes):
    with BytesIO(bytes) as f:
        df = pd.read_csv(f, index_col=0, compression="gzip")
    return df


# Convert to list of integers
def str_to_list(string_list):
    out = list(map(int, string_list.strip("[]").split()))

    return out


def format_synapses(synapses, side="pre"):
    synapses[f"{side}_pt_position"] = synapses[f"{side}_pt_position"].apply(str_to_list)

    synapses["x"] = synapses[f"{side}_pt_position"].apply(lambda x: x[0])
    synapses["y"] = synapses[f"{side}_pt_position"].apply(lambda x: x[1])
    synapses["z"] = synapses[f"{side}_pt_position"].apply(lambda x: x[2])
    synapses["x"] = synapses["x"] * 4
    synapses["y"] = synapses["y"] * 4
    synapses["z"] = synapses["z"] * 40

    return synapses


def decode_synapses(input, side="pre"):
    out = bytes_to_df(input)
    out = format_synapses(out, side=side)
    return out


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


root_id = finished_roots[3]

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
client = CAVEclient("minnie65_public", version=1181)

metaedits, metaedit_map = get_metaedits(edits)
nuc_supervoxel_id = get_nucleus_supervoxel(root_id, client)
anchor_nodes = get_node_aliases(nuc_supervoxel_id, client, stop_layer=2)
print(f"{time.time() - currtime:.3f} seconds elapsed to pull info.")

# %%

currtime = time.time()

graphs_by_sequence = {}
skeletons_by_sequence = {}
dfs = []
# ("historical", edits),
for sequence_name, sequence_edits in [("idealized", metaedits)]:
    graphs_by_state = apply_edit_sequence(
        initial_graph,
        sequence_edits,
        anchor_nodes,
        return_graphs=True,
        include_initial=True,
    )

    skeletons_by_state, mappings_by_state = skeletonize_sequence(
        graphs_by_state, root_id=root_id, client=client, remove_unchanged=True
    )

    skeletons_by_sequence[sequence_name] = skeletons_by_state

    synapses_by_state = map_synapses_to_sequence(pre_synapses, graphs_by_state)

    synapse_to_skeleton_by_state = {}
    for state_id in skeletons_by_state.keys():
        synapse_to_level2 = synapses_by_state[state_id]
        level2_to_skeleton = mappings_by_state[state_id]
        synapse_to_skeleton = {
            syn_id: level2_to_skeleton[level2_id]
            for syn_id, level2_id in synapse_to_level2.items()
        }
        synapse_to_skeleton_by_state[state_id] = synapse_to_skeleton

    precision_recall_df = compute_precision_recall(synapse_to_skeleton_by_state)
    precision_recall_df["sequence"] = sequence_name
    precision_recall_df["order"] = np.arange(len(precision_recall_df))
    dfs.append(precision_recall_df)

currtime = time.time()

# %%

results = pd.concat(dfs)
results = results.reset_index().set_index(["sequence", "state_id", "order"])
results = results.melt(value_name="value", var_name="metric", ignore_index=False)
results = results.reset_index()

# %%

fig, ax = plt.subplots()

sns.lineplot(data=results, x="order", y="value", hue="sequence", style="metric", ax=ax)

# %%
from pathlib import Path

current_file = Path(__file__)
output_path = current_file.parent.parent / "panels" / current_file.name.strip(".py")
output_path.mkdir(exist_ok=True, parents=True)


# %%
n_steps = 10
for sequence_name, skeleton_sequence in skeletons_by_sequence.items():
    is_new = check_skeleton_changes(skeleton_sequence)

    plotter = pv.Plotter(window_size=(1600, 1600))
    plotter.open_gif(
        str(output_path / f"{root_id}_{sequence_name}_skeleton_sequence.gif"), fps=20
    )

    last_state = list(skeleton_sequence.keys())[-1]
    last_skeleton = skeleton_sequence[last_state]
    morphology = MorphSync()
    morphology.add_graph(last_skeleton, "skeleton")
    skel_actor = plotter.add_mesh(
        morphology.skeleton.to_pyvista(), color="black", line_width=3
    )
    plotter.camera_position = "zy"
    # flip the camera so that -y is up
    plotter.camera.up = [0, -1, 0]
    plotter.camera.elevation = 0
    plotter.camera.zoom(1.0)
    plotter.remove_actor(skel_actor)

    current_skeleton = None
    for state, skeleton in tqdm(
        skeleton_sequence.items(), total=len(skeleton_sequence)
    ):
        if is_new[state]:
            morphology = MorphSync()
            morphology.add_graph(skeleton, "skeleton")

            if current_skeleton is not None:
                delta, spatial_points = compare_skeletons(current_skeleton, skeleton)
                added_skeleton = (spatial_points, delta.added_edges)
                removed_skeleton = (spatial_points, delta.removed_edges)
                morphology.add_graph(added_skeleton, "added_skeleton")
                morphology.add_graph(removed_skeleton, "removed_skeleton")
            actors = []
            if len(morphology.skeleton.edges) > 0:
                skel_actor = plotter.add_mesh(
                    morphology.skeleton.to_pyvista(),
                    color="black",
                    line_width=3,
                )
                actors.append(skel_actor)
            if current_skeleton is not None:
                if len(morphology.added_skeleton.edges) > 0:
                    added_actor = plotter.add_mesh(
                        morphology.added_skeleton.to_pyvista(),
                        color="#00e3ff",
                        line_width=7,
                    )
                    actors.append(added_actor)
                if len(morphology.removed_skeleton.edges) > 0:
                    removed_actor = plotter.add_mesh(
                        morphology.removed_skeleton.to_pyvista(),
                        color="#e935a1",
                        line_width=7,
                    )
                    actors.append(removed_actor)
            for i in range(n_steps):
                plotter.camera.azimuth += 1
                plotter.write_frame()
            for actor in actors:
                plotter.remove_actor(actor)
        current_skeleton = skeleton
    plotter.close()

# %%
is_new = check_skeleton_changes(skeleton_sequence)

pv.set_jupyter_backend("client")
plotter = pv.Plotter()

# plotter.camera_position = "zy"
# flip the camera so that -y is up
plotter.camera.up = [0, -1, 0]
# plotter.camera.elevation = 0
# plotter.camera.zoom(1.0)
istate = 11
state = list(skeleton_sequence.keys())[istate]
state_before = list(skeleton_sequence.keys())[istate - 1]


nuc_info = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_equal_dict={"pt_root_id": root_id},
    split_positions=True,
    desired_resolution=[1, 1, 1],
)
nuc_loc = nuc_info[["pt_position_x", "pt_position_y", "pt_position_z"]].values.squeeze()

morphology = MorphSync()
morphology.add_graph(skeleton, "skeleton")

current_skeleton = skeleton_sequence[state]
skeleton = skeleton_sequence[state_before]

delta, spatial_points = compare_skeletons(current_skeleton, skeleton)
added_skeleton = (spatial_points, delta.added_edges)
removed_skeleton = (spatial_points, delta.removed_edges)
morphology.add_graph(added_skeleton, "added_skeleton")
morphology.add_graph(removed_skeleton, "removed_skeleton")

skel_actor = plotter.add_mesh(
    morphology.skeleton.to_pyvista(),
    color="black",
    line_width=3,
)
if len(morphology.added_skeleton.edges) > 0:
    added_actor = plotter.add_mesh(
        morphology.added_skeleton.to_pyvista(),
        color="#00e3ff",
        line_width=7,
    )
    actors.append(added_actor)
if len(morphology.removed_skeleton.edges) > 0:
    removed_actor = plotter.add_mesh(
        morphology.removed_skeleton.to_pyvista(),
        color="#e935a1",
        line_width=7,
    )
    actors.append(removed_actor)

plotter.camera.focal_point = nuc_loc

plotter.show()

# %%
