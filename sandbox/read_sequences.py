# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from caveclient import CAVEclient
from cloudfiles import CloudFiles
from sklearn.metrics import pairwise_distances

VERSION = int(os.getenv("VERSION", 1181))
BUCKET = os.getenv("BUCKET", "allen-minnie-phase3/paleo_edits")
BUCKET = f"gs://{BUCKET}"
BUCKET = BUCKET + "/v" + str(VERSION)

client = CAVEclient("minnie65_phase3_v1", version=VERSION)
cf = CloudFiles(BUCKET)
# %%

pre_synapse_files = list(cf.list("pre_synapses"))
pre_synapse_roots = [
    int(file.split("/")[1].split("_")[0]) for file in pre_synapse_files
]

synapse_sequence_files = list(cf.list("pre_synapse_sequences"))
synapse_sequence_roots = [
    int(file.split("/")[1].split("_")[0]) for file in synapse_sequence_files
]

state_info_files = list(cf.list("state_info"))
state_info_roots = [int(file.split("/")[1].split("_")[0]) for file in state_info_files]

finished_roots = list(
    set(synapse_sequence_roots)
    .intersection(state_info_roots)
    .intersection(pre_synapse_roots)
)
len(finished_roots)
# %%
scheme = "idealized"


from io import BytesIO

import pandas as pd


def intize_dict_keys(d):
    return {int(k): v for k, v in d.items()}


def get_dataframe(cf, file, **kwargs):
    bytes_out = cf.get(file)
    with BytesIO(bytes_out) as f:
        df = pd.read_csv(f, compression="gzip", **kwargs)
    return df


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


def compute_target_proportions(df):
    return (
        df["post_cell_type"].value_counts(dropna=True).transform(lambda x: x / x.sum())
    )


def compute_target_counts(df):
    return df["post_cell_type"].value_counts(dropna=True)


def apply_to_sequence(func, synapse_sequence, synapse_df):
    rows = {}
    for state_id, synapse_ids in synapse_sequence.items():
        sub_synapses = synapse_df.loc[synapse_ids]
        target_proportions = func(sub_synapses)
        rows[state_id] = target_proportions

    target_proportions_df = pd.DataFrame(rows).T.fillna(0)
    target_proportions_df.index.name = "state_id"
    return target_proportions_df


VERSION = 1181
client = CAVEclient("minnie65_phase3_v1", version=VERSION)


cell_types_table = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")
cell_types_table = cell_types_table.drop_duplicates("pt_root_id")
cell_types_table = cell_types_table.set_index("pt_root_id")


for root in np.random.permutation(finished_roots)[:50]:
    file = f"pre_synapse_sequences/{root}_{scheme}_sequence.json"
    if file in synapse_sequence_files:
        synapse_sequence = intize_dict_keys(cf.get_json(file))
    else:
        continue

    file = f"state_info/{root}_{scheme}_state_info.csv.gz"
    if file in state_info_files:
        state_info = get_dataframe(cf, file, index_col=0)
    else:
        continue

    file = f"pre_synapses/{root}_pre_synapses.csv.gz"
    if file in pre_synapse_files:
        pre_synapses = decode_synapses(
            cf.get(f"pre_synapses/{root}_pre_synapses.csv.gz"), side="pre"
        )
    else:
        continue

    pre_synapses["post_cell_type"] = pre_synapses["post_pt_root_id"].map(
        cell_types_table["cell_type"]
    )

    rows = {}
    for state_id, synapse_ids in synapse_sequence.items():
        sub_synapses = pre_synapses.loc[synapse_ids]
        target_proportions = compute_target_proportions(sub_synapses)
        rows[state_id] = target_proportions

    target_proportions_df = pd.DataFrame(rows).T.fillna(0)
    target_proportions_df.index.name = "state_id"
    feature_cols = target_proportions_df.columns
    target_proportions_df = target_proportions_df.join(state_info)

    target_proportions_df_melt = target_proportions_df.reset_index().melt(
        id_vars=state_info.columns.tolist() + ["state_id"], value_vars=feature_cols
    )

    # fig, ax = plt.subplots()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    ax = axs[0, 0]
    sns.lineplot(
        data=target_proportions_df_melt,
        x="cumulative_n_edits",
        y="value",
        hue="variable",
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Cumulative edits")
    ax.set_ylabel("Proportion of synapses")

    X_df = target_proportions_df[feature_cols]

    metric = "euclidean"
    distances = pairwise_distances(X_df, metric=metric)
    distances_df = pd.DataFrame(distances, index=X_df.index, columns=X_df.index)

    state_index = X_df.index
    final_state = state_index[-1]
    half_state_iloc = len(state_index) // 2
    half_state = state_index[half_state_iloc]

    final_state_distances = distances_df.loc[final_state]
    half_state_distances = distances_df.loc[half_state]

    ax = axs[0, 1]
    sns.lineplot(y=final_state_distances, x=state_info["cumulative_n_edits"], ax=ax)
    sns.lineplot(
        y=half_state_distances[: half_state_iloc + 1],
        x=state_info["cumulative_n_edits"][: half_state_iloc + 1],
        ax=ax,
        linestyle="--",
    )
    ax.set_xlabel("Cumulative edits")
    ax.set_ylabel("Euclidean distance to final")

    window = 5
    differences = np.diagonal(distances, offset=window)
    difference_index = state_info["cumulative_n_edits"][window:]

    ax = axs[1, 0]
    sns.lineplot(y=differences, x=difference_index, ax=ax)
    ax.set_xlabel("Cumulative edits")
    ax.set_ylabel(f"Euclidean distance to {window} states ago")

    ax = axs[1, 1]
    ax.axis("off")

    plt.savefig(
        f"panels/read_sequences/{root}_{scheme}.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

# %%
