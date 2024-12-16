from io import BytesIO
from pathlib import Path

import pandas as pd


def get_path(file, directory):
    current_file = Path(file)
    output_path = (
        current_file.parent.parent / directory / current_file.name.strip(".py")
    )
    output_path.mkdir(exist_ok=True, parents=True)
    return output_path


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


def subset_dict(d, keys):
    return {k: d[k] for k in keys}
