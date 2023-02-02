import pandas as pd
from tabulate import tabulate


# Normalization function (min-max)
def min_max(data: pd, labels, new_min, new_max):
    norm_df = pd.DataFrame(columns=labels)
    norm_df[labels[0]] = data[labels[0]].values

    for column in data[1:]:
        min = data[column].min(axis=0)
        max = data[column].max(axis=0)
        values = data[column].values
        new_vals = []
        for val in values:
            new_val = ((val - min) / (max - min)) * (new_max - new_min) + new_min
            new_vals.append(round(new_val, 3))
        norm_df[column] = new_vals
    return norm_df


def print_table(data_frame: pd):
    print(tabulate(data_frame, headers=data_frame.columns, tablefmt='fancy_grid', numalign='center'))
