# import numpy as np
import pandas as pd
# import os
from tqdm import tqdm

def raw_data_checker(data):
    """
    Zee
    This fun checks if the input raw data fulfill the required format.
    :param data: pandas dataframe raw data.
    :return: the original dataframe.
    """
    df = data.rename(columns={
        'I Torque (A)': 'i_torque',
        'Step (nb)': 'step',
        'Position (mm)': 'xpos'
    })[['i_torque', 'HOLE_ID', 'step', 'xpos', 'local',
        'PREDRILLED']]

    column_name_list = [
        'i_torque', 'HOLE_ID', 'step', 'xpos', 'local',
        'PREDRILLED'
    ]
    set1 = set(column_name_list)
    set2 = set(df.keys())
    assert set1 == set2

    if df['xpos'].head(200).sum() < 0:
        df['xpos'] = df['xpos'] * -1

    return df


def add_entry_point(df_processed):
    """
    Yue
    This fun finds&adds entering point into the data.
    :param data: pandas dataframe raw data.
    :return: the original dataframe & enter xpos dictionary.
    """
    unique_hole_ids = df_processed['HOLE_ID'].nunique()
    grouped_p = df_processed.groupby('HOLE_ID').apply(lambda x: pd.Series([x.index.min(), x.index.max()], index=['Start_Index', 'End_Index']))
    sorted_grouped_p = grouped_p.sort_values(by='Start_Index')
    hole_dict = {}
    enter_pos = {}
    for iter in tqdm(range(unique_hole_ids)):
        row_index = iter

        start_index = sorted_grouped_p.iloc[row_index].iloc[0]  # replace with your start index
        end_index = sorted_grouped_p.iloc[row_index].iloc[1]    # replace with your end index
        columns_to_plot = ['i_torque', 'step', 'xpos']  # replace with your column names
        age_columns = 'local'

        # Slicing the DataFrame
        subset = df_processed.loc[start_index:end_index, columns_to_plot + [age_columns]]
        subset.loc[:, 'step'] = subset.loc[:, 'step'] * 10
        # subset.loc[:, 'xpos'] = subset.loc[:, 'xpos'] * 0.5

        index_first_gt1 = False
        for i in range(len(subset['i_torque']) - 2):  # Subtract 10 to avoid index out of bounds
            # Check if the current value is greater than 1 and the next 10 values are all greater than 1
            if subset['i_torque'].iloc[i] > 0.1 and all(subset['i_torque'].iloc[i+1:i+3] > 0.1) and subset['xpos'].iloc[i] > 3:
                index_first_gt0 = i
                index_first_gt1 = i + start_index
                break  # Stop the loop once we find the first match

        if index_first_gt1:
            for i in range(index_first_gt0 - 1, 0, -1):
                if subset['i_torque'].iloc[i] - subset['i_torque'].iloc[i - 3] < 0.0001 and subset['i_torque'].iloc[i + 3] - subset['i_torque'].iloc[i] >= 0.05:
                    index_first_gt0 = i
                    index_first_gt1 = i + start_index
                    break


        hole_id = df_processed.at[start_index, 'HOLE_ID']
        subset_for_save = df_processed.loc[start_index:end_index].copy()
        # subset_for_save = df_processed.loc[start_index:end_index]
        subset_for_save['Entery_point'] = 0.0

        if index_first_gt1 in subset_for_save.index:
            subset_for_save.loc[index_first_gt1, 'Entery_point'] = 1
            enter_pos[hole_id] = subset_for_save.loc[index_first_gt1,'xpos']

        subset_for_save
        hole_dict[hole_id] = subset_for_save
        
    concateenated_df = pd.concat(hole_dict.values(), axis=0, ignore_index=True)
    
    return concateenated_df, enter_pos

