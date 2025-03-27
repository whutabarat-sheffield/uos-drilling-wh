import pandas as pd
import numpy as np
# import torch
import copy
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt


def generate_idx(begin, sample_rate, len):
    """
    Yue
    This fun generates a index list with a sampling rate.
    :param begin: begining index.
           sample_rate: sampling rate.
           len: the length of the index list
    :return: the index list.
    """
    return_array=[]
    while begin < len:
        # print(begin)
        return_array.append(begin)
        begin += sample_rate
    return return_array


def shifting(data, fill, pad_rate, padding):
    """
    Yue
    This fun shifts a column to make the data orgnization.
    :param data: pandas dataframe raw data.
           fill: 
           pad_rate: Padded number every padding.
           padding: Padded value.
    :return: the original dataframe.
    """
    data_tmp = copy.deepcopy(data)
    data_return = copy.deepcopy(data)
    if fill == 0:
        return data
    else:
        for _ in range(fill):
            for value in data_tmp:
                for _ in range(pad_rate):
                    value.insert(0, padding)
            data_return += copy.deepcopy(data_tmp)
    max_length = max(len(row) for row in data_return)
    for row in data_return:
        while len(row) < max_length:
            row.append(padding)
    return data_return, max_length


def shifting_sub(data, fill, pad_rate, padding, mid, max_length):
    """
    Yue
    This fun shifts a column to make the data orgnization.
    :param data: pandas dataframe raw data.
           fill: 
           pad_rate: Padded number every padding.
           padding: Padded value.
           mid: the column num of middle.
           max_length: max length of the dataframe.
    :return: the original dataframe.
    """
    data_tmp = copy.deepcopy(data)
    data_return = copy.deepcopy(data)
    if fill == 0:
        return data
    else:
        for _ in range(mid):
            for value in data_tmp:
                for _ in range(pad_rate):
                    value.insert(0, padding)
    data_return = data_tmp
    for row in data_return:
        while len(row) < max_length:
            row.append(padding)
    return data_return


def process_data(data,sample_rate, num_color, num_slide, pad_rate, id, padding=0.0):
    """
    Yue
    This fun process data for reorgnization.
    :param data: pandas dataframe raw data.
           sample rate: Number of intervals per down sample.
           num_color: Number of downsampling done.
           num_slide: Total number of down sampling.
           pad_rate: Padded number every padding.
           id: hole id.
           padding: padding value.
    :return: the original dataframe.
    """
    if num_slide % 2 ==0:
        print("num_slide must be an odd number.")
        return False
    filteddata = data[data['HOLE_ID']==id]
    filteddata = filteddata[data['step']==2]
    fill = num_slide-1
    mid = int(fill/2)
    torque = []
    step = []
    Entery_point = []
    xpos = []
    others = []
    colums = ['PREDRILLED', 'HOLE_ID', 'local']
    for i in range(num_color):
        idx = generate_idx(i, sample_rate, filteddata.shape[0])
        torque.append(filteddata.iloc[idx]['i_torque'].tolist())
        step.append(filteddata.iloc[idx]['step'].tolist())
        Entery_point.append(filteddata.iloc[idx]['Entery_point'].tolist())
        xpos.append(filteddata.iloc[idx]['xpos'].tolist())
    column_positions = [filteddata.columns.get_loc(name) for name in colums]
    others = filteddata.iloc[idx, column_positions]
    others = others.reset_index(drop=True)
    torque_return, max_length = shifting(torque, fill, pad_rate, padding)
    step_return = shifting_sub(step, fill, pad_rate, padding, mid, max_length)
    Entery_point_return = shifting_sub(Entery_point, fill, pad_rate, padding, mid, max_length)
    xpos_return = shifting_sub(xpos, fill, pad_rate, padding, mid, max_length)
    return torque_return, step_return, Entery_point_return, xpos_return, others


def data_orgnization(data,sample_rate = 3,num_color = 3,num_slide = 5,pad_rate = 1,padding = 0):
    """
    Yue
    This fun orgnize data and merge them into one dataframe.
    :param data: pandas dataframe raw data.
           sample_rate: Number of intervals per down sample.
           num_color: Number of downsampling done.
           num_slide: Total number of down sample.
           pad_rate: Padded number every padding.
           padding: The value padded for empty.
    :return: the original dataframe.
    """
    HOLE_ID = data['HOLE_ID'].unique()
    torque_dict={}
    step_dict={}
    Entery_point_dict={}
    enter_end_dict={}
    exit_start_dict={}
    exit_end_dict={}
    top_len_dict={}
    xpos_dict={}
    others_dict={}
    for i in HOLE_ID:
        torque_return, step_return, Entery_point_return, xpos_return, others = process_data(data,sample_rate, num_color, num_slide, pad_rate, i, padding)
        torque_dict[i] = np.asarray(torque_return, dtype=np.float32)
        step_dict[i] = np.asarray(step_return, dtype=np.float32)
        Entery_point_dict[i] = np.asarray(Entery_point_return, dtype=np.float32)
        xpos_dict[i] = np.asarray(xpos_return, dtype=np.float32)
        others_dict[i] = others

    df_merge = pd.DataFrame()
    for id in HOLE_ID:
        # Create an empty DataFrame
        df = pd.DataFrame()

        # Iterate through the ndarray and add each sub-array as a new column to the DataFrame
        for i, sublist in enumerate(torque_dict[id]):  # Transpose the array and iterate through columns
            column_name = f'torque{i+1}'  # Create column name dynamically
            df[column_name] = sublist
        for i, sublist in enumerate(step_dict[id]):  # Transpose the array and iterate through columns
            column_name = f'step{i+1}'  # Create column name dynamically
            df[column_name] = sublist
        for i, sublist in enumerate(Entery_point_dict[id]):  # Transpose the array and iterate through columns
            column_name = f'Entery_point{i+1}'  # Create column name dynamically
            df[column_name] = sublist
        for i, sublist in enumerate(xpos_dict[id]):  # Transpose the array and iterate through columns
            column_name = f'xpos{i+1}'  # Create column name dynamically
            df[column_name] = sublist
        df = pd.merge(df, others_dict[id], left_index=True, right_index=True)
        df_merge = pd.concat([df_merge, df], axis=0)
    return df_merge


def data_scaler(df, ref_data_path):
    """
    Zee
    :param df: 15T to scale,
    :param ref_data: for consistent scaling
    :return: scaled df
    """

    '''Remove the useless columns, Check if there is a NaN, True False --> 1, 0'''
    columns_to_drop = ['Entery_point1', 'Entery_point2',
                       'Entery_point3']
    df = df.drop(columns=columns_to_drop)

    '''add ref data for consistent scaling'''
    df_ref = pd.read_csv(ref_data_path)
    df_ref['local'] = 0
    df_ref['PREDRILLED'] = 0
    df_ref['xpos1'] = 0
    df_ref['xpos2'] = 0
    df_ref['xpos3'] = 0
    df_ref['HOLE_ID'] = 0

    new_order = ['torque1', 'torque2', 'torque3', 'torque4', 'torque5', 'torque6',
                 'torque7', 'torque8', 'torque9', 'torque10', 'torque11', 'torque12',
                 'torque13', 'torque14', 'torque15', 'local', 'PREDRILLED', 'xpos1', 'xpos2',
                 'xpos3', 'HOLE_ID']

    df = df[new_order]
    df_ref = df_ref[new_order]

    # Check if there is a NaN
    has_nan = df.isna().any().any()
    #print(f"DataFrame contains NaN values: {has_nan}")
    df = df.replace({True: 1, False: 0})

    df['local'] = df['local'] * 0.034 - 1.7
    df['PREDRILLED'] = df['PREDRILLED'].replace({0: -1.45, 1: 0.69})

    '''scale the combined df'''
    df['source'] = 'original'
    df_ref['source'] = 'reference'
    df_combined = pd.concat([df, df_ref])

    columns_to_scale = ['torque1', 'torque2', 'torque3', 'torque4', 'torque5', 'torque6',
                        'torque7', 'torque8', 'torque9', 'torque10', 'torque11', 'torque12',
                        'torque13', 'torque14', 'torque15']
    scaler = StandardScaler()
    df_combined[columns_to_scale] = scaler.fit_transform(df_combined[columns_to_scale])
    df_extracted = df_combined[df_combined['source'] == 'original'].copy()
    df_extracted.drop('source', axis=1, inplace=True)

    '''plot, only for debug'''
    # columns_to_plot = ['torque1', 'torque2', 'torque3', 'torque4', 'torque5', 'torque6',
    #                    'torque7', 'torque8', 'torque9', 'torque10', 'torque11', 'torque12',
    #                    'torque13', 'torque14', 'torque15', 'PREDRILLED', 'local']
    #
    # subset = df_extracted.loc[:, columns_to_plot]
    # plt.figure(figsize=(16, 9))
    # for col in columns_to_plot:
    #     plt.plot(subset.index, subset[col], label=col, linewidth=1)
    # plt.show()

    start_timestamp = pd.Timestamp('2024-02-25 00:00:00')  # only for ref, the exact values are not important
    df_extracted['timestamp'] = [start_timestamp + pd.Timedelta(seconds=i) for i in range(len(df_extracted))]

    return df_extracted

