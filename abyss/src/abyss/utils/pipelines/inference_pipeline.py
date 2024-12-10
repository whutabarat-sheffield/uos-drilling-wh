from transformers import PatchTSMixerConfig
from tsfm_public.toolkit.dataset import ForecastDFDataset
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from abyss.utils.functions.inference_visualisation import scatter_plot
from abyss.utils.functions.inference_visualisation import output_heatmap_plot
from abyss.utils.functions.inference_data_operation import extend_dataframe
from abyss.utils.functions.inference_data_operation import df_to_tensor_input

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import set_seed
from tqdm import tqdm

import time

set_seed(42)



def depth_estimation_pipeline(model, dataset_for_evl, start_xpos, threshold=-1):


    df_subset = dataset_for_evl.copy()
    df_subset = df_subset.reset_index(drop=True)

    unique_hole_ids = df_subset['HOLE_ID'].nunique()
    grouped = df_subset.groupby('HOLE_ID').apply(
        lambda x: pd.Series([x.index.min(), x.index.max()], index=['Start_Index', 'End_Index']))
    sorted_grouped = grouped.sort_values(by='Start_Index')

    for i in range(unique_hole_ids):
        start_index = sorted_grouped.iloc[i].iloc[0]
        end_index = sorted_grouped.iloc[i].iloc[1]
        df_onehole = df_subset.loc[start_index:end_index, :]

        '''This func add 400 rows to the beginning and end of dataframe'''
        df_onehole = extend_dataframe(df_onehole)

        exit_xpos_list = []
        for j in tqdm(range(0, 200, 5)):
            '''looking for the window to perform inference'''
            max_index = df_onehole['torque8'].idxmax()
            input_df = df_onehole.iloc[max_index-330+j:max_index+182+j]
            input_df = input_df.reset_index(drop=True)
            input_data = torch.tensor(
                input_df.iloc[:, 0:17].values,
                dtype=torch.float32).unsqueeze(0)
            out = model(input_data)
            original_data = input_data.squeeze(0).cpu().detach().numpy()
            heatmap_data = out['prediction_outputs'].squeeze(0).cpu().detach().numpy()
            heatmap_data[heatmap_data < threshold] = 0
            heatmap_all = heatmap_data

            heatmap_data = heatmap_data[:, :-2]
            column_sum = heatmap_data.sum(axis=1)
            column_sum = column_sum.reshape(-1, 1)
            max_heat_index = np.argmax(column_sum)

            original_data = original_data[:, :-2]

            '''exit xpos'''
            exit_hat = input_df.at[max_heat_index, 'xpos2']
            exit_xpos_list.append(exit_hat)

            '''estimation visualisation'''
            # output_heatmap_plot(true_exit_index=0,
            #                     estimated_exit_index=max_heat_index,
            #                     original_data=original_data,
            #                     heatmap_all=heatmap_all)


        avg_exit_xpos = sum(exit_xpos_list) / len(exit_xpos_list)

        '''plot the moving window avg heatmap for one hole'''
        # diff = abs(input_df['xpos2'] - avg_exit_xpos)
        # max_heat_avg = diff.idxmin()
        # heatmap_total = np.zeros((512, 17))
        # output_heatmap_plot(true_exit_index=0,
        #                     estimated_exit_index=max_heat_avg,
        #                     original_data=original_data,
        #                     heatmap_all=heatmap_total,
        #                     full_label_plot=True,
        #                     exit_end=0,
        #                     save_figure=False,
        #                     hole_id='hole_id',
        #                     error=None)

    return avg_exit_xpos - start_xpos


def exit_estimation_pipeline(model, dataset_for_evl, threshold=-1):


    df_subset = dataset_for_evl.copy()
    df_subset = df_subset.reset_index(drop=True)

    unique_hole_ids = df_subset['HOLE_ID'].nunique()
    grouped = df_subset.groupby('HOLE_ID').apply(
        lambda x: pd.Series([x.index.min(), x.index.max()], index=['Start_Index', 'End_Index']))
    sorted_grouped = grouped.sort_values(by='Start_Index')

    for i in range(unique_hole_ids):
        start_index = sorted_grouped.iloc[i].iloc[0]
        end_index = sorted_grouped.iloc[i].iloc[1]
        df_onehole = df_subset.loc[start_index:end_index, :]

        '''This func add 400 rows to the beginning and end of dataframe'''
        df_onehole = extend_dataframe(df_onehole)

        exit_xpos_list = []
        for j in tqdm(range(0, 200, 5)):
            '''looking for the window to perform inference'''
            max_index = df_onehole['torque8'].idxmax()
            input_df = df_onehole.iloc[max_index-330+j:max_index+182+j]
            input_df = input_df.reset_index(drop=True)
            input_data = torch.tensor(
                input_df.iloc[:, 0:17].values,
                dtype=torch.float32).unsqueeze(0)
            out = model(input_data)
            original_data = input_data.squeeze(0).cpu().detach().numpy()
            heatmap_data = out['prediction_outputs'].squeeze(0).cpu().detach().numpy()
            heatmap_data[heatmap_data < threshold] = 0
            heatmap_all = heatmap_data

            heatmap_data = heatmap_data[:, :-2]
            column_sum = heatmap_data.sum(axis=1)
            column_sum = column_sum.reshape(-1, 1)
            max_heat_index = np.argmax(column_sum)

            original_data = original_data[:, :-2]

            '''exit xpos'''
            exit_hat = input_df.at[max_heat_index, 'xpos2']
            exit_xpos_list.append(exit_hat)

            '''estimation visualisation'''
            # output_heatmap_plot(true_exit_index=0,
            #                     estimated_exit_index=max_heat_index,
            #                     original_data=original_data,
            #                     heatmap_all=heatmap_all)


        avg_exit_xpos = sum(exit_xpos_list) / len(exit_xpos_list)

        '''plot the moving window avg heatmap for one hole'''
        # diff = abs(input_df['xpos2'] - avg_exit_xpos)
        # max_heat_avg = diff.idxmin()
        # heatmap_total = np.zeros((512, 17))
        # output_heatmap_plot(true_exit_index=0,
        #                     estimated_exit_index=max_heat_avg,
        #                     original_data=original_data,
        #                     heatmap_all=heatmap_total,
        #                     full_label_plot=True,
        #                     exit_end=0,
        #                     save_figure=False,
        #                     hole_id='hole_id',
        #                     error=None)

    return avg_exit_xpos