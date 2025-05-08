# # import matplotlib.pyplot as plt
# # import numpy as np
# # import os
# import pandas as pd
# from tqdm import tqdm




# def visualization(df_processed,save_path):
#     """
#     Yue
#     This fun visulise the dataset.
#     :param df_processed: pandas dataframe raw data.
#            save_path: the directory for saving the figures.
#     :return: NA.
#     """
#     no_measured_depth = []
#     unique_hole_ids = df_processed['HOLE_ID'].nunique()
#     grouped_p = df_processed.groupby('HOLE_ID').apply(lambda x: pd.Series([x.index.min(), x.index.max()], index=['Start_Index', 'End_Index']))
#     sorted_grouped_p = grouped_p.sort_values(by='Start_Index')
#     for iter in tqdm(range(unique_hole_ids)):
#         row_index = iter
#         # row_index = 108
#         start_index = sorted_grouped_p.iloc[row_index][0]  # replace with your start index
#         end_index = sorted_grouped_p.iloc[row_index][1]    # replace with your end index
#         print(start_index)
#         columns_to_plot = ['i_torque', 'step', 'xpos']  # replace with your column names
#         age_columns = 'local'



#         # Slicing the DataFrame
#         subset = df_processed.loc[start_index:end_index, columns_to_plot + [age_columns]]
#         subset.loc[:, 'step'] = subset.loc[:, 'step'] * 10
#         # subset.loc[:, 'xpos'] = subset.loc[:, 'xpos'] * 0.5

#         index_first_gt1 = False
#         for i in range(len(subset['i_torque']) - 2):  # Subtract 10 to avoid index out of bounds
#             # Check if the current value is greater than 1 and the next 10 values are all greater than 1
#             if subset['i_torque'].iloc[i] > 0.1 and all(subset['i_torque'].iloc[i+1:i+3] > 0.1) and subset['xpos'].iloc[i] > 3:
#             # if subset['i_torque'].iloc[i] > 0.1 and subset['xpos'].iloc[i] > 3:
#                 index_first_gt0 = i
#                 index_first_gt1 = i + start_index
#                 break  # Stop the loop once we find the first match

#         if index_first_gt1:
#             for i in range(index_first_gt0, 0, -1):
#                 if subset['i_torque'].iloc[i] - subset['i_torque'].iloc[i - 3] < 0.00001 and subset['i_torque'].iloc[i + 3] - subset['i_torque'].iloc[i] >= 0.05:
#                     index_first_gt0 = i
#                     index_first_gt1 = i + start_index
#                     break

#         # Plotting
#         plt.figure(figsize=(16, 9))
#         for col in columns_to_plot:
#             plt.plot(subset.index, subset[col], linewidth=1, label=col)

#         plt.plot(subset.index, subset[age_columns], linewidth=1, label=age_columns, color='grey', linestyle='--')


#         # Get hole ID for name the image
#         hole_id = df_processed.at[start_index, 'HOLE_ID']
#         # Check if the id of holes are consistant
#         id_set = df_processed.loc[start_index:end_index, 'HOLE_ID']
#         assert id_set.nunique() == 1


#         ## plot the measured depth (assume the step 0 to 1 is the start point)
#         mask = subset['step'] == 10
#         # first_index = mask.idxmax()  # Find the index of the first True in the mask (i.e., the first occurrence of 'step' == 1)
#         if index_first_gt1:
#             plt.axvline(x=index_first_gt1, color='blue', linestyle='-', linewidth=1, label='Entry point')



#         # Get the age depth of drilling hole
#         age = df_processed.at[start_index, 'local']
#         age_set = df_processed.loc[start_index:end_index, 'local']
#         assert age_set.nunique() == 1

#         # Get the predrilling flag
#         predrilling = str(df_processed.at[start_index, 'PREDRILLED'])
#         predrilling_set = df_processed.loc[start_index:end_index, 'PREDRILLED']
#         # assert predrilling_set.nunique() == 1


#         plt.title('Selected Data Visualization')
#         plt.xlabel('Row Index')
#         plt.ylabel('Value')
#         plt.legend()
#         # plt.show()
#         print(hole_id)
#         plt.savefig(f'{save_path}{iter}_{hole_id}_age{age}_predrilled{predrilling}.png', dpi=300)  # Saves each figure with a unique name
#         plt.close()