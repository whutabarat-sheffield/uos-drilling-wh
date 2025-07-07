import pandas as pd
import numpy as np
import torch

"""           The functions provided in this script are hard coded!               """
"""   Please make sure the columns of the data are exactly the same as required   """



def extend_dataframe(df_onehole, num_rows=400):
    """
    Zee
    This function adds 400 rows to both start and end of the datafrome.
    This is because for a single hole, if we perform moving window prediction, the window may out of the data range.
    :param df_onehole: pandas dataframe to be processed, it should only contain the data of a single hole.
    :return: processed dataframe
    """
    constant_values = df_onehole.iloc[0, 15:26].values
    rows_to_add = pd.DataFrame(-1, index=np.arange(num_rows), columns=df_onehole.columns[:])
    rows_to_add[df_onehole.columns[15:26]] = constant_values[:]
    df_onehole = pd.concat([rows_to_add, df_onehole, rows_to_add], ignore_index=True)

    # Ensure all the values are numeric and do not have NaN
    df_onehole = df_onehole.apply(pd.to_numeric, errors='coerce')
    df_onehole.fillna(0, inplace=True)
    return df_onehole


def df_to_tensor_input(dataframe):
    """
    Zee
    this func take pandas date frame as input, return the tensor for model's input with batch size of 1
    :param dataframe: pandas data frame
    :return: (1, 512, 17)
    """
    # dataframe = dataframe.reset_index(drop=True)
    input_data = torch.tensor(
        dataframe.iloc[:, 0:17].values,
        dtype=torch.float32).unsqueeze(0)
    return input_data
