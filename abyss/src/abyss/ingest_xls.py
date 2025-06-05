import os
import pandas as pd
from glob import glob
import abyss.dataparser as dp

def process_xls_file(filename: object, columns_selected: list) -> pd.DataFrame:
    '''        
    Process a xls file containing Setitec drilling data.
    
    Args:
        filename (str): Name of the xls file to process.
        columns_selected (list): List of column names to select for the metadata DataFrame.

    Returns:
        df_merged (pd.DataFrame): DataFrame containing columns from the xls file, merged with the metadata and time series data.

    Effect:
        None
    '''
    return process_xls_file__firmware_v3(filename, columns_selected)

def create_dataframe_from_xls_files(filelist: list, df: pd.DataFrame, columns_selected: list)-> pd.DataFrame:
    return create_dataframe_from_xls_files__firmware_v3(filelist, df, columns_selected)

def process_xls_file__firmware_v3(filename: object, columns_selected: list) -> pd.DataFrame:
    '''        
    Process a xls file containing firmware v3 data. This works only with data from firmware v3.
    
    Args:
        filename (str): Name of the xls file to process.
        columns_selected (list): List of column names to select for the metadata DataFrame.

    Returns:
        df_merged (pd.DataFrame): DataFrame containing columns from the xls file, merged with the metadata and time series data.

    Effect:
        None
    '''
    # Load the data using the "auto" option so that all of the contents are loaded into lists and one dataframe
    fn = str(filename)
    data_v3 = dp.loadSetitecXls(fn, version="auto")

    # For firmware v3.1.3.1, the first 8 element of the data_v3 list are dicts, the last one is a time series dataframe
    list_of_dicts = data_v3[0:-1]

    # Create DataFrame for pset data
    df_pset = pd.DataFrame.from_dict(list_of_dicts[7]) # Pandas version
    # df_pset = pd.from_dict(list_of_dicts[7]) # Vaex version

    # Create DataFrame for time series data
    df_ts = data_v3[-1]

    # Let's store the file path to the time series dataframe to enable merging later
    df_ts['File_Path'] = fn

    # We combine the dicts into a single dict containing all metadata        
    # https://stackoverflow.com/questions/3494906/how-do-i-merge-a-list-of-dicts-into-a-single-dict
    dict_combine = {k: v for d in list_of_dicts for k, v in d.items()}
    
    # Create DataFrame for metadata with selected columns
    df_metadata = pd.DataFrame.from_dict({sk: dict_combine[sk] for sk in columns_selected}) # Pandas version
    # df_metadata = pd.from_dict({sk: dict_combine[sk] for sk in columns_selected}) # Vaex version

    # Let's store the file path to the metadata dataframe to enable merging with the time series dataframe from earlier
    df_metadata['File_Path'] = fn

    # Merge the time series dataframe with the metadata dataframe
    df_merged = pd.merge(df_metadata, df_ts, on='File_Path')
    df_merged = pd.merge(df_merged, df_pset, left_on='Step (nb)', right_on='Step Nb', how='left')

    return df_merged


def create_dataframe_from_xls_files__firmware_v3(filelist: list, df: pd.DataFrame, columns_selected: list) -> pd.DataFrame:
    '''
        Based on a list of files, create a dataframe from the files. 
        This works only with data from firmware v3.

        Args:
            filelist (list): List of files to create a dataframe from.
            df (pandas.DataFrame): Dataframe to append to. Defaults to None.
            columns_selected (list): List of columns to select. Defaults to None, which selects

        Returns:
            df (pandas.DataFrame): Dataframe containing all the files.

        Effect:
            None
    '''
    # ====================================================================
    # Input checking
    # ====================================================================
    assert(len(filelist) > 0)
    if columns_selected is None:
        # columns_selected = ['BOX SN', 'Motor SN', 'Head Name','Head Global Counter','Head Local Counter', 'Cycle Time (s)', 'Distance (mm)']
        columns_selected_0 = ['BOX SN', 'Motor SN', 'Head Name','Head Global Counter','Head Local Counter', 'Cycle Time (s)', 'Distance (mm)']
        columns_selected_1 = ['BOX SN', 'Motor SN', 'Head Name','Head Global Counter','Head Local Counter 1', 'Head Local Counter 2', 'Cycle Time (s)', 'Distance (mm)']
    else:
        columns_selected_1 = columns_selected
    if df is None:
        df = pd.DataFrame()

    # ====================================================================
    # Process files
    # ====================================================================
    for fn in sorted(filelist, key=os.path.basename):
        # print(fn)
        try:
            df_temp = process_xls_file__firmware_v3(fn, columns_selected_1)
        except KeyError:
            print(fn)
            df_temp = process_xls_file__firmware_v3(fn, columns_selected_1)
        df = pd.concat([df, df_temp], ignore_index=True)
        # break

    # Casting strings to ints and floats
    # Use original column names
    try:
        df = df.astype({
            'BOX SN': int, 
            'Motor SN': int, 
            'Head Name': int, 
            'Head Global Counter': int, 
            'Head Local Counter': int, 
            'Cycle Time (s)': 'float16', 
            'Distance (mm)': 'float16', 
            'File_Path': str, 
            'Step (nb)': 'int8', 
            'Stop code': 'int32',
            'Step Nb': 'int8',
            'Step On/Off': bool
            })
    except KeyError:
        df = df.astype({
            'BOX SN': int, 
            'Motor SN': int, 
            'Head Name': int, 
            'Head Global Counter': int, 
            'Head Local Counter 1': int, 
            'Head Local Counter 2': int, 
            'Cycle Time (s)': 'float16', 
            'Distance (mm)': 'float16', 
            'File_Path': str, 
            'Step (nb)': 'int8', 
            'Stop code': 'int32',
            'Step On/Off': bool
            })

    # Column names tidying up
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    df.columns = df.columns.str.replace("(", "", regex=False)
    df.columns = df.columns.str.replace(")", "", regex=False)
    df.columns = df.columns.str.replace("/", "_", regex=False)

    # Casting strings to ints and floats
    # Use new column names
    df = df.astype({
        'DEP_mm': 'float32',
        'RPM': 'float32',
        'AV_mm_s': 'float32',
        'AV_mm_tr': 'float32',
        'Thrust_Max_A': 'float32',
        'Torque_Max_A': 'float32',
        'Thrust_Min_A': 'float32',
        'Torque_Min_A': 'float32',
        'Thrust_Safety_A': 'float32',
        'Torque_Safety_A': 'float32',
        'Gap_mm': 'float32',
        'Peck_nb': 'float32',
        'Delay_ms': 'float32',
        'Stroke_Limit_A': 'float32',
        'Thrust_Limit_A': 'float32',
        'Torque_Limit_A': 'float32',
        'LUB_AIR': 'category',
        'LUB_FLOW': 'category',
        'Vacuum': 'float16',
        'Material': 'float16',
        })

    return df



def ingest_all_xls_files(directory_path):
    """
    Ingest data from all .xls files in the given directory.
    
    Args:
        directory_path (str): Path to directory containing .xls files
        
    Returns:
        pd.DataFrame: Combined data from all .xls files
    """
    all_data = []
    
    # Find all .xls files in the directory
    xls_files = glob(os.path.join(directory_path, "*.xls"))
    
    if not xls_files:
        print(f"No .xls files found in {directory_path}")
        return pd.DataFrame()
    
    for file_path in xls_files:
        try:
            print(f"Processing: {os.path.basename(file_path)}")
            
            # Read the Excel file
            df = dp.loadSetitecXls(file_path, version='auto_data')
            
            # Add source file column for tracking
            df['filename'] = os.path.basename(file_path)
            
            # Basic data cleaning (adjust based on your needs)
            # Remove rows where all values are NaN
            # df = df.dropna(how='all')
            
            # Remove columns where all values are NaN
            # df = df.dropna(axis=1, how='all')
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if all_data:
        # Combine all DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['Position (mm)'] = -combined_df['Position (mm)']
        print(f"Successfully processed {len(all_data)} files")
        print(f"Combined dataset shape: {combined_df.shape}")
        return combined_df
    else:
        print("No data was successfully processed")
        return pd.DataFrame()