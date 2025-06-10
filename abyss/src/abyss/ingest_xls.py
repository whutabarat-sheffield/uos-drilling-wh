import os
import pandas as pd
import abyss.dataparser as dp
import warnings

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




class SetitecDataIngestor:
    """Class for ingesting Setitec XLS drilling data files."""
    
    def __init__(self, columns_selected=None):
        """
        Initialize the ingestion class.
        
        Args:
            columns_selected (list): List of columns to select from metadata.
                                    If None, default columns are used.
        """
        if columns_selected is None:
            self.columns_selected_0 = ['BOX SN', 'Motor SN', 'Head Name', 'Head Global Counter',
                                      'Head Local Counter', 'Cycle Time (s)', 'Distance (mm)']
            self.columns_selected_1 = ['BOX SN', 'Motor SN', 'Head Name', 'Head Global Counter',
                                       'Head Local Counter 1', 'Head Local Counter 2', 
                                       'Cycle Time (s)', 'Distance (mm)']
            self.columns_selected = self.columns_selected_1
        else:
            self.columns_selected = columns_selected
    
    def process_file(self, filename):
        """
        Process a single Setitec XLS file.
        
        Args:
            filename (str): Path to the XLS file to process.
            
        Returns:
            pd.DataFrame: DataFrame containing the processed data.
        """
        fn = str(filename)
        data_v3 = dp.loadSetitecXls(fn, version="auto")

        # For firmware v3.1.3.1, the first 8 elements are dicts, the last one is a time series dataframe
        list_of_dicts = data_v3[0:-1]

        # Create DataFrame for pset data
        df_pset = pd.DataFrame.from_dict(list_of_dicts[7])

        # Create DataFrame for time series data
        df_ts = data_v3[-1]

        # Store the file path to enable merging later
        df_ts['File_Path'] = fn

        # Combine the dicts into a single dict containing all metadata
        dict_combine = {k: v for d in list_of_dicts for k, v in d.items()}
        
        # Create DataFrame for metadata with selected columns
        try:
            df_metadata = pd.DataFrame.from_dict({sk: dict_combine[sk] for sk in self.columns_selected})
        except KeyError:
            # Try with alternate column set
            df_metadata = pd.DataFrame.from_dict({sk: dict_combine[sk] for sk in self.columns_selected_0})

        # Store the file path to enable merging
        df_metadata['File_Path'] = fn

        # Merge the time series dataframe with the metadata dataframe
        df_merged = pd.merge(df_metadata, df_ts, on='File_Path')
        df_merged = pd.merge(df_merged, df_pset, left_on='Step (nb)', right_on='Step Nb', how='left')

        return df_merged
        
    def process_filelist(self, filelist, df=None):
        """
        Process a list of Setitec XLS files.
        
        Args:
            filelist (list): List of file paths to process.
            df (pd.DataFrame, optional): Existing DataFrame to append to.
            
        Returns:
            pd.DataFrame: DataFrame containing all processed data.
        """
        assert(len(filelist) > 0), "File list is empty"
        
        if df is None:
            df = pd.DataFrame()

        for fn in sorted(filelist, key=os.path.basename):
            try:
                df_temp = self.process_file(fn)
                df = pd.concat([df, df_temp], ignore_index=True)
            except KeyError:
                print(f"KeyError when processing {fn}")
                continue
            
        return self._clean_and_cast_dataframe(df)
        
    def process_directory(self, directory_path, df=None):
        """
        Process all XLS files in a specified directory.
        
        Args:
            directory_path (str): Path to directory containing XLS files.
            df (pd.DataFrame, optional): Existing DataFrame to append to.
            
        Returns:
            pd.DataFrame: DataFrame containing all processed data.
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")
        
        xls_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                     if f.lower().endswith('.xls')]
        
        if not xls_files:
            warnings.warn(f"No XLS files found in {directory_path}")
            return df if df is not None else pd.DataFrame()
        
        return self.process_filelist(xls_files, df)
        
    def _clean_and_cast_dataframe(self, df):
        """
        Clean column names and cast data types in DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to clean and cast.
            
        Returns:
            pd.DataFrame: Cleaned and cast DataFrame.
        """
        # Cast to correct data types - try both schema variants
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

        # Clean column names
        df.columns = df.columns.str.replace(" ", "_", regex=False)
        df.columns = df.columns.str.replace("(", "", regex=False)
        df.columns = df.columns.str.replace(")", "", regex=False)
        df.columns = df.columns.str.replace("/", "_", regex=False)

        # Cast additional columns to correct data types
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
        
    def process_multiple_directories(self, directory_paths):
        """
        Process XLS files from multiple directories, creating a separate DataFrame for each.
        
        Args:
            directory_paths (list): List of directory paths to process.
            
        Returns:
            dict: Dictionary mapping directory paths to their respective DataFrames.
        """
        results = {}
        
        for directory in directory_paths:
            try:
                results[directory] = self.process_directory(directory)
            except Exception as e:
                warnings.warn(f"Error processing directory {directory}: {str(e)}")
                results[directory] = pd.DataFrame()  # Empty DataFrame for failed directory
                
        return results