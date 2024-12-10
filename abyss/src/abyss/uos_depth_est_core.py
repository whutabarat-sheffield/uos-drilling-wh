from typing import Tuple
import numpy as np
import pandas as pd
import sysconfig
from pathlib import Path

from abyss.dataparser import loadSetitecXls
from abyss.inference import DepthInference


abyss_path = Path(sysconfig.get_path('platlib')) / 'abyss'


def get_setitec_signals(file_to_open: str) -> Tuple[np.ndarray, np.ndarray]:
    df = loadSetitecXls(file_to_open, version="auto_data")
    position = np.abs(df['Position (mm)'].values)
    torque = df['I Torque (A)'].values
    torque_empty = df['I Torque Empty (A)'].values
    torque_full = torque + torque_empty
    return position, torque_full

########################################################
# Functions for converting borehole depths into grip lengths
########################################################

def calculate_quotient_remainder(borehole_depth: float, fastener_unit_length: float = 1.5875) -> Tuple[int, float]:
    if not isinstance(borehole_depth, float) or not isinstance(fastener_unit_length, float):
        raise TypeError("Both borehole_depth and fastener_unit_length must be floats.")
    quotient = borehole_depth // fastener_unit_length
    remainder = borehole_depth % fastener_unit_length
    return quotient, remainder

# def calculate_fastener_grip_length(borehole_depth, fastener_unit_length=1.5875, lower_threshold=0.396875, upper_threshold=1.42875):
#     quotient, remainder = calculate_quotient_remainder(borehole_depth, fastener_unit_length)
#     d1 = quotient + 1
#     rule2 = remainder > upper_threshold
#     d2 = quotient + rule2
#     rule3 = remainder < lower_threshold
#     if remainder < lower_threshold:
#         rule3 = 0
#     elif remainder > upper_threshold:
#         rule3 = 2
#     else:
#         rule3 = 1
#     d3 = quotient + rule3
#     return d1, d2, d3


def calculate_fastener_grip_length(borehole_depth: float, fastener_unit_length: float = 1.5875, 
                                   lower_threshold: float = 0.396875, upper_threshold: float = 1.42875) -> Tuple[float, float, float]:
    """Calculates the fastener grip lengths based on the provided borehole depth,
    fastener unit length, lower threshold, and upper threshold.

    Args:
        borehole_depth (float): The depth of the borehole.
        fastener_unit_length (float, optional): The length of the fastener unit. Defaults to 1.5875.
        lower_threshold (float, optional): The lower threshold. Defaults to 0.396875.
        upper_threshold (float, optional): The upper threshold. Defaults to 1.42875.

    Returns:
        Tuple[float, float, float]: A tuple containing d1, d2, and d3, which represent the fastener grip lengths.

    Example:
        calculate_fastener_grip_length(10.0, 2.5, 1.0, 2.0)
        # Output: (5.0, 5.0, 6.0)
    """
    quotient, remainder = calculate_quotient_remainder(borehole_depth, fastener_unit_length)
    
    d1 = quotient + 1
    
    rule2 = remainder > upper_threshold
    d2 = quotient + rule2
    
    rule3 = remainder < lower_threshold
    if remainder < lower_threshold:
        rule3 = 0
    elif remainder > upper_threshold:
        rule3 = 2
    else:
        rule3 = 1
    
    d3 = quotient + rule3
    
    return d1, d2, d3

########################################################
# DATAFRAMES FUNCTIONS
########################################################

# REMOVED

########################################################
# Segmented rolling gradient method
########################################################


def depth_est_xls_persegment_stats(file_path, bit_depth=0.7, stat = 'median', simple_results=True, debug=False):
    # Get signals from source file
    # bit_depth = 2.0 # mm, this needs to be supplied externally to be accurate
    pos, torque = get_setitec_signals(file_path)
    return depth_est_persegment_stats(torque, pos, bit_depth=bit_depth, stat=stat, simple_results=simple_results, debug=debug)

def depth_est_persegment_stats(torque, pos, bit_depth=0.7, stat = 'median', simple_results=True, debug=False):
    """
    Function to perform depth estimation using the break-gradient on a single XLS file.
    The function will return a dictionary containing the estimated positions and the
    estimated depths. A key feature is that the window size is calculated based on the bit depth
    and the feed rate, so that the window size is larger for slower feed rates.

    Parameters
    ----------
    file_path : str
        Path to the XLS file to be processed.
    bit_depth : float
        Depth of the front face of the drilling bit.
    simple_results : bool
        Flag to return a simple dictionary with only the estimated positions and depths.

    Returns
    -------
    dict
        Dictionary containing the estimated positions and depths.
    """
    dfcyc = pd.DataFrame({'pos':pos, 'torque':torque})
    dfcyc['diffpos'] = dfcyc.pos.diff()
    dfcyc['diffpos2'] = dfcyc.pos.diff().diff()
    # split into step segments
    # nsteps = dfcyc.diffpos.lt(0).sum() # in case it's needed # OBSOLETE
    dfcyc['step'] = dfcyc.diffpos.lt(0).cumsum()
    # dfcyc['step'] = dfcyc.diffpos2.lt(-0.07).cumsum()
    # calculate the mean of the torque gradient in each segment separately
    dfresult = pd.DataFrame()
    l_idx, l_pos = [], []
    for i in dfcyc.step.unique():
        dftemp = dfcyc[dfcyc.step==i].copy()
        dftemp['diffpos'] = dftemp.pos.diff()
        dftemp.diffpos.dropna()
        feedrate = dftemp.diffpos.median() # can be mean or median
        # feedrate = dftemp.diffpos.mean() # can be mean or median
        wsize = round(bit_depth/feedrate) # window size set according to the bit depth and feed rate. Slower feed rate means more samples per mm so a larger window size is needed
        
        dftemp['difftorque'] = dftemp.torque.diff()/dftemp.pos.diff()
        dftemp['difftorquemean'] = dftemp['difftorque'].rolling(wsize).mean()
        dftemp['difftorquemax'] = dftemp['difftorque'].rolling(wsize).max()
        dftemp['difftorquemin'] = dftemp['difftorque'].rolling(wsize).min()
        dftemp['difftorquemedian'] = dftemp['difftorque'].rolling(wsize).median()
        l_idx.append(dict( meanmin = dftemp['difftorquemean'].idxmin(), meanmax = dftemp['difftorquemean'].idxmax(), medianmin = dftemp['difftorquemedian'].idxmin(), medianmax = dftemp['difftorquemedian'].idxmax()))
        l_pos.append(dict( meanmin = l_idx[i]['meanmin'], meanmax = l_idx[i]['meanmax'], medianmin = l_idx[i]['medianmin'], medianmax = l_idx[i]['medianmax']))
        # l_pos.append(dict( meanmin = dftemp['pos'].iloc[dftemp['difftorquemean'].idxmin()], meanmax = dftemp['pos'].iloc[dftemp['difftorquemean'].idxmax()], medianmin = dftemp['pos'].iloc[dftemp['difftorquemedian'].idxmin()], medianmax = dftemp['pos'].iloc[dftemp['difftorquemedian'].idxmax()]))
        
        dfresult = pd.concat([dfresult, dftemp], axis=0)
        if debug: 
            print(f"For step {i}, window size={wsize}")

    residx = []
    respos = []

    # for i in range(dfresult.step.max()):
    for i in range(3):
        max1 = dfresult[dfresult.step==i].difftorquemean.idxmax()
        max2 = dfresult[dfresult.step==i].difftorquemedian.idxmax()
        min1 = dfresult[dfresult.step==i].difftorquemean.idxmin()
        min2 = dfresult[dfresult.step==i].difftorquemedian.idxmin()
        residx.append((i, max1, max2))
        respos.append((i, pos[max1], pos[max2], pos[min1], pos[min2]))
    # dfresult = dfresult.dropna()
    keys = ['depth_estimate', 'depth_estimate_rag',
        'depth_estimate_quality', 'cfrp_depth_estimate',
        'cfrp_depth_estimate_rag', 'cfrp_depth_estimate_quality',
        'ti_depth_estimate', 'ti_depth_estimate_rag',
        'ti_depth_estimate_quality', 'estimated_positions']
    
    # Perform the depth estimation using the stats
    if stat == 'mean':
        estimated_positions_idx = [l_pos[1]['meanmax'], l_pos[2]['meanmax'], l_pos[2]['meanmin']]
    elif stat == 'median':
        estimated_positions_idx = [l_pos[1]['medianmax'], l_pos[2]['medianmax'], l_pos[2]['medianmin']]
    else:
        raise ValueError('stat must be either mean or median')
    
    estimated_positions = [pos[i] for i in estimated_positions_idx]

    depth_estimate = estimated_positions[2] - estimated_positions[0] # l_pos[2]['meanmin'] - l_pos[1]['meanmax']
    cfrp_depth_estimate =estimated_positions[1] - estimated_positions[0]
    ti_depth_estimate = estimated_positions[2] - estimated_positions[1]

    assert depth_estimate > 0, "Depth estimate is negative, check the data"
    assert cfrp_depth_estimate > 0, "CFRP depth estimate is negative, check the data"
    assert ti_depth_estimate > 0, "Ti depth estimate is negative, check the data"
    assert estimated_positions[0] < estimated_positions[1] < estimated_positions[2], "Estimated positions are not in the correct order, check the data"

    full_stack_info_dict = dict(depth_estimate = depth_estimate, # l_pos[2]['meanmin'] - l_pos[1]['meanmax'],
                                depth_estimate_rag = None, depth_estimate_quality = None, 
                                cfrp_depth_estimate = cfrp_depth_estimate, # l_pos[2]['meanmax'] - l_pos[1]['meanmax'],
                                cfrp_depth_estimate_rag = None, cfrp_depth_estimate_quality = None,
                                ti_depth_estimate = ti_depth_estimate, #l_pos[2]['meanmin'] - l_pos[2]['meanmax'], 
                                ti_depth_estimate_rag = None, ti_depth_estimate_quality = None,
                                estimated_positions = estimated_positions,
                                )
    if debug:
        print(dfresult.step.max())
        print(respos)
        print(f"Method 1 (mean-based)   depth CFRP= {respos[2][1]-respos[1][1]:0.3f} mm")
        print(f"                        depth Ti=   {respos[2][3]-respos[2][1]:0.3f} mm")
        print(f"                        depth total={respos[2][3]-respos[1][1]:0.3f} mm")
        print(f"Method 2 (median-based) depth CFRP= {respos[2][2]-respos[1][2]:0.3f} mm")
        print(f"                        depth Ti=   {respos[2][4]-respos[2][2]:0.3f} mm")
        print(f"                        depth total={respos[2][4]-respos[1][2]:0.3f} mm")
    if simple_results:
        # return full_stack_info_dict['depth_estimate']
        return full_stack_info_dict['estimated_positions']
    else:
        return full_stack_info_dict


########################################################
def keypoint_recognition_gradient(position, signal, smo_w = 30, wsize=None, bit_depth=0.7):
    """
    Perform keypoints recognition using the rolling gradient method.

    Args:
        position (array-like): Array of positions.
        signal (array-like): Array of signal signals.
        wsize (int): Window size for smoothing and rolling calculations.

    Returns:
        tuple: A tuple containing two elements:
            - dftemp (DataFrame): Processed DataFrame with additional columns.
            - l_pos (dict): Dictionary containing position values for different metrics.

    Notes:
        - This function performs smoothing, differentiation, and rolling calculations on the input data.
        - It calculates various metrics such as mean, max, min, and median of the differentiated signal with respect to position.
        - The function returns the processed DataFrame and a dictionary of position values for different metrics.

    """
    dftemp = pd.DataFrame({'pos':position, 'signal':signal})
    l_idx, l_pos = [], []
    dftemp.reset_index(drop=True, inplace=True)
    dftemp['signal_raw'] = dftemp['signal']
    dftemp['pos_raw'] = dftemp['pos']
    # This smoothing is needed otherwise the rest of the code falls
    dftemp['signal'] = dftemp['signal_raw'].rolling(smo_w, center = True).mean().bfill().ffill()
    dftemp['pos'] = dftemp['pos_raw'].rolling(smo_w, center = True).mean().bfill().ffill()
    dftemp['diffpos'] = dftemp['pos'].diff()
    feedrate = dftemp.diffpos.median(skipna=True) # can be mean or median
    # feedrate = dftemp.diffpos.mean() # can be mean or median, but for the Bremen data it's better to use the mean
    if not wsize:
        wsize = round(abs(bit_depth/feedrate)) # window size set according to the bit depth and feed rate. Slower feed rate means more samples per mm so a larger window size is needed
    # wsize = 30
    dftemp['diffsignal'] = dftemp['signal'].diff()
    dftemp['dsignal_dpos'] = dftemp['diffsignal']/dftemp['diffpos']
    dftemp['diffsignalmean'] = dftemp['dsignal_dpos'].rolling(wsize).mean().bfill().ffill()
    dftemp['diffsignalmax'] = dftemp['dsignal_dpos'].rolling(wsize).max().bfill().ffill()
    dftemp['diffsignalmin'] = dftemp['dsignal_dpos'].rolling(wsize).min().bfill().ffill()
    dftemp['diffsignalmedian'] = dftemp['dsignal_dpos'].rolling(wsize).median().bfill().ffill()
    l_idx.append(dict( meanmin = dftemp['diffsignalmean'].idxmin(), meanmax = dftemp['diffsignalmean'].idxmax(), medianmin = dftemp['diffsignalmedian'].idxmin(), medianmax = dftemp['diffsignalmedian'].idxmax()))
    l_pos.append(dict( meanmin = dftemp['pos'].iloc[dftemp['diffsignalmean'].idxmin()], meanmax = dftemp['pos'].iloc[dftemp['diffsignalmean'].idxmax()], medianmin = dftemp['pos'].iloc[dftemp['diffsignalmedian'].idxmin()], medianmax = dftemp['pos'].iloc[dftemp['diffsignalmedian'].idxmax()]))
    return dftemp, l_pos


def standard_xls_loading(file, signal='Torque'):
    """
    Loads and processes a Setitec XLS file.

    This function loads a Setitec XLS file, converts the data types of the columns,
    takes the absolute value of the 'Position (mm)' column, calculates the total
    torque and thrust by adding the respective 'I Torque' and 'I Thrust' columns
    with their 'Empty' counterparts. It then assigns the 'Signal' column based on
    the 'signal' argument.

    Parameters:
    file (str): The path to the Setitec XLS file to be loaded.
    signal (str, optional): The signal type. It must be either 'Torque' or 'Thrust'.
        Defaults to 'Torque'.

    Returns:
    df (pandas.DataFrame): The processed DataFrame.

    Raises:
    ValueError: If the 'signal' argument is not 'Torque' or 'Thrust'.
    """
    df = loadSetitecXls(file, version="auto_data")
    df = df.convert_dtypes()
    df['Position (mm)'] = df['Position (mm)'].abs()
    df['I Torque Total (A)'] = df['I Torque (A)'] + df['I Torque Empty (A)']
    df['I Thrust Total (A)'] = df['I Thrust (A)'] + df['I Thrust Empty (A)']
    if signal == 'Torque':
        df['Signal'] = df['I Torque Total (A)']
    elif signal == 'Thrust':
        df['Signal'] = df['I Thrust Total (A)']
    else:
        raise ValueError('signal must be either Torque or Thrust')
    return df


def kp_conversion(kplist, metric='median', drilling_conf='a'):
    # This is to deal with the different drilling configurations 
    entr= f'{metric}max'
    exit = f'{metric}min'
    res_array = []
    if drilling_conf == "HAM_upper":
        '''
        2 stacks, 4 steps - 3 key points'''
        pass
    elif drilling_conf == "MDB":
        '''
        2 stacks, 4 steps - 3 key points
        Step 1: approach, entry CFRP
        Step 2: exit CFRP / entry Ti
        Step 3: entry Ti / exit Ti
        Step 4: CSK'''
        res_array = (kplist[1][0], kplist[2][0], kplist[2][1])
    elif drilling_conf == "HAM_lower":
        '''
        2 stacks, 3 steps - 3 key points'''
        pass
    else:
        '''
        Single stack, single step - 2 key points'''
        pass

    return res_array


def kp_recognition_gradient(file, signal = 'Torque', smo_w = 30, wsize=30, bit_depth=0.7):
    """
    Perform keypoints recognition using the rolling gradient method.

    Args:
        file (str): Path to the XLS file to be processed.
        wsize (int): Window size for smoothing and rolling calculations.

    Returns:
        tuple: A tuple containing two elements:
            - dftemp (DataFrame): Processed DataFrame with additional columns.
            - l_pos (dict): Dictionary containing position values for different metrics.

    Notes:
        - This function performs smoothing, differentiation, and rolling calculations on the input data.
        - It calculates various metrics such as mean, max, min, and median of the differentiated signal with respect to position.
        - The function returns the processed DataFrame and a dictionary of position values for different metrics.

    """
    df = standard_xls_loading(file, signal=signal)
    i_steps = df['Step (nb)'].unique().tolist()
    l_idx, l_pos = [], []

    for i_step in i_steps:
        dfcyc = df[df['Step (nb)']==i_step]
        dftemp, l_pos_temp = keypoint_recognition_gradient(dfcyc['Position (mm)'], dfcyc['Signal'], smo_w=smo_w, wsize=wsize, bit_depth=bit_depth)
        dftemp['Step (nb)'] = i_step
        l_pos_temp[-1]['Step (nb)'] = i_step
        if i_step == i_steps[0]:
            dfresult = dftemp      
            l_pos = [l_pos_temp]
        else:
            dfresult = pd.concat([dfresult, dftemp], axis=0)
            l_pos.append(l_pos_temp)

    # for col in key_kp[istep]:
    #     # print(row[col].values)
    #     kp.append(row[col].values[0]) # Collect the keypoint values

    return l_pos


def kp_recognition_ml(file, signal = 'Torque'):
    di = DepthInference()
    l_output = di.infer_xls(file)
    return l_output

def kp_recognition(file):
    return None


def depth_est_segmented(file):
    try:
        result = depth_est_segmented_2stack(file)
    except IndexError:
        result = depth_est_segmented_1stack(file)
    return result


def depth_est_segmented_2stack(file):
    """
    Estimate the depth using segmented keypoint recognition gradient.

    Parameters:
    file (str): The file path of the input image.

    Returns:
    list: A list of depth estimation results.

    """
    key_kp = {1: ['medianmax'], 2: ['medianmax', 'medianmin']}
    l_result = kp_recognition_gradient(file)
    l_output = []
    for istep, row in enumerate(l_result):
        print(row)
        l_row = [row[0]['medianmax'], row[0]['medianmin']]
        l_output.append(l_row)

    l_output_depths = (l_output[1][0], l_output[2][0], l_output[2][1])
    return l_output_depths


def depth_est_segmented_1stack(file):
    """
    Estimate the depth using segmented keypoint recognition gradient.

    Parameters:
    file (str): The file path of the input image.

    Returns:
    list: A list of depth estimation results.

    """
    key_kp = {1: ['medianmax'], 2: ['medianmin']}
    l_result = kp_recognition_gradient(file)
    l_output = []
    for istep, row in enumerate(l_result):
        print(row)
        l_row = [row[0]['medianmax'], row[0]['medianmin']]
        l_output.append(l_row)

    l_output_depths = (l_output[0][0], l_output[0][1])
    return l_output_depths

def depth_est_ml(file):
    """
    Estimate the depth using machine learning.

    Args:
        file (str): The file path of the input data.

    Returns:
        list: The estimated depth values.

    """
    l_result = kp_recognition_ml(file)
    return l_result


def depth_est_combined(file):

    l_seg = depth_est_segmented(file)
    if len(l_seg) == 3:
        l_ml = depth_est_ml(file)
        delay0 = l_seg[0] - l_ml[0]
        delay2 = l_seg[-1] - l_ml[-1]
        delay1 = 0.5 * (delay0 + delay2)
        transition_point = l_seg[1] - delay1

        result = [l_ml[0], transition_point, l_ml[1]]
    elif len(l_seg) == 2:
        result = [l_seg[0], 0, l_seg[1]]
    return result

class GripCodeCalc():
    def __init__(self):
        self.gripref_df = pd.DataFrame()
        self.gripref_df['code'] = [ i for i in range(1,26)] 
        self.gripref_df['G'] = [ 1.5875 * i for i in self.gripref_df['code'] ]
        self.gripref_df['GUL'] = [ x + 0.127 for x in self.gripref_df['G'] ] # added the tolerance of 0.127 mm
        self.gripref_df['GLL'] = np.concatenate([[0], self.gripref_df['GUL'][:-1]])# - 0.127]) # added the tolerance of 0.127 mm
        
    def length_code(self, depth):
        code = self.length_array([depth])
        return code[0]
 
    def length_array(self, depths):
        try:
            iterator = iter(depths)
        except TypeError:
            depths = [depths]
        else:
            pass
        code = np.dot(
            (np.array(depths)[:, None] > self.gripref_df['GLL'].values) &
            (np.array(depths)[:, None] <= self.gripref_df['GUL'].values),
            self.gripref_df['code'].values
        )
        return code

if __name__ == '__main__':
    import sys
    # print("Running abyss functions")
    # print(abyss_path)
    # gr_result = kp_recognition_gradient(f'{abyss_path}/test_data/17070141_17070141_ST_753_55.xls')
    # ml_result = kp_recognition_ml(f'{abyss_path}/test_data/17070141_17070141_ST_753_55.xls')
    # print(f"Gradient result: {gr_result}")
    # print(f"ML result: {ml_result}")
    
    if len(sys.argv) < 2:
        # abspath = r'C:\Users\NG9374C\Documents\uos-drilling\abyss\test_data\UNK\2312041_23120041_ST_1378_95.xls'
        # abspath = r'C:\Users\NG9374C\Documents\uos-drilling\abyss\test_data\UNK\MQTT_to_XLS_17070144_175b.xls'
        # abspath = r'C:\Users\NG9374C\Documents\uos-drilling\abyss\test_data\UNK\MQTT_to_XLS_17070141_798a.xls'
        abspath = r'C:\Users\NG9374C\Documents\uos-drilling\abyss\test_data\UNK\MQTT_to_XLS_17070144_30221.xls'
        # abspath = r'C:\Users\NG9374C\Documents\uos-drilling\abyss\test_data\UNK\22030021_4507-4504-E_ST_12905_1007.xls'
    else:
        abspath = sys.argv[1]
    # result = depth_est_segmented(abspath)
    # print("*********************************************")
    # print("Segmented Keypoint Recognition Gradient Results:")
    # print("*********************************************")
    # print(result)
    # print("*********************************************")
    # print("Machine Learning Results:")
    # print("*********************************************")
    # # result_ml = depth_est_ml(f'{abysspath}/test_data/{filename}')
    # result_ml = depth_est_ml(abspath)
    # print(result_ml)
    path=abspath
    dest_segmented = depth_est_segmented(path)
    print("depth_keypoints_segmented :"+str(dest_segmented))
    dest_ml = depth_est_ml(path)
    print("depth_keypoints_ml :"+str(dest_ml))
    dest = depth_est_combined(path)
    print("depth_keypoints_combined:"+str(dest))
    