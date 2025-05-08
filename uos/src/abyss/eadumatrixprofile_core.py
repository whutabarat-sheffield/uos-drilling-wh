from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import json
from pathlib import Path
import stumpy
from scipy.signal import find_peaks
from dtaidistance import dtw
# from tslearn.metrics import dtw
from abyss.dataparser import loadSetitecXls
import abyss.signal_template
import abyss.cleanup_template

########################################################
# CONSTANTS
########################################################
#template_extract = {'entry_cfrp' : (110, 110, 135), 'entry_ti' : (1500, 250, 1550), 'exit_ti' :(2200, 225, 2245) }, \
template_default = dict( \
        template_dataset='Dec22', template_local_counter = 2, \
        template_extract = {'entry_cfrp' : (110, 110, 135), 'entry_ti' : (1500, 250, 1550), 'exit_ti' :(2200, 225, 2260) }, \
        template_delete = {'delete_1' : (400, 80, 1), 'delete_2' : (2620, 80, 1)})

########################################################
# SIGNAL FUNCTIONS
########################################################

def get_depth_estimation_templates() -> dict: #copied
    ''' Returns the depth estimation templates.
    '''
    # Get the templates from the json file
    # with open('signal_templates.json', 'r') as fp:
    #     segments = json.load(fp, cls=NpDecoder)
    segments = signal_template.signal_template
    return segments

def get_signal_cleanup_templates() -> dict: #copied
    ''' Returns the signal cleanup templates.
    '''
    # Get the templates from the json file
    # with open('cleanup_templates.json', 'r') as fp:
    #     cleanup_segments = json.load(fp, cls=NpDecoder)
    cleanup_segments = cleanup_template.cleanup_template
    return cleanup_segments

# def find_match_from_template(segment: np.ndarray, target: np.ndarray) -> tuple:
#     ''' Returns the index of the first occurence of the target in the segment template.
#     Putting this as a separate function allows for more flexibility if we want to use other methods.
#     '''
#     match_distance_profile = stumpy.mass(segment, target, normalize=True, p=2.0)
#     start_idx = np.argmin(match_distance_profile)
#     return start_idx, match_distance_profile[start_idx], match_distance_profile


def find_match_from_template(segment: np.ndarray, target: np.ndarray,
                             normalize: bool = True, p: float = 2.0) -> tuple:
    '''
    Returns the index of the first occurrence of the target in the segment template.
    Putting this as a separate function allows for more flexibility if we want to use other methods.

    Args:
        segment (np.ndarray): The segment template to search for the target.
        target (np.ndarray): The target to search for within the segment template.
        normalize (bool): Whether or not to normalize the mass calculation.
                           Defaults to True.
        p (float): The order of the norm for the distance calculation. Defaults to 2.0.

    Returns:
        A tuple containing the index of the first occurrence of the target in the segment template,
        the distance of the match, and the distance profile of the match as a Numpy array.
    '''
    # Calculate the distance profile between the segment and the target
    match_distance_profile = stumpy.mass(segment, target, normalize=normalize, p=p)

    # Find the starting index of the best match
    start_idx = np.argmin(match_distance_profile)

    # Return a tuple containing the details of the match
    return start_idx, match_distance_profile[start_idx], match_distance_profile

def find_peak_matches_from_template(segment: np.ndarray, target: np.ndarray,
                             normalize: bool = True, p: float = 2.0, nreturn: int = 1) -> tuple:
    '''
    Returns the index of the first N occurrences of the target in the segment template.
    Putting this as a separate function allows for more flexibility if we want to use other methods.

    Args:
        segment (np.ndarray): The segment template to search for the target.
        target (np.ndarray): The target to search for within the segment template.
        normalize (bool): Whether or not to normalize the mass calculation.
                           Defaults to True.
        p (float): The order of the norm for the distance calculation. Defaults to 2.0.
        N (int): Number of peak locations to return. Defaults to 1.

    Returns:
        A list of tuples containing the index of the first occurrence of the target in the segment template,
        the distance of the match, and the distance profile of the match as a Numpy array.
    '''
    # Calculate the distance profile between the segment and the target
    match_distance_profile = stumpy.mass(segment, target, normalize=normalize, p=p)

    # # Find the starting index of the best match
    # start_idx = np.argmin(match_distance_profile)
    raise NotImplementedError

    # Return a tuple containing the details of the match
    return start_idx, match_distance_profile[start_idx], match_distance_profile


def cleanup_signal_single(segment, target, cleanup_value = np.nan):
    ''' Overwrites troublesome ranges in the signal with the cleanup_value.
    '''
    cleanup_target_segment_start_idx, _, _ = find_match_from_template(segment, target)
    cleanup_target_segment_end_idx = cleanup_target_segment_start_idx + len(segment)
    np.put(target, 
            np.arange(cleanup_target_segment_start_idx,cleanup_target_segment_end_idx), 
            [cleanup_value])
    return target


def cleanup_signal(target, cleanup_segments, cleanup_value = np.nan):
    ''' Overwrites troublesome ranges in the target signal with the cleanup_value.
    Returns the cleaned up signal.
    '''
    signal = np.array(target, dtype=np.float64)
    for key in cleanup_segments.keys():
        cleanup_segment = np.array(cleanup_segments[key]['segment_torque'],  dtype=np.float64)
        signal = cleanup_signal_single(cleanup_segment, signal, cleanup_value)
    return signal


def _cleanup_signal(source, target, cleanup_dict, cleanup_value = np.nan):
    ''' Overwrites troublesome ranges in the signal with the cleanup_value.
    '''
    for cleanup_key in cleanup_dict.keys():
        start, width = cleanup_dict[cleanup_key]
        cleanup_segment = source[start:start+width]
        cleanup_target_segment_start_idx, _, _ = find_match_from_template(cleanup_segment, target)
        cleanup_target_segment_end_idx = cleanup_target_segment_start_idx + width
        np.put(target, 
               np.arange(cleanup_target_segment_start_idx,cleanup_target_segment_end_idx), 
               [cleanup_value])
    return target


class NpEncoder(json.JSONEncoder):
    '''
    JSON encoder for numpy arrays.'''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

class NpDecoder(json.JSONDecoder):
    '''
    JSON decoder for numpy arrays.'''
    def default(self, obj):
        if isinstance(obj, int):
            return np.integer(obj)
        if isinstance(obj, float):
            return np.floating(obj)
        if isinstance(obj, list):
            return np.ndarray
        return super(NpDecoder, self).default(obj) 


def bottom_n_valleys_idx(yvals, n=10) -> tuple:
    """This is used for the traffic light system.
    Find N (default=10) valleys at the bottom of the signal.
    """
    y_temp = np.array(yvals)
    ## Avoid any issues with NaNs, TODO: check if the hardcoded values can be avoided
    y_filt = np.nan_to_num(y_temp, nan=-100, posinf=100)
    y_max = np.max(y_filt[y_filt != 100])
    ## Find the troughs by changing the sign and searching for peaks
    pks,_ = find_peaks(-1*y_temp, height=(-0.5*y_max,0))
    ## Find the indices for N lowest troughs
    idx_peaks = np.argsort(y_temp[pks])[:n]
    # y_peaks = np.sort(y_temp[pks])[:n]
    pks_selected = pks[idx_peaks]
    ## Find the N lowest troughs
    y_temp_selected = y_temp[pks_selected]
    return pks_selected, y_temp_selected


def match_z_score_idx(yvals, nvalleys=6) -> np.ndarray:
    """This is used for the traffic light system.
    Compares the lowest distance profile value with the mean of the next _nvalleys_ values
    Uses z-score method: z = (x – mean) / [standard deviation]"""
    idx_peaks, y_peaks = bottom_n_valleys_idx(yvals, nvalleys)
    z_score = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        z_score = np.abs(y_peaks[0] - np.mean(y_peaks[1:nvalleys])) / np.std(y_peaks[1:nvalleys])
    return z_score

def match_z_score_indices(yvals, nvalleys=6, nreturn=5) -> np.ndarray:
    """This is used for the traffic light system.
    Compares the _nreturn_ lowest distance profile values with the mean of the next _nvalleys_ values
    Uses z-score method: z = (x – mean) / [standard deviation]"""
    idx_peaks, y_peaks = bottom_n_valleys_idx(yvals, nreturn+nvalleys)
    z_scores = 5*[0]
    for i in range(nreturn):
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores[i] = np.abs(y_peaks[i] - np.mean(y_peaks[i+1:i+nvalleys])) / np.std(y_peaks[i+1:i+nvalleys])
    return z_scores


def index_matches(signal, templates, nvalleys=3):
    # TODO: use cosine distance instead of Euclidean distance 
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)
    signal = np.array(signal, dtype=np.float64)
    keypoint_indices = []
    distance_values = []
    distance_profiles = []
    z_scores = []
    for key in templates.keys():
        segment = np.array(templates[key]['segment_torque'], dtype=np.float64)
        segment_target_idx = np.array(templates[key]['segment_target_idx'], dtype=np.float64)
        # MATRIX PROFILE START
        target_segment_start_idx, distance_value, distance_profile = find_match_from_template(segment, signal)
        z_score = match_z_score_idx(distance_profile, nvalleys=nvalleys)
        # MATRIX PROFILE END
        keypoint_idx = target_segment_start_idx + segment_target_idx
        keypoint_indices.append(keypoint_idx)
        distance_values.append(distance_value)
        distance_profiles.append(np.array(distance_profile, dtype=np.float64))
        z_scores.append(z_score)
    return np.array(keypoint_indices, dtype=np.int32), np.array(distance_values, dtype=np.float64), distance_profiles, np.array(z_scores, dtype=np.float64)

def index_matches_simple(signal, templates):
    # TODO: use cosine distance instead of Euclidean distance 
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)
    signal = np.array(signal, dtype=np.float64)
    keypoint_indices = []
    distance_values = []
    distance_profiles = []
    z_scores = []
    for key in templates.keys():
        segment = np.array(templates[key]['segment_torque'], dtype=np.float64)
        segment_target_idx = np.array(templates[key]['segment_target_idx'], dtype=np.float64)
        # MATRIX PROFILE START
        target_segment_start_idx, distance_value, distance_profile = find_match_from_template(segment, signal)
        # MATRIX PROFILE END
        keypoint_idx = target_segment_start_idx + segment_target_idx
        keypoint_indices.append(keypoint_idx)
        distance_values.append(distance_value)
        distance_profiles.append(np.array(distance_profile, dtype=np.float64))
    return np.array(keypoint_indices, dtype=np.int32), np.array(distance_values, dtype=np.float64), distance_profiles


def index_matches_multi(signal, templates, nvalleys=3, nreturn=5):
    # TODO: use cosine distance instead of Euclidean distance 
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)
    signal = np.array(signal, dtype=np.float64)
    keypoint_indices = []
    distance_values = []
    distance_profiles = []
    z_scores = []
    for key in templates.keys():
        segment = np.array(templates[key]['segment_torque'], dtype=np.float64)
        segment_target_idx = np.array(templates[key]['segment_target_idx'], dtype=np.float64)
        # MATRIX PROFILE START
        target_segment_start_idxs, distance_value_retlist, distance_profile_retlist = find_peak_matches_from_template(segment, signal, nreturn=nreturn) # Changed to find_peak_matches_from_template

        # MATRIX PROFILE END
        # keypoint_idx = target_segment_start_idx + segment_target_idx
        # keypoint_indices.append(keypoint_idx)
        # distance_values.append(distance_value)
        # distance_profiles.append(np.array(distance_profile, dtype=np.float64))
        # z_scores.append(z_score)
        for i in range(nreturn):
            z_scores_retlist = match_z_score_indices(distance_profile_retlist[i], nvalleys=nvalleys, nreturn=nreturn) # Changed to match_z_score_indices
            keypoint_idx = target_segment_start_idxs[i] + segment_target_idx
            keypoint_indices.append(keypoint_idx)
            distance_values.append(distance_value_retlist[i])
            distance_profiles.append(np.array(distance_profile_retlist[i], dtype=np.float64))
            z_scores.append(z_scores_retlist[i])
    return np.array(keypoint_indices, dtype=np.int32), np.array(distance_values, dtype=np.float64), distance_profiles, np.array(z_scores, dtype=np.float64)


def index_matches_add_dtw(signal, templates, nvalleys=3):
    # TODO: use cosine distance instead of Euclidean distance 
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)
    signal = np.array(signal, dtype=np.float64)
    keypoint_indices = []
    distance_values = []
    distance_profiles = []
    z_scores = []
    dtw_scores = []
    for key in templates.keys():
        segment = np.array(templates[key]['segment_torque'], dtype=np.float64)
        segment_target_idx = np.array(templates[key]['segment_target_idx'], dtype=np.float64)
        # MATRIX PROFILE START
        target_segment_start_idx, distance_value, distance_profile = find_match_from_template(segment, signal)
        z_score = match_z_score_idx(distance_profile, nvalleys=nvalleys)
        dtw_score = dtw.distance(segment, signal[target_segment_start_idx:target_segment_start_idx+len(segment)])
        # MATRIX PROFILE END
        keypoint_idx = target_segment_start_idx + segment_target_idx
        keypoint_indices.append(keypoint_idx)
        distance_values.append(distance_value)
        distance_profiles.append(np.array(distance_profile, dtype=np.float64))
        z_scores.append(z_score)
        dtw_scores.append(dtw_score)
    return np.array(keypoint_indices, dtype=np.int32), np.array(distance_values, dtype=np.float64), distance_profiles, np.array(z_scores, dtype=np.float64), np.array(dtw_scores, dtype=np.float64)

# def depth_estimation_matrixprofile(filename_path) -> dict:
#     """Performs depth estimation using matrix profile.
#     """
#     filename = Path(filename_path).name
#     # Get the signals from the source file
#     position, torque = get_setitec_signals(filename_path)
#     # Get the templates from the json file
#     segments = get_depth_estimation_templates()
#     cleanup_segments = get_signal_cleanup_templates()
#     # Clean up the signal
#     clean_torque = cleanup_signal(torque, cleanup_segments)
#     # Identify the matches
#     # idxs, dvals, dparrays, zscores = index_matches(clean_torque, segments, nvalleys=3)
#     estimated_positions, orags, composites = position_matches(position, clean_torque, segments)
#     stack_depths, compo_results = calculate_estimated_depths_with_qualmetrics(estimated_positions, composites)
#     compo_rags = [convert_ordinal_rag_to_rag(convert_composite_score_to_ordinal_rag(compo_result)) for compo_result in compo_results]
#     z = [item for t in zip(stack_depths, compo_rags, compo_results) for item in t]
#     result = (filename,  *z)
#     keys = ['filename', 'depth_estimate', 'depth_estimate_rag', 'depth_estimate_quality',\
#             'cfrp_depth_estimate', 'cfrp_depth_estimate_rag', 'cfrp_depth_estimate_quality', \
#             'ti_depth_estimate', 'ti_depth_estimate_rag', 'ti_depth_estimate_quality']
#     return dict(zip(keys, result))


def depth_est_matrixprofile(torque, position, default_templates=True, simple_results=True):
    """Performs depth estimation using matrix profile on a specified file.

    If you only need the depth estimate, set simple_results to True. 
    This will return a single float value for the depth estimate.

    If you need the full results, including individual material depths 
    and the quality estimates, set simple_results to False. This will return 
    a dictionary with the following keys:
        depth_estimate (float): The estimated depth value.
        depth_estimate_rag (int): The ordinal ranking of the estimated depth value.
        depth_estimate_quality (float): The quality metric of the edef find_match_from_template(segment: np.ndarray, target: np.ndarray) -> tuple:
        cfrp_depth_estimate (float): The estimated depth value for CFRP material.
        cfrp_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for CFRP material.
        cfrp_depth_estimate_quality (float): The quality metric of the estimated depth value for CFRP material.
        ti_depth_estimate (float): The estimated depth value for TI material.
        ti_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for Ti material.
        ti_depth_estimate_quality (float): The quality metric of the estimated depth value for Ti material.

    Example (full stack thickness only):
        # Get the signals from the source file, put it into 'torque' and 'pos' variables
        total_borehole_depth_float = depth_est_matrixprofile(torque, pos, default_templates=True, simple_results=True)

    Example (full stack, individual stack thicknesses, and quality metrics):
        # Get the signals from the source file, put it into 'torque' and 'pos' variables
        full_stack_info_dict = depth_est_matrixprofile(torque, pos, default_templates=True, simple_results=False)
    

    """
    if default_templates:
        # Get templates from json file
        depth_estimation_templates = get_depth_estimation_templates()
        signal_cleanup_templates = get_signal_cleanup_templates()
    else:
        raise NotImplementedError
    
    # Clean up the signal
    cleaned_torque = cleanup_signal(torque, signal_cleanup_templates)
    
    # Identify matches
    estimated_positions, _, composites = position_matches(
        position, cleaned_torque, depth_estimation_templates)
    stack_depths, comp_results = calculate_estimated_depths_with_qualmetrics(
        estimated_positions, composites)
    comp_rags = [
        convert_ordinal_rag_to_rag(
            convert_composite_score_to_ordinal_rag(result))
        for result in comp_results]
    depth_estimate = [
        item for stack in zip(stack_depths, comp_rags, comp_results)
        for item in stack]   
    
    # Set keys and values
    keys = ['depth_estimate', 'depth_estimate_rag',
            'depth_estimate_quality', 'cfrp_depth_estimate',
            'cfrp_depth_estimate_rag', 'cfrp_depth_estimate_quality',
            'ti_depth_estimate', 'ti_depth_estimate_rag',
            'ti_depth_estimate_quality']
    if simple_results:
        # Return estimated total borehole depth only
        return depth_estimate[0]
    else:
        # Return dictionary
        depth_estimate_dict = dict(zip(keys, depth_estimate))
        depth_estimate_dict['estimated_positions'] = estimated_positions
        return depth_estimate_dict


def depth_est_matrixprofile_multi(torque, position, default_templates=True, simple_results=True):
    """Performs depth estimation using matrix profile on a specified file.

    If you only need the depth estimate, set simple_results to True. 
    This will return a single float value for the depth estimate.

    If you need the full results, including individual material depths 
    and the quality estimates, set simple_results to False. This will return 
    a dictionary with the following keys:
        depth_estimate (float): The estimated depth value.
        depth_estimate_rag (int): The ordinal ranking of the estimated depth value.
        depth_estimate_quality (float): The quality metric of the edef find_match_from_template(segment: np.ndarray, target: np.ndarray) -> tuple:
        cfrp_depth_estimate (float): The estimated depth value for CFRP material.
        cfrp_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for CFRP material.
        cfrp_depth_estimate_quality (float): The quality metric of the estimated depth value for CFRP material.
        ti_depth_estimate (float): The estimated depth value for TI material.
        ti_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for Ti material.
        ti_depth_estimate_quality (float): The quality metric of the estimated depth value for Ti material.

    Example (full stack thickness only):
        # Get the signals from the source file, put it into 'torque' and 'pos' variables
        total_borehole_depth_float = depth_est_matrixprofile(torque, pos, default_templates=True, simple_results=True)

    Example (full stack, individual stack thicknesses, and quality metrics):
        # Get the signals from the source file, put it into 'torque' and 'pos' variables
        full_stack_info_dict = depth_est_matrixprofile(torque, pos, default_templates=True, simple_results=False)
    

    """
    if default_templates:
        # Get templates from json file
        depth_estimation_templates = get_depth_estimation_templates()
        signal_cleanup_templates = get_signal_cleanup_templates()
    else:
        raise NotImplementedError
    
    # Clean up the signal
    cleaned_torque = cleanup_signal(torque, signal_cleanup_templates)
    
    # Identify matches
    estimated_positions, _, composites = position_matches(
        position, cleaned_torque, depth_estimation_templates)
    stack_depths, comp_results = calculate_estimated_depths_with_qualmetrics(
        estimated_positions, composites)
    comp_rags = [
        convert_ordinal_rag_to_rag(
            convert_composite_score_to_ordinal_rag(result))
        for result in comp_results]
    depth_estimate = [
        item for stack in zip(stack_depths, comp_rags, comp_results)
        for item in stack]
    
    # Set keys and values
    keys = ['depth_estimate', 'depth_estimate_rag',
            'depth_estimate_quality', 'cfrp_depth_estimate',
            'cfrp_depth_estimate_rag', 'cfrp_depth_estimate_quality',
            'ti_depth_estimate', 'ti_depth_estimate_rag',
            'ti_depth_estimate_quality']
    if simple_results:
        # Return estimated total borehole depth only
        return depth_estimate[0]
    else:
        # Return dictionary
        return dict(zip(keys, depth_estimate))
    
    

def depth_est_matrixprofile_debug(torque, position, default_templates=True, simple_results=True):
    """Performs depth estimation using matrix profile on a specified file.

    If you only need the depth estimate, set simple_results to True. 
    This will return a single float value for the depth estimate.

    If you need the full results, including individual material depths 
    and the quality estimates, set simple_results to False. This will return 
    a dictionary with the following keys:
        depth_estimate (float): The estimated depth value.
        depth_estimate_rag (int): The ordinal ranking of the estimated depth value.
        depth_estimate_quality (float): The quality metric of the edef find_match_from_template(segment: np.ndarray, target: np.ndarray) -> tuple:
        estimated_positions (list): The estimated positions of the matches.
        cfrp_depth_estimate (float): The estimated depth value for CFRP material.
        cfrp_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for CFRP material.
        cfrp_depth_estimate_quality (float): The quality metric of the estimated depth value for CFRP material.
        ti_depth_estimate (float): The estimated depth value for TI material.
        ti_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for Ti material.
        ti_depth_estimate_quality (float): The quality metric of the estimated depth value for Ti material.

    Example (full stack thickness only):
        # Get the signals from the source file, put it into 'torque' and 'pos' variables
        total_borehole_depth_float = depth_est_matrixprofile(torque, pos, default_templates=True, simple_results=True)

    Example (full stack, individual stack thicknesses, and quality metrics):
        # Get the signals from the source file, put it into 'torque' and 'pos' variables
        full_stack_info_dict = depth_est_matrixprofile(torque, pos, default_templates=True, simple_results=False)
    

    """
    if default_templates:
        # Get templates from json file
        depth_estimation_templates = get_depth_estimation_templates()
        signal_cleanup_templates = get_signal_cleanup_templates()
        # Get templates from json file
        depth_estimation_templates = get_depth_estimation_templates()
        signal_cleanup_templates = get_signal_cleanup_templates()
    else:
        raise NotImplementedError
    
    # Clean up the signal
    cleaned_torque = cleanup_signal(torque, signal_cleanup_templates)
    
    # Identify matches
    estimated_positions, _, composites = position_matches_debug(
        position, cleaned_torque, depth_estimation_templates)
    stack_depths, comp_results = calculate_estimated_depths_debug(
        estimated_positions, composites)
    comp_rags = [
        convert_ordinal_rag_to_rag(
            convert_composite_score_to_ordinal_rag(result))
        for result in comp_results]
    depth_estimate = [
        item for stack in zip(stack_depths, comp_rags, comp_results)
        for item in stack]
    
    # Set keys and values
    keys = ['depth_estimate', 'depth_estimate_rag',
            'depth_estimate_quality', 'estimated_positions',
            'cfrp_depth_estimate',
            'cfrp_depth_estimate_rag', 'cfrp_depth_estimate_quality',
            'ti_depth_estimate', 'ti_depth_estimate_rag',
            'ti_depth_estimate_quality']
    if simple_results:
        # Return estimated total borehole depth only
        return depth_estimate[0]
    else:
        # Return dictionary
        depth_estimate_dict = dict(zip(keys, depth_estimate))
        depth_estimate_dict['estimated_positions'] = estimated_positions
        return depth_estimate_dict


def depth_est_xls_matrixprofile(file_path, default_templates=True, simple_results=True) -> dict:
    """Performs depth estimation using matrix profile on a specified file.

    Args:
        file_path (str): The path to the file to be processed.

    Returns:
        A dictionary with the following keys:
            filename (str): The name of the processed file.
            depth_estimate (float): The estimated depth value.
            depth_estimate_rag (int): The ordinal ranking of the estimated depth value.
            depth_estimate_quality (float): The quality metric of the edef find_match_from_template(segment: np.ndarray, target: np.ndarray) -> tuple:
            cfrp_depth_estimate (float): The estimated depth value for CFRP material.
            cfrp_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for CFRP material.
            cfrp_depth_estimate_quality (float): The quality metric of the estimated depth value for CFRP material.
            ti_depth_estimate (float): The estimated depth value for TI material.
            ti_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for Ti material.
            ti_depth_estimate_quality (float): The quality metric of the estimated depth value for Ti material.
    """

    # Extract filename from file path
    filename = Path(file_path).name
    
    # Get signals from source file
    position, torque = get_setitec_signals(file_path)

    if simple_results:
        return_value = depth_est_matrixprofile(torque, position, default_templates=default_templates, simple_results=simple_results)
    else:
        return_value = depth_est_matrixprofile(torque, position, default_templates=default_templates, simple_results=simple_results)
        return_value['filename'] = filename
    
    return return_value


def _depth_est_xls_matrixprofile(file_path) -> dict:
    """Performs depth estimation using matrix profile on a specified file.

    Args:
        file_path (str): The path to the file to be processed.

    Returns:
        A dictionary with the following keys:
            filename (str): The name of the processed file.
            depth_estimate (float): The estimated depth value.
            depth_estimate_rag (int): The ordinal ranking of the estimated depth value.
            depth_estimate_quality (float): The quality metric of the edef find_match_from_template(segment: np.ndarray, target: np.ndarray) -> tuple:
            cfrp_depth_estimate (float): The estimated depth value for CFRP material.
            cfrp_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for CFRP material.
            cfrp_depth_estimate_quality (float): The quality metric of the estimated depth value for CFRP material.
            ti_depth_estimate (float): The estimated depth value for TI material.
            ti_depth_estimate_rag (int): The ordinal ranking of the estimated depth value for Ti material.
            ti_depth_estimate_quality (float): The quality metric of the estimated depth value for Ti material.
    """

    # Extract filename from file path
    filename = Path(file_path).name
    
    # Get signals from source file
    position, torque = get_setitec_signals(file_path)
    
    # Get templates from json file
    depth_estimation_templates = get_depth_estimation_templates()
    signal_cleanup_templates = get_signal_cleanup_templates()
    
    # Clean up the signal
    cleaned_torque = cleanup_signal(torque, signal_cleanup_templates)
    
    # Identify matches
    estimated_positions, _, composites = position_matches(
        position, cleaned_torque, depth_estimation_templates)
    stack_depths, comp_results = calculate_estimated_depths_with_qualmetrics(
        estimated_positions, composites)
    comp_rags = [
        convert_ordinal_rag_to_rag(
            convert_composite_score_to_ordinal_rag(result))
        for result in comp_results]
    depth_estimate = [
        item for stack in zip(stack_depths, comp_rags, comp_results)
        for item in stack]
    
    # Set keys and values
    keys = ['filename', 'depth_estimate', 'depth_estimate_rag',
            'depth_estimate_quality', 'cfrp_depth_estimate',
            'cfrp_depth_estimate_rag', 'cfrp_depth_estimate_quality',
            'ti_depth_estimate', 'ti_depth_estimate_rag',
            'ti_depth_estimate_quality']
    values = [filename, *depth_estimate]
    
    # Return dictionary
    return dict(zip(keys, values))


def convert_scores_to_ordinal_rag(dval, z_score):
    composite_score = dval / z_score
    if composite_score < 1.4: # return 'green'
        # dval < 1.2 and z_score > 3:
        return 2
    elif composite_score < 1.75: # return 'amber'
        # dval < 1.5 and z_score > 2:
        return 1
    else: # return 'red'
        return 0


def convert_ordinal_rag_to_rag(rag_ordinal):
    if rag_ordinal == 2:
        return 'green'
        # return 2
    elif rag_ordinal == 1:
        return 'yellow'
        # return 1
    elif rag_ordinal == 0:
        return 'red'
        # return 0
    else:
        raise ValueError(f'Invalid rag_ordinal value: {rag_ordinal}')


def convert_scores_to_rag(dval, z_score):
    return convert_ordinal_rag_to_rag(convert_scores_to_ordinal_rag(dval, z_score))


def convert_composite_score_to_ordinal_rag(composite_score):
    if composite_score < 1.4: # return 'green'
        # dval < 1.2 and z_score > 3:
        return 2
    elif composite_score < 1.75: # return 'amber'
        # dval < 1.5 and z_score > 2:
        return 1
    else: # return 'red'
        return 0


def position_matches(position, signal, templates, nvalleys=3):
    """
    This function takes in three arguments:
    position: A numpy array containing the positions of keypoints.
    signal: A numpy array containing the signal.
    templates: A list of numpy arrays containing the templates.

    This function returns three lists:
    keypoint_positions: A list of positions of keypoints.
    rag_ordinals: A list of ordinal RAGs.
    rags: A list of RAGs.
    """
    
    # Convert position and signal to numpy arrays
    position = np.array(position)
    signal = np.array(signal)
    
    # Initialize empty lists
    keypoint_positions = []
    rag_ordinals = []
    composite_metric = []
    
    # Get keypoint indices, distance values, and z-scores
    keypoint_indices, distance_values, _, z_scores = index_matches(signal, templates, nvalleys=nvalleys)
    
    # Loop through keypoint indices, distance values, and z-scores
    for keypoint_idx, dval, z_score in zip(keypoint_indices, distance_values, z_scores):
        
        # Get keypoint position
        keypoint_position = position[keypoint_idx]
        # Append keypoint position to keypoint_positions list
        keypoint_positions.append(keypoint_position)
        
        # Convert distance value and z-score to ordinal RAG
        rag_ordinal = convert_scores_to_ordinal_rag(dval, z_score)
        # Append ordinal RAG to rag_ordinals list
        rag_ordinals.append(rag_ordinal)
        
        # Convert metrics to composite metric
        composite = dval / z_score
        # Append composite metric to the list
        composite_metric.append(composite)
    
    # Return keypoint_positions, rag_ordinals, and rags
    return keypoint_positions, rag_ordinals, composite_metric

def position_matches_simple(position, signal, templates):
    """
    This function takes in three arguments:
    position: A numpy array containing the positions of keypoints.
    signal: A numpy array containing the signal.
    templates: A list of numpy arrays containing the templates.

    This function returns three lists:
    keypoint_positions: A list of positions of keypoints.
    rag_ordinals: A list of ordinal RAGs.
    rags: A list of RAGs.
    """
    
    # Convert position and signal to numpy arrays
    position = np.array(position)
    signal = np.array(signal)
    
    # Initialize empty lists
    keypoint_positions = []
    
    # Get keypoint indices, distance values, and z-scores
    keypoint_indices, distance_values, _ = index_matches_simple(signal, templates)
    
    # Loop through keypoint indices, distance values, and z-scores
    for keypoint_idx, dval in zip(keypoint_indices, distance_values):
        
        # Get keypoint position
        keypoint_position = position[keypoint_idx]
        # Append keypoint position to keypoint_positions list
        keypoint_positions.append(keypoint_position)
    
    # Return keypoint_positions, rag_ordinals, and rags
    return keypoint_positions

def position_matches_multi(position, signal, templates, nvalleys=3, nreturn=5):
    """
    This function takes in four arguments:
    position: A numpy array containing the positions of keypoints.
    signal: A numpy array containing the signal.
    templates: A list of numpy arrays containing the templates.
    nvalleys: number of valleys to use for z-score calculation
    nreturn: number of matches to return

    This function returns three lists:
    keypoint_positions: A list of positions of keypoints.
    rag_ordinals: A list of ordinal RAGs.
    rags: A list of RAGs.
    """
    
    # Convert position and signal to numpy arrays
    position = np.array(position)
    signal = np.array(signal)
    
    # Initialize empty lists
    keypoint_positions = []
    rag_ordinals = []
    composite_metric = []
    
    # Get keypoint indices, distance values, and z-scores
    keypoint_indices, distance_values, _, z_scores = index_matches_multi(signal, templates, nvalleys=nvalleys)
    
    # Loop through keypoint indices, distance values, and z-scores
    for keypoint_idx, dval, z_score in zip(keypoint_indices, distance_values, z_scores):
        
        # Get keypoint position
        keypoint_position = position[keypoint_idx]
        # Append keypoint position to keypoint_positions list
        keypoint_positions.append(keypoint_position)
        
        # Convert distance value and z-score to ordinal RAG
        rag_ordinal = convert_scores_to_ordinal_rag(dval, z_score)
        # Append ordinal RAG to rag_ordinals list
        rag_ordinals.append(rag_ordinal)
        
        # Convert metrics to composite metric
        composite = dval / z_score
        # Append composite metric to the list
        composite_metric.append(composite)
    
    # Return keypoint_positions, rag_ordinals, and rags
    return keypoint_positions, rag_ordinals, composite_metric

def position_matches_debug(position: np.ndarray, signal: np.ndarray, templates: List[np.ndarray], nvalleys: int = 3) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """
    This function takes in three arguments:
    position: A numpy array containing the positions of keypoints.
    signal: A numpy array containing the signal.
    templates: A list of numpy arrays containing the templates.
    nvalleys: An integer representing the number of valleys. Default is 3.

    This function returns four lists:
    keypoint_positions: A list of positions of keypoints.
    dval_metric: A list of distance values.
    zscore_metric: A list of z-scores.
    dtwscore_metric: A list of dtw scores.
    """
    
    # Convert position and signal to numpy arrays
    position = np.array(position)
    signal = np.array(signal)
    
    # Initialize empty lists
    keypoint_positions = []
    dval_metric = []
    zscore_metric = []
    dtwscore_metric = []
    
    # Get keypoint indices, distance values, and z-scores
    keypoint_indices, distance_values, _, z_scores, dtw_scores = index_matches_add_dtw(signal, templates, nvalleys=nvalleys)
    
    # Loop through keypoint indices, distance values, and z-scores
    for keypoint_idx, dval, z_score, dtw_score in zip(keypoint_indices, distance_values, z_scores, dtw_scores):
        
        # Get keypoint position
        keypoint_position = position[keypoint_idx]
        # Append keypoint position to keypoint_positions list
        keypoint_positions.append(keypoint_position)

        # Return the raw scores
        dval_metric.append(dval)
        zscore_metric.append(z_score)
        dtwscore_metric.append(dtw_score)        
    
    # Return keypoint_positions, rag_ordinals, and rags
    return keypoint_positions, dval_metric, zscore_metric, dtwscore_metric


def calculate_estimated_depths_debug(positions, metrics):
    """
    Calculate estimated total borehole depth followed by individual stack depths.

    Args:
        positions (list): A list of positions.
        ordinal_rags (list): A list of estimated quality metrics.

    Returns:
        tuple: A tuple containing:
            - stack_depths (list): A list of stack depths.
            - orag_results (list): A list of encoded red amber green results [0, 1, 2].
            - rag_results (list): A list of red amber green results ['red', 'amber', 'green'].
    """
    # Full borehole depth
    end_to_end_depth = positions[-1] - positions[0]
    end_to_end_comp_result = max(metrics[-1], metrics[0])
    # Stack depth
    stack_depths = [end_to_end_depth]
    comp_results = [end_to_end_comp_result]
    for i in range(len(positions) - 1):
        stack_depths.append(positions[i+1] - positions[i])
        comp_result = max(metrics[i+1], metrics[i])
        comp_results.append(comp_result)
    return stack_depths, comp_results


def calculate_estimated_depths_with_qualmetrics(positions, composites):
    """
    Calculate estimated total borehole depth followed by individual stack depths.

    Args:
        positions (list): A list of positions.
        ordinal_rags (list): A list of estimated quality metrics.

    Returns:
        tuple: A tuple containing:
            - stack_depths (list): A list of stack depths.
            - orag_results (list): A list of encoded red amber green results [0, 1, 2].
            - rag_results (list): A list of red amber green results ['red', 'amber', 'green'].
    """
    # Full borehole depth
    end_to_end_depth = positions[-1] - positions[0]
    end_to_end_comp_result = max(composites[-1], composites[0])
    # Stack depth
    stack_depths = [end_to_end_depth]
    comp_results = [end_to_end_comp_result]
    for i in range(len(positions) - 1):
        stack_depths.append(positions[i+1] - positions[i])
        comp_result = max(composites[i+1], composites[i])
        comp_results.append(comp_result)
    return stack_depths, comp_results

def calculate_estimated_depths_with_rags(positions, ordinal_rags):
    """
    Calculate estimated total borehole depth followed by individual stack depths.

    Args:
        positions (list): A list of positions.
        ordinal_rags (list): A list of ordinal RAG values.

    Returns:
        tuple: A tuple containing:
            - stack_depths (list): A list of stack depths.
            - orag_results (list): A list of encoded red amber green results [0, 1, 2].
            - rag_results (list): A list of red amber green results ['red', 'amber', 'green'].
    """
    # Full borehole depth
    end_to_end_depth = positions[-1] - positions[0]
    end_to_end_orag_result = min(ordinal_rags[-1], ordinal_rags[0])
    # Stack depth
    stack_depths = [end_to_end_depth]
    orag_results = [end_to_end_orag_result]
    rag_results = [convert_ordinal_rag_to_rag(end_to_end_orag_result)]
    for i in range(len(positions) - 1):
        stack_depths.append(positions[i+1] - positions[i])
        orag_result = min(ordinal_rags[i+1], ordinal_rags[i])
        orag_results.append(orag_result)
        rag_results.append(convert_ordinal_rag_to_rag(orag_result))
    return stack_depths, orag_results, rag_results


# def keypoint_matches(position, signal, templates):
def keypoint_matches(position: List[float], signal: List[float], templates: List[Tuple[List[float], int]]) -> Tuple[List[float], List[float]]: 
    ''' Find keypoint matches between a signal and a set of templates.
    Args:
        position (List[float]): The positions corresponding to each element of the signal.
        signal (List[float]): The signal to match against the templates.
        templates (List[Tuple[List[float], int]]): The templates to match against the signal, along with the target index within each template.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists: the positions of the matched keypoints and the traffic lights for each keypoint.

    The function iterates through each template and finds the closest segment in the signal using the matrix profile technique. It then calculates the keypoint position by taking the target segment index and adding it to the index of the closest segment. Finally, it appends the keypoint position and the corresponding traffic light value to the respective lists.

    Note: The function assumes that the position and signal lists have the same length, and each template is represented as a tuple containing the segment (list of floats) and the target index (integer) within that segment.

    Docstring prepared by CodeGPT.
    
    '''

    position = np.array(position)
    signal = np.array(signal)
    keypoint_positions = []
    traffic_lights = []

    for segment, segment_target_idx in templates:
        # find the closest segment to the target
        # MATRIX PROFILE START
        distance_profile = stumpy.mass(segment, signal, normalize=True)
        target_segment_start_idx = np.argmin(distance_profile)
        traffic_light = match_z_score_idx(distance_profile)
        # MATRIX PROFILE END
        # target_segment_end_idx = target_segment_start_idx + segment_width
        keypoint_position_idx = target_segment_start_idx + segment_target_idx
        keypoint_position = position[keypoint_position_idx]
        keypoint_positions.append(keypoint_position)
        traffic_lights.append(traffic_light)

    return keypoint_positions, traffic_lights


# def get_setitec_signals(file_to_open):



def get_setitec_signals(file_to_open: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Retrieve position and torque signals from a Setitec file.

    Args:
        file_to_open (str): The path of the Setitect file to open.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays: the position signal and the combined torque signal.

    Note:
        - The position is obtained from the 'Position (mm)' column of the DataFrame.
        - The torque signal is calculated by adding the 'I Torque (A)' and 'I Torque Empty (A)' columns of the DataFrame.

    '''

    df = loadSetitecXls(file_to_open, version="auto_data")
    position = np.abs(df['Position (mm)'].values)
    torque = df['I Torque (A)'].values
    torque_empty = df['I Torque Empty (A)'].values
    torque_full = torque + torque_empty
    return position, torque_full


def create_template_segments(position: List[float], signal: List[float], template_dict: Dict[str, Tuple[int, int, int]]) -> Dict[str, Dict[str, Any]]:
    '''
    Create template segments from the position and signal data.

    Args:
        position (List[float]): The positions corresponding to each element of the signal.
        signal (List[float]): The signal from which to create the template segments.
        template_dict (Dict[str, Tuple[int, int, int]]): A dictionary containing the template names as keys and a tuple 
                                                        of the segment start index, segment width, and segment target index as values.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the template segments.

    '''

    source_segments = {}

    for segment_name in template_dict.keys():
        segment_start = template_dict[segment_name][0]
        segment_width = template_dict[segment_name][1]
        segment_target = template_dict[segment_name][2]

        segment_end = segment_start + segment_width
        segment_torque = signal[segment_start:segment_end]
        segment_position = position[segment_start:segment_end]
        segment_range = np.arange(segment_start, segment_end)

        source_segments[segment_name] = {
            'segment_range': segment_range,
            'segment_target_idx': segment_target-segment_start,
            'segment_position': segment_position,
            'segment_torque': segment_torque
        }
        
    return source_segments


# def calculate_estimated_segments(position, torque, idxs, template=template_default):
def calculate_estimated_segments(position: List[float], 
                                 torque: List[float], 
                                 idxs: List[int], 
                                 template: Dict[str, Tuple[int, int, int]] = template_default) -> Tuple[List[List[float]], List[List[float]]]:
    """Calculate estimated segments based on the provided position, torque, and template information. 

    Args:
        position (List[float]): The position array.
        torque (List[float]): The torque array.
        idxs (List[int]): The list of estimated target indices.
        template (Dict[str, Tuple[int, int, int]]): The template dictionary containing segment start index, range index, and target index.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: A tuple containing the estimated positions and estimated torque segments.
    """

    estimated_positions = []
    estimated_segments = []

    for i, key in enumerate(template.keys()):
        start_idx, range_idx, target_idx = template[key]

        estimated_target_idx = idxs[i]
        delta_start = estimated_target_idx - target_idx

        estimated_start_idx = start_idx + delta_start
        estimated_end_idx = estimated_start_idx + range_idx
            
        estimated_segments.append(torque[estimated_start_idx:estimated_end_idx])
        estimated_positions.append(position[estimated_start_idx:estimated_end_idx])

    return estimated_positions, estimated_segments


# def calculate_quotient_remainder(borehole_depth, fastener_unit_length=1.5875):
#     return borehole_depth // fastener_unit_length, borehole_depth % fastener_unit_length


def calculate_quotient_remainder(borehole_depth: float, fastener_unit_length: float = 1.5875) -> Tuple[int, float]:
    """
    Calculate the quotient and remainder when dividing the borehole depth by the fastener unit length.

    Args:
        borehole_depth (float): The depth of the borehole.
        fastener_unit_length (float): The length of the fastener unit. Defaults to 1.5875.

    Returns:
        Tuple[int, float]: A tuple containing the quotient and remainder of the division.

    Raises:
        TypeError: If borehole_depth or fastener_unit_length is not a float.

    Example:
        calculate_quotient_remainder(10.0, 2.5)
        # Output: (4, 0.0)
    """
    if not isinstance(borehole_depth, float):
        raise TypeError("borehole_depth must be a float.")
    if not isinstance(fastener_unit_length, float):
        raise TypeError("fastener_unit_length must be a float.")

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


def depth_est_xls_persegment_stats(file_path, bit_depth=0.7, stat = 'median', simple_results=False, debug=False):
    # Get signals from source file
    # bit_depth = 2.0 # mm, this needs to be supplied externally to be accurate
    pos, torque = get_setitec_signals(file_path)
    return depth_est_persegment_stats(torque, pos, bit_depth=bit_depth, stat=stat, simple_results=simple_results, debug=debug)

def depth_est_persegment_stats(torque, pos, bit_depth=0.7, stat = 'median', simple_results=False, debug=False):
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
