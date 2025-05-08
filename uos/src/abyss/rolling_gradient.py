import json
import os
import warnings
import math
from glob import glob
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy import integrate
from scipy import interpolate as interp
from scipy.signal import find_peaks, get_window, peak_prominences, peak_widths, wiener
from scipy.signal.windows import tukey

from .plotting import plot_depthest, combineLegends
from . import dataparser as dp
# from abyss.uos_depth_est_core import *
from . import uos_depth_est_core as uos


# pre sorted paths to save time
PATHS = sorted(glob("AirbusData/Seti-Tec data files life test/*.xls"),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3])

# custom warning to inform the user when no peaks have been found
class MissingPeaksWarning(Warning):
    def __init__(self,message):
        self.message = message
    def __str__(self):
        return repr(self.message)

# custom exception for when no peaks have been found
class MissingPeaksException(Exception):
    def __init__(self,*args,**kwargs):
        if not args: args = ("Failed to find peaks in target window!",)
        super().__init__(*args,**kwargs)

# custom warning to inform the user when the mask is empty
class EmptyMaskWarning(Warning):
    def __init__(self,message):
        self.message = message
    def __str__(self):
        return repr(self.message)

# custom exception for when the mask is empty/doesn't select any values
class EmptyMaskException(Exception):
    def __init__(self,*args,**kwargs):
        if not args: args = ("Mask to define window is empty!",)
        super().__init__(*args,**kwargs)

# lots of scipy functions tend to raise RuntimeWarnings about invalid divisions and multiplocations
warnings.simplefilter(action='ignore',category=RuntimeWarning)

############################# FUNCTIONS #############################
# from https://stackoverflow.com/questions/67997826/pandas-rolling-gradient-improving-reducing-computation-time
def get_slope(df):
    # drop na values
    df = df.dropna()
    # find the minimum index
    min_idx = df.index.min()
    x = df.index - min_idx
    y = df.values.flatten()
    slope,_ = np.polyfit(x,y,1)
    return slope

def get_slope_v2(df):
    # drop na values
    df = df.dropna()
    min_idx = df.index.min()
    max_idx = df.index.max()
    return (df.values.flatten()[-1] - df.values.flatten()[0])/(max_idx - min_idx)

# calculate the abs normed rolling gradient
def rolling_gradient(y,N,norm=False,keep_signs=True):
    '''
        Calculate the rolling gradient and post-process it

        Uses pandas.rolling to calculate the rolling windows and apply the funciton get_slope
        to each window returning the rolling gradient.

        The flags control how the gradient is processed. The norm flag, normalizes the gradient to 0-1.
        By default the gradient is absoluted as it's used in find_peaks which prefers all +ve values.
        If keep_signs is True, then it's not abs.

        Inputs:
            y : Full input vector
            N : Size of rolling windows
            norm : Flag to normalize the gradient. Default False.
            keep_signs : Flag to not abs the gradient. Default True

        Returns the processed rolling gradient.
    '''
    if isinstance(y,np.ndarray):
        y = pd.Series(y)
    # calculate rolling gradient
    slope = y.rolling(N).apply(get_slope,raw=False).values.flatten()
    # replace NaNs with 0.0
    np.nan_to_num(slope,copy=False)
    # keep the sign
    if not keep_signs:
        return np.abs(slope)
    return slope

def rolling_gradient_v2(y,N,norm=False,keep_signs=True):
    '''
        Calculate the rolling gradient and post-process it

        Uses pandas.rolling to calculate the rolling windows and apply the funciton get_slope
        to each window returning the rolling gradient.

        The flags control how the gradient is processed. The norm flag, normalizes the gradient to 0-1.
        By default the gradient is absoluted as it's used in find_peaks which prefers all +ve values.
        If keep_signs is True, then it's not abs.

        Inputs:
            y : Full input vector
            N : Size of rolling windows
            norm : Flag to normalize the gradient. Default False.
            keep_signs : Flag to not abs the gradient. Default True

        Returns the processed rolling gradient.
    '''
    if isinstance(y,np.ndarray):
        y = pd.Series(y)
    # calculate rolling gradient
    slope = y.rolling(N).apply(get_slope_v2,raw=False).values.flatten()
    # replace NaNs with 0.0
    np.nan_to_num(slope,copy=False)
    # keep the sign
    if not keep_signs:
        return np.abs(slope)
    return slope

def rolling_variance(y,N,keep_signs=True):
    '''
        Calculate the rolling variance and post-process it

        Uses pandas.rolling to calculate the rolling windows and apply the funciton get_slope
        to each window returning the rolling gradient.

        The flags control how the gradient is processed. The norm flag, normalizes the gradient to 0-1.
        By default the gradient is absoluted as it's used in find_peaks which prefers all +ve values.
        If keep_signs is True, then it's not abs.

        Inputs:
            y : Full input vector
            N : Size of rolling windows
            norm : Flag to normalize the gradient. Default False.
            keep_signs : Flag to not abs the gradient. Default True

        Returns the processed rolling gradient.
    '''
    if isinstance(y,np.ndarray):
        y = pd.Series(y)
    # calculate rolling gradient
    slope = y.rolling(N).apply(get_slope,raw=False).values.flatten()
    # replace NaNs with 0.0
    np.nan_to_num(slope,copy=False)
    # keep the sign
    if not keep_signs:
        return np.abs(slope)
    return slope

def rolling_gradient_integ(y, N):
    """
    Calculates the rolling gradient of a given signal and integrates it back up to the original signal.

    Parameters:
    y (array-like): The input signal.
    N (int): The window size for calculating the rolling gradient.

    Returns:
    array-like: The integrated signal.

    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    # calculate rolling gradient
    slope = y.rolling(N).apply(get_slope, raw=False).values.flatten()
    # replace NaNs with 0.0
    np.nan_to_num(slope, copy=False)
    # integrate back up to the original signal
    y_hat = integrate.cumulative_trapezoid(slope)
    # add offset and return
    return y_hat + y.min()

# rolling gradient going forwards and backwards
# combine the slopes together according to mode
def fr_gradient(ydata,N,mode="sum"):
    slopeF = pd.Series(ydata).rolling(N).apply(get_slope,raw=False).values.flatten()
    slopeB = -1*pd.Series(ydata[::-1]).rolling(N).apply(get_slope,raw=False).values.flatten()
    if mode == "sum":
        slope = slopeF+slopeB
    elif mode in ["average","avg","mean"]:
        slope = (slopeF+slopeB)/2
    else:
        slope = mode(slopeF,slopeB)
    return slope

# rolling gradient going forwards and backwards
# combine the slopes together according to mode
# integrate signal to smooth
def fr_gradient_smooth(ydata,N,mode="sum"):
    # perform forward rolling slope
    slopeF = pd.Series(ydata).rolling(N).apply(get_slope,raw=False).values.flatten()
    # flip the data and perform rolling slope
    # since slopes are mirrored, reverse the direction for backwards
    slopeB = -1*pd.Series(ydata[::-1]).rolling(N).apply(get_slope,raw=False).values.flatten()
    # if mode is sum
    # sum the slopes together
    if mode == "sum":
        slope = slopeF+slopeB
    # if mean or average then average the data together
    elif mode in ["average","avg","mean"]:
        slope = (slopeF+slopeB)/2
    # for everything else
    # assume mode is a callable entity
    else:
        slope = mode(slopeF,slopeB)
    return integrate.cumulative_trapezoid(slope)

# rolling gradient but in reverse
def rolling_gradient_r(ydata,N):
    # calculate rolling gradient
    slope = y.rolling(N).apply(get_slope,raw=False).values.flatten()[::-1]
    # replace NaNs with 0.0
    np.nan_to_num(slope,copy=False)
    slope = np.abs(slope)
    # normalize
    if norm:
        slope -= slope.min()
        slope /= slope.max()
    return slope[::-1]

# slightly changed version of scipy's fit_edge
# addition of keyword window which is applied as series of weights when evaluating polynomial
# type of window is set either as string or valid numpy array
def _fit_edge_weight(x, window_start, window_stop, interp_start, interp_stop,
              axis, polyorder, deriv, delta, y, window='hann'):
    """
    Given an N-d array `x` and the specification of a slice of `x` from
    `window_start` to `window_stop` along `axis`, create an interpolating
    polynomial of each 1-D slice, and evaluate that polynomial in the slice
    from `interp_start` to `interp_stop`. Put the result into the
    corresponding slice of `y`.
    """
    from scipy.signal import get_window
    from scipy.signal._arraytools import axis_slice
    # Get the edge into a (window_length, -1) array.
    x_edge = axis_slice(x, start=window_start, stop=window_stop, axis=axis)
    if axis == 0 or axis == -x.ndim:
        xx_edge = x_edge
        swapped = False
    else:
        xx_edge = x_edge.swapaxes(axis, 0)
        swapped = True
    xx_edge = xx_edge.reshape(xx_edge.shape[0], -1)
    # get the desired window
    # if it's a string then use get_window function
    if isinstance(window,str):
        win = get_window(window,xx_edge.shape[0])
    # else if it's a numpy array
    elif isinstance(window,np.ndarray):
        win = window
    else:
        raise ValueError(f"Weight Window has to either be a string supported by scipy.signal.get_window or a numpy array. Received {window}!")
    # Fit the edges.  poly_coeffs has shape (polyorder + 1, -1),
    # where '-1' is the same as in xx_edge.
    # supplying window as weights to use
    poly_coeffs = np.polyfit(np.arange(0, window_stop - window_start),
                             xx_edge, polyorder,w=win)

    if deriv > 0:
        poly_coeffs = _polyder(poly_coeffs, deriv)

    # Compute the interpolated values for the edge.
    i = np.arange(interp_start - window_start, interp_stop - window_start)
    values = np.polyval(poly_coeffs, i.reshape(-1, 1)) / (delta ** deriv)

    # Now put the values into the appropriate slice of y.
    # First reshape values to match y.
    shp = list(y.shape)
    shp[0], shp[axis] = shp[axis], shp[0]
    values = values.reshape(interp_stop - interp_start, *shp[1:])
    if swapped:
        values = values.swapaxes(0, axis)
    # Get a view of the data to be replaced by values.
    y_edge = axis_slice(y, start=interp_start, stop=interp_stop, axis=axis)
    y_edge[...] = values
    return y_edge

# windowed savgol filter adapted from scipy code
# applies a weight window when evaluating the polynomial
def weighted_savgol_filter(x, window_length, polyorder=1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0, window="hann"):
    from scipy.signal._savitzky_golay import savgol_coeffs
    from scipy.ndimage import convolve1d
    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)
    # compute the coefficients
    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
    if mode == "interp":
        if window_length > x.size:
            raise ValueError("If mode is 'interp', window_length must be less "
                             "than or equal to the size of x.")

        # Do not pad. Instead, for the elements within `window_length // 2`
        # of the ends of the sequence, use the polynomial that is fitted to
        # the last `window_length` elements.
        y = convolve1d(x, coeffs, axis=axis, mode="constant")
        # same as function _fit_edges_polyfit but uses modified version of _fit_edge
        # saves redefining it
        # Use polynomial interpolation of x at the low and high ends of the axis
        # to fill in the halflen values in y.
        halflen = window_length // 2
        _fit_edge_weight(x, 0, window_length, 0, halflen, axis,
                  polyorder, deriv, delta, y, window)
        n = x.shape[axis]
        _fit_edge_weight(x, n - window_length, n, n - halflen, n, axis,
                  polyorder, deriv, delta, y, window)
    else:
        # Any mode other than 'interp' is passed on to ndimage.convolve1d.
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)

    return y

def calcTorqueNM(I,V=48.0,E=0.75,w=6000.0):
    wrad = w * 2.0*np.pi/60.0
    return (I*V*E)/wrad

def depth_find_first_ref(ydata,xdata=None,method='weiner',NA=20,NB=None,xstart=10.0,filt_grad=True,use_signs=True,hh=0.1,pselect='argmax'):
    # if window size for rolling gradient is not set
    # use the same as the function
    if (NB is None) and (NA is not None):
        NB = NA
    elif (NB is not None) and (NA is None):
        NA = NB
    else:
        raise ValueError(f"Need to supply a proper window size for both smoothing filter and rolling gradient! NA={NA}, NB={NB}")
    # if the xdata is not set
    # create an artificial set based on length of data
    if xdata is None:
        xdata = np.arange(len(ydata))
    # filter data using target method
    if method == 'weiner':
        filt_weight = wiener(ydata,NA)
    elif method == 'savgol':
        filt_weight = weighted_savgol_filter(ydata,NA,1,deriv=0, window=window)
    # perform rolling gradient on smoothed data
    grad = rolling_gradient(filt_weight,NB,keep_signs=use_signs)
    # if it's a flag
    # create tukey window
    if isinstance(filt_grad,bool) and filt_grad:
        win = tukey(len(grad),0.1,True)
        grad *= win
    # if the user gave a window
    # apply that instead
    elif isinstance(filt_grad,np.ndarray):
        grad *= filt_grad
    ## filter first window
    # create mask
    mask = xdata <= xstart
    # if mask doesn't select and values
    # set first reference as first value
    if np.where(mask)[0].shape[0] ==0:
        msg = f"Empty mask for 2nd window!\nxstart={xstart},xmax={xmax}, NA={NA}, NB={NB}"
        if default:
            warnings.warn(msg,category=EmptyMaskWarning)
            xA = xdata[0]
        else:
            raise EmptyMaskException(msg)
    else:
        # mask gradient to first period
        grad_mask = grad[mask]
        xmask = xdata[mask]
        # if using the signs
        # set negative gradient to zero
        # during the start of the signal the signal is ramping up as the tool sides change
##        if use_signs:
##            grad_mask *= -1.0
        # get max gradient value
        hlim = grad_mask.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad_mask, height=hh*hlim)
        # if no peaks were found
        # use location of max
        if len(pks)==0:
            msg = f"No peaks found for first period {xdata.min()} to {xstart} NA={NA}, NB={NB}! Defaulting to max value in period"
            if default:
                warnings.warn(msg,category=MissingPeaksWarning)
                return xmask[grad_mask.argmax()]
            else:
                raise MissingPeaksException(msg[:msg.find("!")+1])
        else:
            # then the maximum peak within the period is used
            if (pselect == 'argmax') or (pselect == 'max'):
                pkA = grad_mask[pks].argmax()
                # find correspondng x value
                return xmask[pks][pkA]
            # if using the first value
            elif (pselect == 'limit') or (pselect == 'first'):
                return xmask[pks][0]
            # if using the last value
            elif pselect == 'last':
                return xmask[pks][-1]
            # if user gave something unsupported
            # default to argmax
            else:
                msg = f"Unsupported peak selection mode {pselect}. Defaulting to argmax for first period"
                if default:
                    warnings.warn(msg)
                    # find where the highest peak occurs
                    pkA = grad_mask[pks].argmax()
                    # find correspondng x value
                    return grad_mask[pks][pkA]
                else:
                    raise ValueError(msg[:msg.find("!")+1])

def depth_est_rolling(ydata,xdata=None,method='wiener',NA=20,NB=None,xstart=10.0,hh=0.1,pselect='argmax',filt_grad=True,default=True,end_ref='end',window="hann",**kwargs):
    '''
        Depth Estimate using rolling gradient of smoothed signal

        Smoothes the signal using the method specified by method parameter and performs rolling gradient on the result.

        There is also an option to apply a weighting window to the rolling gradient to help remove edge artifacts caused by ripples at the start/end of the
        signal not removed by the smoothing. The window is specified by the filt_grad keyword. If True (default) then a Tukey window from scipy is applied with an
        alpha of 0.1 which found to perform well in testing. A numpy array the same length as the gradient can be given too.

        Peaks are then detected in two specified periods in the start and end of the signal.

        The first period is specified as

        xdata <= xstart

        The definition of the second period is specified based on the keyword end_ref. If the window is set to be relative to the end of the signal ('end')
        then the mask is defined as follows.

        (xdata >= (xdata.max()-xendA)) & (xdata <= (xdata.max()-xendB))

        If end_ref is 'start' then the mask is defined as follows.

        (xdata >= (xdata.min()+xendA)) & (xdata <= (xdata.min()+xendB))

        The idea is to provide a way of including the nominal material thickness e.g. xendA = 32.0 and xendB = 32.0 + 5.0 where 5mm is a buffer region allowing for dead space.

        Another method to include depth estimate is with the depth_exp and depth_win. In this approach the 2nd window is defined as

        (xdata >= (xA + depth_exp - (depth_win/2))) & (xdata >= (xA + depth_exp + (depth_win/2)))

        This can be easier to define than with xendA and xendB.

        The peaks detected are filtered to those above a fraction (hh) of the max gradient. This is passed to the scipy find_peaks function under the
        height keyword.

        pks,_ = find_peaks(grad,height=hh*grad.max())

        The filtering condition is designed to remove the peaks detected on the noise floor.

        Inputs:
            ydata : An iterable collection of ydata
            xdata : An iterate collection of xdata to accompany ydata. If None then it's constructed from the indicies of the ydata.
            method : Function to smooth the signal. Supported strings are wiener and savgol
            NA : Window size of the smoothing function
            NB : Window size for rolling gradient
            xstart : Upper limit of starting boundary
            xendA : Lower limit of ending boundary period
            xendB : Upper limit of ending bounary period
            hh : Fraction of max gradient to use to filter peaks
            window : Window type to use as weights in savgol filter. Can be a string supported by scipy get_window or an array of weights of the correct size. Default hann.
            pselect : Method to select peaks within the target periods. Default argmax. Supported modes
                argmax, max: Select the max height peak in the target period
                first : Choose first peak.
                last : Choose last peak.
                limit : Choose first peak for first period and last peak for second period.
            filt_grad : Filter the rolling gradient using a window. This is to help avoid edge artifacts that would results in a bad depth estimate. Can be a flag or a numpy
                        array representing the values. If True, then a Tukey window with alpha of 0.1 is used
            end_ref : String indicating whether to use the start or end of the signal for the 2nd window references. Supports 'start' or 'end' referencing the start or end of the signal.
            default : Flag to default to certain values if the algorithm fails to mask any values. For the first window, the starting reference is set to the first X value. For the 2nd window
                        the reference is set to the last value. If False, then if it fails to find a reference point then a ValueError is raised. Default True.
            try_gmin : Flag to use where the minimum gradient occurs as the 2nd reference point. This useful when the expected depth is not known. WARNING: THIS IS USEFUL WHEN THE 2nd MATERIAL
                        IS HARDER THAN THE 1st AND CAUSES A LARGE CHANGE IN THE SUPPLIED CURRENT WHEN FINISHED DRILLING

        Returns the depth estimate in mm
    '''
    # if window size for rolling gradient is not set
    # use the same as the function
    if (NB is None) and (NA is not None):
        NB = NA
    elif (NB is not None) and (NA is None):
        NA = NB
    else:
        raise ValueError(f"Need to supply a proper window size for both rolling gradient and smoothing filter! NA={NA}, NB={NB}")
    # if the xdata is not set
    # create an artificial set based on length of data
    if xdata is None:
        xdata = np.arange(len(ydata))
    # filter data using target method
    if method == 'wiener':
        filt_weight = wiener(ydata,NA)
    elif method == 'savgol':
        filt_weight = weighted_savgol_filter(ydata,NA,1,deriv=0, window=window)
    else:
        raise ValueError(f"Unsupported filtering method {method}!")
    # perform rolling gradient on smoothed data
    #grad = rolling_gradient(filt_weight,NB,keep_signs=True)
    grad = rolling_gradient_v2(filt_weight,NB)
    # if it's a flag
    # create tukey window
    if isinstance(filt_grad,bool) and filt_grad:
        win = tukey(len(grad),0.1,True)
        grad *= win
    # if the user gave a window
    # apply that instead
    elif isinstance(filt_grad,np.ndarray):
        grad *= filt_grad
    ## filter first window
    # create mask
    mask = xdata <= xstart
    # if mask doesn't select and values
    # set first reference as first value
    if np.where(mask)[0].shape[0] ==0:
        if default:
            warnings.warn(f"Empty mask for 1st window!\nxstart={xstart},xmax={xmax}, NA={NA}, NB={NB}\nDefaulting to first value {xdata[0]}",category=EmptyMaskWarning)
            xA = xdata[0]
        # else raise exception if default is False
        else:
            raise EmptyMaskException(f"Empty mask for 1st window!\nxstart={xstart},xmax={xmax}, NA={NA}, NB={NB}")
    else:
        # mask gradient to first period
        grad_mask = grad[mask]
        xmask = xdata[mask]
        # get max gradient value
        hlim = grad_mask.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad_mask, height=hh*hlim)                    
        # if no peaks were found
        if len(pks)==0:
            # if defaulting set to where max peak occurs
            if default:
                warnings.warn(f"No peaks found for 1st window {xdata.min()} to {xstart}, NA={NA}, NB={NB}!",category=MissingPeaksWarning)
                xA = xmask[grad_mask.argmax()]
            else:
                raise MissingPeaksException(f"No peaks found for 1st window {xdata.min()} to {xstart}, NA={NA}, NB={NB}! Defaulting to max value in window")
        else:
            # then the maximum peak within the period is used
            if (pselect == 'argmax') or (pselect == 'max'):
                pkA = grad_mask[pks].argmax()
                # find correspondng x value
                xA = xmask[pks][pkA]
            # if using the first value
            elif (pselect == 'limit') or (pselect == 'first'):
                xA = xmask[pks][0]
            # if using the last value
            elif pselect == 'last':
                xA = xmask[pks][-1]
            # if user gave something unsupported
            # default to argmax
            else:
                if default:
                    warnings.warn(f"Unsupported peak selection mode {pselect}. Defaulting to argmax for first period")
                    # find where the highest peak occurs
                    pkA = grad_mask[pks].argmax()
                    # find correspondng x value
                    xA = grad_mask[pks][pkA]
                else:
                    raise ValueError(f"Unsupported peak selection mode {pselect}! Should be either argmax,max,limit,first or last")
    # depth correction value
    dist_corr = 0.0
    # if the user wants to correct distance using rolling correction
    if kwargs.get('correct_dist',False):
        # if the user hasn't specified either the index where the change occurs or the entire step vector
        if (not ('change_idx' in kwargs)) and (not ('step_vector' in kwargs)):
            warnings.warn("Missing change index or step vector! Cannot correct distance using rolling peak correction!")
        else:
            # if the user specified the data index where it changed between two steps
            if 'change_idx' in kwargs:
                sc_min = kwargs['change_idx']
            # if the user gave the vector of step codes from the file            
            elif 'step_vector' in kwargs:
                sc = kwargs['step_vector']
                # if there are less then two step codes then we can't find where it transitions
                if sc.shape[0]<2:
                    warnings.warn(f"{fn} data does not contain multiple step codes so cannot correct!")
                else:
                    # find the first index where 2nd step code occurs
                    # IOW where it transtiioned between step codes
                    sc_min = data[data['Step (nb)'] == sc[1]].index.min()
            # get where the step code changes from 0 to 1
            dep = xdata[sc_min]
            cwin = kwargs.get("correct_win",1.0)
            # mask data to v. small search window
            cmask = (xdata >= (dep-cwin)) & (xdata <= (dep+cwin))
            xdata_filt = xdata[cmask]
            torque_filt = ydata[cmask]
            grad_cmask = grad[cmask]
            # set distance correction to distance between step code transition and max gradient within the narrow search window
            gi = grad_cmask.argmax()
            # if the max gradient occurs at the very end, warn user that the correction is likely to not be helpful
            # as the gradient peak tends to be within the search window so if it's towards the end, it likely failed to find the actual peak
            if gi == (grad_mask.shape[0]-1):
                warnings.warn("Found gradient peak at the end the vector! Depth correction likely incorrect")
                dist_corr = 0.0
            else:
                dist_corr = abs(dep - xdata_filt[gi])
    ## filter end period
    # calculate max x value
    xmax = xdata.max()
    # calculate min x value
    xmin = xdata.min()
    # if the user has given two specific reference points
    if ('xendB' in kwargs) and ('xendA' in kwargs):
        # extract values
        xendB = kwargs['xendB']
        xendA = kwargs['xendA']
        # if the reference points are relative to the end of the file
        if end_ref == 'end':
            # create mask for target period
            mask = (xdata >= (xmax-xendA))&(xdata <= (xmax-xendB))
        # else it's relative to the start
        elif end_ref == 'start':
            # create mask for target period
            mask = (xdata >= (xmin+xendA))&(xdata <= (xmin+xendB))
        else:
            raise ValueError(f"Unsupported ending reference {end_ref}! Must be either start or end")
    # if the user has given an expected depth estimate
    elif 'depth_exp' in kwargs:
        # expected depth estimate to help select target window
        # forms middle point of window
        depth_exp = kwargs['depth_exp']
        # if the user deliberately gave None
        if depth_exp is None:
            raise ValueError(f"Expected depth cannot be None! Received {depth_exp}")
        # target window size around mid point
        depth_win = kwargs.get('depth_win',None)
        # if the user didn't supply an accompanying window size
        if depth_win is None:
            raise ValueError("Missing depth window depth_win to pair with expected depth!")
        # if the user provided a negative window
        # a neg window is likely to create an empty window and raise an error
        elif depth_win < 0:
            raise ValueError(f"Target window period cannot be negative! Received {depth_win}")
        # create mask starting from the first reference point+expected depth estimate
        # with a window period around it
        mask= (xdata >= (xA+depth_exp-(depth_win/2)))&(xdata <= (xA+depth_exp+(depth_win/2)))
    # if using minimum gradient as 2nd point
    # calculate depth estimate as distance between xA and where the minimum gradient occurs
    elif kwargs.get('try_gmin',True):
        return xdata[grad.argmin()] - xA + dist_corr
    # if the user hasn't specified a way to define the 2nd period to search
    # raise error
    else:
        raise ValueError("2nd window not specified! Need to specify either xendA & xendB, depth_exp & depth_win or try_gmin.")
    # if the mask is empty
    if np.where(mask)[0].shape[0]==0:
        # if set to default to a value
        if default:
            # if user specified window period
            if ('xendB' in kwargs) and ('xendA' in kwargs):
                # relative to the end
                # take the upper limit
                if end_ref == 'end':
                    warnings.warn(f"All empty mask for 2nd window!\nxendA={xendA},xendB={xendB},end_ref={end_ref},xmax={xmax}. Defaulting to {(xmax - xendB) - xA}",category=EmptyMaskWarning)
                    return (xmax - xendB) - xA + dist_corr
                elif end_ref == 'start':
                    warnings.warn(f"All empty mask for 2nd window!\nxendA={xendA},xendB={xendB},end_ref={end_ref},xmax={xmax}. Defaulting to {(xmax + xendB) -xA}",category=EmptyMaskWarning)
                    return (xmax + xendB) -xA + dist_corr
            # if specified from expected depth estimate
            # take as upper end of the window period
            elif 'depth_exp' in kwargs:
                warnings.warn(f"All empty mask for 2nd window depth_exp={depth_exp}, win={depth_win}, xA={xA}!\nDefaulting to {xA+depth_exp + (depth_win/2)}",category=EmptyMaskWarning)
                #xB = xA+depth_exp+(win/2)
                return depth_exp+(depth_win/2) - xA + dist_corr  
        # if not set to default raise an exception
        # useful for debugging
        else:
            if ('xendB' in kwargs) and ('xendA' in kwargs): 
                raise EmptyMaskException(f"All empty mask for 2nd window!\nxendA={xendA},xendB={xendB},end_ref={end_ref},xmax={xmax}")
            elif 'depth_exp' in kwargs:
                raise EmptyMaskException(f"All empty mask for 2nd window depth_exp={depth_exp}, win={depth_win}!")
    else: 
        # mask gradient values
        grad_mask = grad[mask]
        # mask xdata
        xmask = xdata[mask]
        # invert to turn -ve gradients to +ve
        grad_mask *= -1.0
        # find max gradient in period
        hlim = grad_mask.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad_mask,height=hh*hlim)
        # if no peaks where found
        if len(pks)==0:
            # if defaulting to a set value
            if default:
                if ('xendB' in kwargs) and ('xendA' in kwargs):
                    warnings.warn(f"Number of peaks found of 2nd window is 0 for 2nd window {xendA} and {xendB}, NA={NA}, NB={NB} for reference {end_ref}! Defaulting to max location",category=MissingPeaksWarning)
                elif 'depth_exp' in kwargs:
                    warnings.warn(f"Number of peaks found of 2nd window is 0 for 2nd window {depth_exp} win={depth_win}, NA={NA}, NB={NB}! Defaulting to max location",category=MissingPeaksWarning)
                # default to the max value
                return xmask[grad_mask.argmax()] - xA + dist_corr
            # raise exception clipping the part after the exclamation mark
            else:
                if ('xendB' in kwargs) and ('xendA' in kwargs):
                    raise MissingPeaksException(f"Number of peaks found of 2nd window is 0 for 2nd period {xendA} and {xendB} for reference {end_ref}!")
                elif 'depth_exp' in kwargs:
                    raise MissingPeaksException(f"Number of peaks found of 2nd window is 0 for 2nd period {depth_exp} win={depth_win}, xA={xA}!")
        # if peaks were found
        else:
            # retrieve the method for choosing the peaks
            if (pselect == 'argmax') or (pselect == 'max'):
                pkB = grad_mask[pks].argmax()
                # find correspondng x value
                return xmask[pks][pkB] - xA + dist_corr
            # get last peak if specified or as part of limit
            elif (pselect == 'limit') or (pselect == 'last'):
                return xmask[pks][-1] - xA + dist_corr
            # use first peak
            elif pselect == 'first':
                return xmask[pks][0] - xA + dist_corr
            # if unsupported default to argmax
            else:
                if default:
                    warnings.warn(f"Unsupported peak selection mode {pselect}. Defaulting to argmax for 2nd window")
                    pkB = grad_mask[pks].argmax()
                    # find correspondng x value
                    return grad_mask[pks][pkB] - xA + dist_corr
                else:
                    raise ValueError(f"Unsupported peak selection mode {pselect}!")

def depth_est_rolling_slim(ydata,xdata=None,NA=20,NB=None,xstart=10.0,depth_exp=20.0,**kwargs):
    '''
        Depth Estimate using rolling gradient of smoothed signal

        Smoothes the signal using the method specified by method parameter and performs rolling gradient on the result.

        There is also an option to apply a weighting window to the rolling gradient to help remove edge artifacts caused by ripples at the start/end of the
        signal not removed by the smoothing. The window is specified by the filt_grad keyword. If True (default) then a Tukey window from scipy is applied with an
        alpha of 0.1 which found to perform well in testing. A numpy array the same length as the gradient can be given too.

        Peaks are then detected in two specified periods in the start and end of the signal.

        The first period is specified as

        xdata <= xstart

        The definition of the second period is specified based on the keyword end_ref. If the window is set to be relative to the end of the signal ('end')
        then the mask is defined as follows.

        (xdata >= (xdata.max()-xendA)) & (xdata <= (xdata.max()-xendB))

        If end_ref is 'start' then the mask is defined as follows.

        (xdata >= (xdata.min()+xendA)) & (xdata <= (xdata.min()+xendB))

        The idea is to provide a way of including the nominal material thickness e.g. xendA = 32.0 and xendB = 32.0 + 5.0 where 5mm is a buffer region allowing for dead space.

        Another method to include depth estimate is with the depth_exp and depth_win. In this approach the 2nd window is defined as

        (xdata >= (xA + depth_exp - (depth_win/2))) & (xdata >= (xA + depth_exp + (depth_win/2)))

        This can be easier to define than with xendA and xendB.

        The peaks detected are filtered to those above a fraction (hh) of the max gradient. This is passed to the scipy find_peaks function under the
        height keyword.

        pks,_ = find_peaks(grad,height=hh*grad.max())

        The filtering condition is designed to remove the peaks detected on the noise floor.

        Inputs:
            ydata : An iterable collection of ydata
            xdata : An iterate collection of xdata to accompany ydata. If None then it's constructed from the indicies of the ydata.
            method : Function to smooth the signal. Supported strings are wiener and savgol
            NA : Window size of the smoothing function
            NB : Window size for rolling gradient
            xstart : Upper limit of starting boundary
            xendA : Lower limit of ending boundary period
            xendB : Upper limit of ending bounary period
            hh : Fraction of max gradient to use to filter peaks
            window : Window type to use as weights in savgol filter. Can be a string supported by scipy get_window or an array of weights of the correct size. Default hann.
            pselect : Method to select peaks within the target periods. Default argmax. Supported modes
                argmax, max: Select the max height peak in the target period
                first : Choose first peak.
                last : Choose last peak.
                limit : Choose first peak for first period and last peak for second period.
            filt_grad : Filter the rolling gradient using a window. This is to help avoid edge artifacts that would results in a bad depth estimate. Can be a flag or a numpy
                        array representing the values. If True, then a Tukey window with alpha of 0.1 is used
            end_ref : String indicating whether to use the start or end of the signal for the 2nd window references. Supports 'start' or 'end' referencing the start or end of the signal.
            default : Flag to default to certain values if the algorithm fails to mask any values. For the first window, the starting reference is set to the first X value. For the 2nd window
                        the reference is set to the last value. If False, then if it fails to find a reference point then a ValueError is raised. Default True.
            try_gmin : Flag to use where the minimum gradient occurs as the 2nd reference point. This useful when the expected depth is not known. WARNING: THIS IS USEFUL WHEN THE 2nd MATERIAL
                        IS HARDER THAN THE 1st AND CAUSES A LARGE CHANGE IN THE SUPPLIED CURRENT WHEN FINISHED DRILLING

        Returns the depth estimate in mm
    '''
    # if window size for rolling gradient is not set
    # use the same as the function
    if (NB is None) and (NA is not None):
        NB = NA
    elif (NB is not None) and (NA is None):
        NA = NB
    else:
        raise ValueError(f"Need to supply a proper window size for both rolling gradient and smoothing filter! NA={NA}, NB={NB}")
    depth_win=4.0
    # if the xdata is not set
    # create an artificial set based on length of data
    if xdata is None:
        xdata = np.arange(len(ydata))
    # filter data using target method
    filt_weight = wiener(ydata,NA)
    # perform rolling gradient on smoothed data
    #grad = rolling_gradient(filt_weight,NB,keep_signs=True)
    grad = rolling_gradient_v2(filt_weight,NB)

    win = tukey(len(grad),0.1,True)
    grad *= win
    ## filter first window
    # create mask
    mask = xdata <= xstart
    # if mask doesn't select and values
    # set first reference as first value
    if np.where(mask)[0].shape[0] ==0:
        raise EmptyMaskException(f"Empty mask for 1st window!\nxstart={xstart},xmax={xmax}, NA={NA}, NB={NB}")
    else:
        # mask gradient to first period
        grad_mask = grad[mask]
        xmask = xdata[mask]
        # get max gradient value
        hlim = grad_mask.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad_mask, height=0.1*hlim)                    
        # if no peaks were found
        if len(pks)==0:
            raise MissingPeaksException(f"No peaks found for 1st window {xdata.min()} to {xstart}, NA={NA}, NB={NB}! Defaulting to max value in window")
        else:
            pkA = grad_mask[pks].argmax()
            # find correspondng x value
            xA = xmask[pks][pkA]
    # depth correction value
    dist_corr = 0.0
    # if the user wants to correct distance using rolling correction
    if kwargs.get('correct_dist',False):
        # if the user hasn't specified either the index where the change occurs or the entire step vector
        if (not ('change_idx' in kwargs)) and (not ('step_vector' in kwargs)):
            warnings.warn("Missing change index or step vector! Cannot correct distance using rolling peak correction!")
        else:
            # if the user specified the data index where it changed between two steps
            if 'change_idx' in kwargs:
                sc_min = kwargs['change_idx']
            # if the user gave the vector of step codes from the file            
            elif 'step_vector' in kwargs:
                sc = kwargs['step_vector']
                # if there are less then two step codes then we can't find where it transitions
                if sc.shape[0]<2:
                    warnings.warn(f"{fn} data does not contain multiple step codes so cannot correct!")
                else:
                    # find the first index where 2nd step code occurs
                    # IOW where it transtiioned between step codes
                    sc_min = data[data['Step (nb)'] == sc[1]].index.min()
            # get where the step code changes from 0 to 1
            dep = xdata[sc_min]
            # mask data to v. small search window
            cmask = (xdata >= (dep-1.0)) & (xdata <= (dep+1.0))
            xdata_filt = xdata[cmask]
            torque_filt = ydata[cmask]
            grad_cmask = grad[cmask]
            # set distance correction to distance between step code transition and max gradient within the narrow search window
            gi = grad_cmask.argmax()
            # if the max gradient occurs at the very end, warn user that the correction is likely to not be helpful
            # as the gradient peak tends to be within the search window so if it's towards the end, it likely failed to find the actual peak
            if gi == (grad_mask.shape[0]-1):
                warnings.warn("Found gradient peak at the end the vector! Depth correction likely incorrect")
                dist_corr = 0.0
            else:
                dist_corr = abs(dep - xdata_filt[gi])
    ## filter end period
    # calculate max x value
    xmax = xdata.max()
    # calculate min x value
    xmin = xdata.min()
    mask= (xdata >= (xA+depth_exp-(depth_win/2)))&(xdata <= (xA+depth_exp+(depth_win/2)))
    # if the mask is empty
    if np.where(mask)[0].shape[0]==0:
        raise EmptyMaskException(f"All empty mask for 2nd window depth_exp={depth_exp}, win={depth_win}!")
    else: 
        # mask gradient values
        grad_mask = grad[mask]
        # mask xdata
        xmask = xdata[mask]
        # invert to turn -ve gradients to +ve
        grad_mask *= -1.0
        # find max gradient in period
        hlim = grad_mask.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad_mask,height=0.1*hlim)
        # if no peaks where found
        if len(pks)==0:
            raise MissingPeaksException(f"Number of peaks found of 2nd window is 0 for 2nd period {depth_exp} win={depth_win}, xA={xA}!")
        # if peaks were found
        else:
            pkB = grad_mask[pks].argmax()
            # find correspondng x value
            return xmask[pks][pkB] - xA + dist_corr

def depth_est_rolling_evaltime(ydata,xdata=None,method='wiener',NA=20,NB=None,xstart=10.0,hh=0.1,pselect='argmax',filt_grad=True,default=True,end_ref='end',window="hann",**kwargs):
    from timeit import default_timer as timer
    
    # if window size for rolling gradient is not set
    # use the same as the function
    if (NB is None) and (NA is not None):
        NB = NA
    elif (NB is not None) and (NA is None):
        NA = NB
    else:
        raise ValueError(f"Need to supply a proper window size for both rolling gradient and smoothing filter! NA={NA}, NB={NB}")
    # if the xdata is not set
    # create an artificial set based on length of data
    if xdata is None:
        xdata = np.arange(len(ydata))
    # filter data using target method
    start = timer()
    if method == 'wiener':
        filt_weight = wiener(ydata,NA)
    elif method == 'savgol':
        filt_weight = weighted_savgol_filter(ydata,NA,1,deriv=0, window=window)
    else:
        raise ValueError(f"Unsupported filtering method {method}!")
    end = timer()
    print(f"Smoothing filter {method} N={NA}: {end-start}s")
    # perform rolling gradient on smoothed data
    start = timer()
    grad = rolling_gradient_v2(filt_weight,NB,keep_signs=True)
    end = timer()
    print(f"Rolling Gradient N={NB}: {end-start}s")
    # if it's a flag
    # create tukey window
    start = timer()
    if isinstance(filt_grad,bool) and filt_grad:
        win = tukey(len(grad),0.1,True)
        grad *= win
    # if the user gave a window
    # apply that instead
    elif isinstance(filt_grad,np.ndarray):
        grad *= filt_grad
    end = timer()
    print(f"Tukey Filter: {end-start}s")
    ## filter first window
    # create mask
    mask = xdata <= xstart
    # if mask doesn't select and values
    # set first reference as first value
    if np.where(mask)[0].shape[0] ==0:
        if default:
            warnings.warn(f"Empty mask for 1st window!\nxstart={xstart},xmax={xmax}, NA={NA}, NB={NB}\nDefaulting to first value {xdata[0]}",category=EmptyMaskWarning)
            xA = xdata[0]
        # else raise exception if default is False
        else:
            raise EmptyMaskException(f"Empty mask for 1st window!\nxstart={xstart},xmax={xmax}, NA={NA}, NB={NB}")
    else:
        start = timer()
        # mask gradient to first period
        grad_mask = grad[mask]
        xmask = xdata[mask]
        # get max gradient value
        hlim = grad_mask.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad_mask, height=hh*hlim)                    
        # if no peaks were found
        if len(pks)==0:
            # if defaulting set to where max peak occurs
            if default:
                warnings.warn(f"No peaks found for first period {xdata.min()} to {xstart}, NA={NA}, NB={NB}!",category=MissingPeaksWarning)
                xA = xmask[grad_mask.argmax()]
            else:
                raise MissingPeaksException(f"No peaks found for first period {xdata.min()} to {xstart}, NA={NA}, NB={NB}! Defaulting to max value in period")
        else:
            # then the maximum peak within the period is used
            if (pselect == 'argmax') or (pselect == 'max'):
                # if the user gave a correction factor
                if 'cfact' in kwargs:
                    # get value
                    cfact = kwargs['cfact']
                    # if the user deliberately gave None
                    if cfact is None:
                        raise ValueError(f"Correction factor cannot be None! Received {cfact}")
                    # check that the value is between 0 and 1
                    if (cfact<0) or (cfact>1.0):
                        raise ValueError(f"Relative height has to be between 0 and 1")
                    # calculate widths using relative height
                    results_full = peak_widths(grad_mask,pks,rel_height=cfact)[2]
                    # if any of the widths turn out as NaNs
                    if any(np.isnan(results_full)):
                        warnings.warn("Skipping widths due to NaNs")
                        pkA = grad_mask[pks].argmax()
                        xA = xmask[pks][pkA]
                    else:
                        pkA = grad_mask[pks].argmax()
                        xA = xmask[int(results_full[pkA])]
                else:
                    pkA = grad_mask[pks].argmax()
                    # find correspondng x value
                    xA = xmask[pks][pkA]
            # if using the first value
            elif (pselect == 'limit') or (pselect == 'first'):
                xA = xmask[pks][0]
            # if using the last value
            elif pselect == 'last':
                xA = xmask[pks][-1]
            # if user gave something unsupported
            # default to argmax
            else:
                if default:
                    warnings.warn(f"Unsupported peak selection mode {pselect}. Defaulting to argmax for first period")
                    # find where the highest peak occurs
                    pkA = grad_mask[pks].argmax()
                    # find correspondng x value
                    xA = grad_mask[pks][pkA]
                else:
                    raise ValueError(f"Unsupported peak selection mode {pselect}! Should be either argmax,max,limit,first or last")
        end = timer()
        print(f"Finding first peak: {end-start}s")
    ## filter end period
    # calculate max x value
    xmax = xdata.max()
    # calculate min x value
    xmin = xdata.min()
    start = timer()
    # if the user has given two specific reference points
    if ('xendB' in kwargs) and ('xendA' in kwargs):
        # extract values
        xendB = kwargs['xendB']
        xendA = kwargs['xendA']
        # if the reference points are relative to the end of the file
        if end_ref == 'end':
            # create mask for target period
            mask = (xdata >= (xmax-xendA))&(xdata <= (xmax-xendB))
        # else it's relative to the start
        elif end_ref == 'start':
            # create mask for target period
            mask = (xdata >= (xmin+xendA))&(xdata <= (xmin+xendB))
        else:
            raise ValueError(f"Unsupported ending reference {end_ref}! Must be either start or end")
    # if the user has given an expected depth estimate
    elif 'depth_exp' in kwargs:
        # expected depth estimate to help select target window
        # forms middle point of window
        depth_exp = kwargs['depth_exp']
        # if the user deliberately gave None
        if depth_exp is None:
            raise ValueError(f"Expected depth cannot be None! Received {depth_exp}")
        # target window size around mid point
        depth_win = kwargs.get('depth_win',None)
        # if the user didn't supply an accompanying window size
        if depth_win is None:
            raise ValueError("Missing depth window depth_win to pair with expected depth!")
        # if the user provided a negative window
        # a neg window is likely to create an empty window and raise an error
        elif depth_win < 0:
            raise ValueError(f"Target window period cannot be negative! Received {depth_win}")
        # create mask starting from the first reference point+expected depth estimate
        # with a window period around it
        mask= (xdata >= (xA+depth_exp-(depth_win/2)))&(xdata <= (xA+depth_exp+(depth_win/2)))
    # if using minimum gradient as 2nd point
    # calculate depth estimate as distance between xA and where the minimum gradient occurs
    elif kwargs.get('try_gmin',True):
        return xdata[grad.argmin()] - xA
    # if the user hasn't specified a way to define the 2nd period to search
    # raise error
    else:
        raise ValueError("2nd window not specified! Need to specify either xendA & xendB, depth_exp & depth_win or try_gmin.")
    # if the mask is empty
    if np.where(mask)[0].shape[0]==0:
        # if set to default to a value
        if default:
            # if user specified window period
            if ('xendB' in kwargs) and ('xendA' in kwargs):
                # relative to the end
                # take the upper limit
                if end_ref == 'end':
                    warnings.warn(f"All empty mask for 2nd window!\nxendA={xendA},xendB={xendB},end_ref={end_ref},xmax={xmax}. Defaulting to {(xmax - xendB) - xA}",category=EmptyMaskWarning)
                    end = timer()
                    print(f"Finish finding depth. Empty mask for 2nd window: {end-start}s")
                    return (xmax - xendB) - xA
                elif end_ref == 'start':
                    warnings.warn(f"All empty mask for 2nd window!\nxendA={xendA},xendB={xendB},end_ref={end_ref},xmax={xmax}. Defaulting to {(xmax + xendB) -xA}",category=EmptyMaskWarning)
                    end = timer()
                    print(f"Finish finding depth. Empty mask for 2nd window: {end-start}s")
                    return (xmax + xendB) -xA
            # if specified from expected depth estimate
            # take as upper end of the window period
            elif 'depth_exp' in kwargs:
                warnings.warn(f"All empty mask for 2nd period depth_exp={depth_exp}, win={depth_win}, xA={xA}!\nDefaulting to {xA+depth_exp + (depth_win/2)}",category=EmptyMaskWarning)
                #xB = xA+depth_exp+(win/2)
                end = timer()
                print(f"Finish finding depth. Empty mask for 2nd window: {end-start}s")
                return depth_exp+(depth_win/2) - xA        
        # if not set to default raise an exception
        # useful for debugging
        else:
            if ('xendB' in kwargs) and ('xendA' in kwargs): 
                raise EmptyMaskException(f"All empty mask for 2nd period!\nxendA={xendA},xendB={xendB},end_ref={end_ref},xmax={xmax}")
            elif 'depth_exp' in kwargs:
                raise EmptyMaskException(f"All empty mask for 2nd period depth_exp={depth_exp}, win={depth_win}!")
    else: 
        # mask gradient values
        grad_mask = grad[mask]
        # mask xdata
        xmask = xdata[mask]
        # invert to turn -ve gradients to +ve
        grad_mask *= -1.0
        # find max gradient in period
        hlim = grad_mask.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad_mask,height=hh*hlim)
        # if no peaks where found
        if len(pks)==0:
            # if defaulting to a set value
            if default:
                if ('xendB' in kwargs) and ('xendA' in kwargs):
                    warnings.warn(f"Number of peaks found of 2nd window is 0 for 2nd period {xendA} and {xendB}, NA={NA}, NB={NB} for reference {end_ref}! Defaulting to max location",category=MissingPeaksWarning)
                elif 'depth_exp' in kwargs:
                    warnings.warn(f"Number of peaks found of 2nd window is 0 for 2nd period {depth_exp} win={depth_win}, NA={NA}, NB={NB}! Defaulting to max location",category=MissingPeaksWarning)
                end = timer()
                print(f"Finish finding depth. No peaks for 2nd window: {end-start}s")
                # default to the max value
                return xmask[grad_mask.argmax()] - xA
            # raise exception clipping the part after the exclamation mark
            else:
                if ('xendB' in kwargs) and ('xendA' in kwargs):
                    raise MissingPeaksException(f"Number of peaks found of 2nd window is 0 for 2nd period {xendA} and {xendB} for reference {end_ref}!")
                elif 'depth_exp' in kwargs:
                    raise MissingPeaksException(f"Number of peaks found of 2nd window is 0 for 2nd period {depth_exp} win={depth_win}, xA={xA}!")
        # if peaks were found
        else:
            # retrieve the method for choosing the peaks
            if (pselect == 'argmax') or (pselect == 'max'):
                # if correction factor is given
                if 'cfact' in kwargs:
                    cfact = kwargs['cfact']
                    # if it's a float
                    if isinstance(cfact,float):
                        # check it's between 0 and 1
                        if (cfact<0) or (cfact>1.0):
                            raise ValueError(f"Relative height has to be between 0 and 1")
                        # extract peak width using target relative height
                        results_full = peak_widths(grad_mask,pks,rel_height=cfact)
                        # peak widths can contain NaNs
                        # if there are NaNs default to normal behaviours
                        if any(np.isnan(results_full[2])):
                            print("Skipping widths due to NaNs")
                            pkB = grad_mask[pks].argmax()
                            return xmask[pks][pkB] - xA
                        else:
                            # find the highest peak
                            pkB = grad_mask[pks].argmax()
                            # find the left interpolated point
                            return xmask[int(results_full[2][pkB])] - xA
                else:
                    pkB = grad_mask[pks].argmax()
                    # find correspondng x value
                    end = timer()
                    print(f"Finishing finding depth. Chose max. {end-start}s")
                    return xmask[pks][pkB] - xA
            # get last peak if specified or as part of limit
            elif (pselect == 'limit') or (pselect == 'last'):
                end = timer()
                print(f"Finishing finding depth. Chose last. {end-start}s")
                return xmask[pks][-1] - xA
            # use first peak
            elif pselect == 'first':
                end = timer()
                print(f"Finish finding depth. Chose first: {end-start}s")
                return xmask[pks][0] - xA
            # if unsupported default to argmax
            else:
                if default:
                    warnings.warn(f"Unsupported peak selection mode {pselect}. Defaulting to argmax for second period")
                    pkB = grad_mask[pks].argmax()
                    # find correspondng x value
                    return grad_mask[pks][pkB] - xA
                else:
                    raise ValueError(f"Unsupported peak selection mode {pselect}!")

def depthCorrectSC(path,N,win=1.0):
    if isinstance(path,str):
        data = dp.loadSetitecXls(path,'auto_data')
    # check there are multiple unique step codes
    sc = np.unique(data['Step (nb)'])
    # if there's only one then it can't be corrected this way
    if sc.shape[0]<2:
        raise warnings.warn("Data does not contain multiple step codes so cannot correct!")
        return 0.0
    sc_min = data[data['Step (nb)'] == sc[1]].index.min()
    dep = data['Position (mm)'][sc_min]
    # due to rolling window, peaks in the gradient occur after the event
    # only search for peaks that come after
    data_filt = data[(data['Position (mm)'] >= (dep-win)) & (data['Position (mm)'] <= (dep+win))]
    # get torque values for target range
    torque_filt = data_filt['I Torque (A)'].values.flatten() + data_filt['I Torque Empty (A)'].values.flatten()
    
    # filter + smooth
    torque_filt = wiener(torque_filt,N)
    grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
    return dep-data_filt['Position (mm)'][data_filt.index.min()+grad.argmax()]

def plotTorqueGradient(path,NN,abs_grad=True):
    f,ax = plt.subplots()
    tax = ax.twinx()
    for fn in glob('AirbusData/Seti-Tec data files life test/*.xls'):
        data = dp.loadSetitecXls(fn,'auto_data')
        tq = data['I Torque (A)'].values
        pos = np.abs(data['Position (mm)'].values)
        ax.plot(pos,tq,'b-')
        for N in NN:
            tq_filt = wiener(tq,N)
            grad = rolling_gradient(tq_filt,N) * tukey(len(tq_filt),0.1,True)
            if abs_grad:
                grad = np.abs(grad)
            tax.plot(pos,grad,'r-')
    lines = [ax.lines[0],tax.lines[0]]
    ax.legend(lines,['Torque','Gradient'])
    return f

# from https://stackoverflow.com/a/22349717
def make_legend_arrow(legend, orig_handle,xdescent, ydescent,width, height, fontsize):
    import matplotlib.patches as mpatches
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

def plotDepthCorrectSC(path,NN=[10,20,30,40,50],win=1.0,opath='',close_figs=True):
    '''
        Correct by distance between first step transition location and nearby gradient peak

        By calculating gradient using a rolling window, there's a delay between the peak in the gradient
        and the change in the data that causes it. The distance is related to the window size in a seemingly
        exponential fashion.

        The biggest changes in the data are when the program changes as there are large jumps in the torque.

        This function plots the data and rolling gradient of the target file and draws an arrow from the peak in the rolling gradient
        to the the location where the file transitions from step 0 to step 1 (hardcoded). This is the correction in the distance that is required.

        The input win controls where the program searches. This is to limit the amount of data and speed up the calculation

        Inputs:
            path : Path to target data file
            NN : Target sample windows to use
            win : Window size in mm to search

        Returns None
    '''
    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.patches as mpatches
    # from https://stackoverflow.com/a/22349717
    def make_legend_arrow(legend, orig_handle,xdescent, ydescent,width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
        return p
    if isinstance(path,str):
        data = dp.loadSetitecXls(path,'auto_data')
    # check there are multiple unique step codes
    sc = np.unique(data['Step (nb)'])
    # if there's only one then it can't be corrected this way
    if sc.shape[0]<2:
        warnings.warn("Data does not contain multiple step codes so cannot correct!")
        return 0.0
    sc_min = data[data['Step (nb)'] == sc[1]].index.min()
    fname = os.path.splitext(os.path.basename(path))[0]
    f,ax = plt.subplots()
    tax = ax.twinx()
    data['Position (mm)'] = np.abs(data['Position (mm)'])
    # get where the program changes
    dep = data['Position (mm)'][sc_min]
    # add vertical line for transition point
    ax.vlines(dep,0.0,10.0,colors=['k'])
    # due to rolling window, peaks in the gradient occur after the event
    # only search for peaks that come after
    data_filt = data[(data['Position (mm)'] >= (dep-win)) & (data['Position (mm)'] <= (dep+win))]
    # get torque values for target range
    torque_filt_o = data_filt['I Torque (A)'].values.flatten() + data_filt['I Torque Empty (A)'].values.flatten()
    all_dist = []
    # for each window size
    arr = None
    for N in NN:
        # filter + smooth
        torque_filt = wiener(torque_filt_o,N)
        grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
        dist = dep-data_filt['Position (mm)'][data_filt.index.min()+grad.argmax()]
        # plot the target window
        ax.plot(data_filt['Position (mm)'].values,torque_filt,'b-')
        # plot the gradient
        tax.plot(data_filt['Position (mm)'].values,grad,'r-')
        # draw an arrow going from the peak in the gradient that would be used to where the new location is
        arr = tax.arrow(x=data_filt['Position (mm)'][data_filt.index.min()+grad.argmax()],y=grad.max(),dx=dist,dy=0,color='green',head_width=0.05,shape='full',length_includes_head=False)
        all_dist.append(abs(dist))
    # get first lines from each
    lines = [ax.lines[0],tax.lines[0],arr]
    ax.legend(lines,['Torque','Gradient','Correction'],handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})
    ax.set(xlabel='Position (mm)',ylabel='Torque (A)',title=fname)
    f.savefig(os.path.join(opath,f"{fname}-depth-correct-sc.png"))
    if close_figs:
        plt.close(f)

    f,ax = plt.subplots()
    ax.plot(NN,all_dist,'x')
    ax.set(xlabel="Window Size",ylabel="Distance (mm)",title=f"{fname}\nWindow Distance")
    f.savefig(os.path.join(opath,f"{fname}-depth-correct-sc-distance.png"))
    if close_figs:
        plt.close(f)

def plotDepthCorrectedDist(path,NN=[10,20,30,40,50],win=1.0,opath='',close_figs=False,no_empty=False):
    '''
        Plot the distance between the gradient peak and distance peak at the different window sizes for
        each file.

        Creates a scatter plot where each color represents a different file, the x-axis is the sample window
        size used and y-axis is the correction distance for each window size.

        Constructs a dictionary containing the rolling window sizes and correction distances. The rolling window
        sizes are under the NN key. For each filename, there's a vector of distances representing the correction
        distance for each rolling window size.

        Inputs:
            path : Wildcard path to data files
            NN : Sample window size used in rolling gradient
            win : Search window used for correction
            opath : Folder to save plot. If None, no plots are saved
            close_figs : Flag to close the figure after saving.
            no_empty : Flag to not use the empty channel

        Returns the dictionary of distance values organised by filename,
    '''
    all_dists = {'NN' : NN}
    
    for fn in glob(path):
        data = dp.loadSetitecXls(fn,'auto_data')
        fname = os.path.splitext(os.path.basename(fn))[0]
        # check there are multiple unique step codes
        sc = np.unique(data['Step (nb)'])
        # if there's only one then it can't be corrected this way
        if sc.shape[0]<2:
            warnings.warn(f"{fn} data does not contain multiple step codes so cannot correct!")
            all_dists[fname] = len(NN)*[None,]
            continue
        all_dists[fname] = []
        sc_min = data[data['Step (nb)'] == sc[1]].index.min()
        data['Position (mm)'] = np.abs(data['Position (mm)'])
        # get where the program changes
        dep = data['Position (mm)'][sc_min]
        # due to rolling window, peaks in the gradient occur after the event
        # only search for peaks that come after
        data_filt = data[(data['Position (mm)'] >= (dep-win)) & (data['Position (mm)'] <= (dep+win))]
        # get torque values for target range
        if no_empty:
            torque_filt = data_filt['I Torque (A)'].values.flatten()
        else:
            torque_filt = data_filt['I Torque (A)'].values.flatten() + data_filt['I Torque Empty (A)'].values.flatten()
        # for each window size
        arr = None
        for N in NN:
            # filter + smooth
            torque_filt = wiener(torque_filt,N)
            grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
            dist = dep-data_filt['Position (mm)'][data_filt.index.min()+grad.argmax()]
            all_dists[fname].append(abs(dist))
    # makes subplots
    f,ax = plt.subplots()
    # plot the distances for each file
    for kk,vv in all_dists.items():
        if kk == 'NN':
            continue
        ax.plot(NN,vv,'x',markersize=10,markeredgewidth=5)
    ax.set(xlabel='Rolling Window Size (Samples)',ylabel="Distance (mm)",title=f"{os.path.dirname(path)} Window Distances")
    # force the xticks to window ticks
    ax.set_xticks(NN)
    if opath is not None:
        f.savefig(os.path.join(opath,f"{os.path.dirname(path)}-window-distances.png"))
    # close the figure if wanted
    if close_figs:
        plt.close(f)
    # return the constructed distance
    return all_dists
    
def depthCorrectPWin(path,N):
    '''
        Correct by width of tallest peak

        Searches within 1mm of where the program changes from step code 0 to 1.

        Peaks in the gradient are searched for using scipy's find_peaks function. The width
        of the tallest peak is then found and the correction distance is set as half the width
        of that peak
        
        Inputs:
            path : Input file path
            N : Rolling window size in samples

        Returns correction distance
    '''
    if isinstance(path,str):
        data = dp.loadSetitecXls(path,'auto_data')
    # check there are multiple unique step codes
    sc = np.unique(data['Step (nb)'])
    # if there's only one then it can't be corrected this way
    if sc.shape[0]<2:
        raise warnings.warn("Data does not contain multiple step codes so cannot correct!")
        return 0.0
    # get where the codes transition from step code 0 to 1
    sc_min = data[data['Step (nb)'] == sc[1]].index.min()
    dep = data['Position (mm)'][sc_min]
    # filter to within 1 mm of this location
    data_filt = data[(data['Position (mm)'] >= (dep-1.0)) & (data['Position (mm)'] <= (dep+1.0))]
    # get torque values for target range
    torque_filt = data_filt['I Torque (A)'].values.flatten() + data_filt['I Torque Empty (A)'].values.flatten()
    # filter + smooth
    torque_filt = wiener(torque_filt,N)
    # find gradient
    grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
    # find peaks in gradient
    pks,_ = find_peaks(grad)
    # find peak widths
    widths = peak_widths(grad,pks)[0]
    # find tallest peak
    pi = pks.argmax()
    return abs(data_filt['Position (mm)'].values[pi] - data_filt['Position (mm)'].values[int(pi-(widths[pi]//2))])
    
def depth_est_rolling_pks(ydata,xdata=None,method='weiner',NA=20,NB=None,xstart=10.0,xendA=7.0,xendB=4.0,hh=0.1,**kwargs):
    '''
        Detects peaks used in calculating depth estimate using rolling methog

        See doc for depth_est_rolling
    '''
    # if window size for rolling gradient is not set
    # use the same as the function
    if (NB is None) and (NA is not None):
        NB = NA
    elif (NB is not None) and (NA is None):
        NA = NB
    else:
        raise ValueError(f"Need to supply a proper window size for both smoothing filter and rolling gradient! NA={NA}, NB={NB}")
    # if the xdata is not set
    # create an artificial set based on length of data
    if xdata is None:
        xdata = np.arange(len(ydata))
    # filter data using target method
    if method == 'weiner':
        filt_weight = wiener(ydata,NA)
    elif method == 'savgol':
        filt_weight = weighted_savgol_filter(ydata,NA,1,deriv=0, window=kwargs.get("window","hann"))
    # get the sign of the flag 
    use_signs = kwargs.get('use_signs',True)
    # perform rolling gradient on smoothed data
    grad = rolling_gradient(filt_weight,NB,keep_signs=use_signs)
    # get weighting window
    win = kwargs.get('filt_grad',True)
    # if it's a flag
    # create tukey window
    if isinstance(win,bool) and win:
        win = tukey(len(grad),0.1,True)
        grad *= win
    elif isinstance(win,np.ndarray):
        grad *= win
    # filter mask
    mask = xdata <= xstart
    # mask gradient to first period
    grad_mask = grad[mask]
    # if using the signs
    # set negative gradient to zero
    # during the start of the signal the signal is ramping up as the tool sides change
    if use_signs:
        grad_mask *= -1.0
    # get max gradient value
    hlim = grad_mask.max()
    # find peaks ignoring those below the target threshold
    pks,_ = find_peaks(grad_mask, height=hh*hlim)
    # if no peaks were found
    # use location of max
    if len(pks) ==0:
        xA = xdata[mask][grad_mask.argmax()]
    else:
        # retrieve the method for choosing the peaks
        sl = kwargs.get('pselect','argmax')
        # if the user set it to None, argmax or max
        # then the maximum peak within the period is used
        if (sl is None) or (sl == 'argmax') or (sl == 'max'):
            pkA = grad_mask[pks].argmax()
            # find correspondng x value
            xA = xdata[mask][pks][pkA]
        # if using the first value
        elif (sl == 'limit') or (sl == 'first'):
            xA = xdata[mask][pks][0]
        # if using the last value
        elif sl == 'last':
            xA = xdata[mask][pks][-1]
        # if user gave something unsupported
        # default to argmax
        else:
            warnings.warn(f"Unsupported peak selection mode {sl}. Defaulting to argmax for first period")
            pkA = grad_mask[pks].argmax()
            # find correspondng x value
            xA = grad_mask[pks][pkA]
    # filter end period
    # calculate max x value
    xmax = xdata[-1]
    # create mask for target period
    mask = (xdata >= (xmax-xendA))&(xdata <= (xmax-xendB))
    grad_mask = grad[mask]
    # if worries about signs
    # invert peaks to focus on -ves
    if use_signs:
        grad_mask *= -1.0
    # find max gradient in period
    hlim = grad_mask.max()
    # find peaks ignoring those below the target threshold
    pks,_ = find_peaks(grad_mask,height=hh*hlim)
    if len(pks)==0:
        xA = xdata[mask][grad_mask.argmax()]
    else:
        # retrieve the method for choosing the peaks
        sl = kwargs.get('pselect','argmax')
        if (sl is None) or (sl == 'argmax') or (sl == 'max'):
            # find the maximum peak
            pkB = grad_mask[pks].argmax()
            # find the corresponding x value
            xB = xdata[mask][pks][pkB]
        # get last peak if specified or as part of limit
        elif (sl == 'limit') or (sl == 'last'):
            xB = xdata[mask][pks][-1]
        # use first peak
        elif sl == 'first':
            xB = xdata[mask][pks][0]
        # if unsupported default to argmax
        else:
            warnings.warn(f"Unsupported peak selection mode {sl}. Defaulting to argmax for second period")
            pkB = grad_mask[pks].argmax()
            # find the corresponding x value
            xB = xdata[mask][pks][pkB]
    # calculate depth as distance between the two
    return xA,xB

def time_depth_est_rolling(xdata,ydata,NN=[10,20,30,40,50],its=1000):
    '''
        Evaluate how long it takes to estimate the depth using the rolling gradient approach

        A series of window sizes for both the filter and rolling gradient are trialled. For each window size N,
        the function is evaluated and timed its number of times

        Creates two dictionaries for the Weiner and Savgol methods respectively. The dictionaries are organised
        by window sizes and each entry has the time the function took to evaluate the data.

        Inputs:
            xdata : Iterable collection of x-values
            ydata : Iterable collection of y-balues
            NN : List of window sizes used for testing
            its : Number of iterations per window size

        Returns Weiner evaluation times dictionary and Savgol evaluation times dictionary
    '''
    from timeit import default_timer as timer

    def tt(y,x,N,method):
        start = timer()
        depth_est_rolling(y,x,NA=N,method=method)
        end = timer()
        return end-start

    weiner_timer = {}
    for N in NN:
        weiner_timer[N] = [tt(ydata,xdata,N,'weiner') for _ in range(its)]

    savgol_timer = {}
    for N in NN:
        savgol_timer[N] = [tt(ydata,xdata,N,'savgol') for _ in range(its)]

    return weiner_timer,savgol_timer

def replot_json_dest_stats(path):
    '''
        Replot the depth estimation JSON file

        Recreates the plots and saves them

        Inputs:
            path : Filepath to target JSON created by first_last_peaks_multiN
            
    '''
    # process the file path to just the filename
    fn_proc = os.path.splitext(os.path.basename(path))[0]
    # split into parts and extract the context
    pts = fn_proc.split('-')
    tool = pts[0]
    var = pts[1]
    stat = pts[-1]
    # load file
    dest = json.load(open(fn,'r'))
    # get the window sizes
    NA = dest['NA']
    NB = dest['NB']
    # iterate over known method names
    for name in ['Weighted Savgol','Weiner']:
        # extract the data
        dest_arr = dest[name]
        dest_arr = np.array(dest_arr)
        # replot the data using target methods
        f,ax = plot_depthest(dest_arr,xticks=NA,yticks=NB,ylabel="Filter Window Size",xlabel="Rolling Gradient Window Size",cmap='Blues',title=f"{tool} {var.capitalize()} {stat.capitalize()} of Depth Estimate from Rolling Gradient of {name.upper()}",col='k')
        # re-save the figure using the same format name as the original
        f.savefig(f"{tool}-{var}-{'-'.join(name.lower().split(' '))}-depth-estimate-filter-rolling-gradient-{stat}.png")
        plt.close(f)

def replot_evaltime(path,**kwargs):
    '''
        Replot the evaluation time dictionary

        Replots the data as a scatter plot using the given kwards

        Inputs:
            path : Filepath to JSON or dict of eval times produced by depth_est_evaltime
            xlabel : X-axis label. Default Iteration.
            ylabel : y-axis label. Default Evaluation Time (s).
            title : Axis title. Default Rolling Depth Est Evaluation Time its={len(tt)} where len(tt) is the number of times i.e. the number of iterations

        Returns figure and axis object
    '''
    if isinstance(path,str):
        data= json.load(open(path,'r'))
    else:
        data = path

    f,ax = plt.subplots(constrained_layout=True)
    for N,tt in data.items():
        ax.plot(tt,'x',label=f"N={N}")
    ax.legend()
    ax.set(xlabel=kwargs.get("xlabel","Iteration"),ylabel=kwargs.get("ylabel","Evaluation Time (s)"),title=kwargs.get("title",f"Rolling Depth Est Evaluation Time its={len(tt)}"))
    return f,ax

def replot_mean_evaltime(path,**kwargs):
    '''
        Replot the given eval time dictionary

        The dict can be a path to the JSON file or the dict itself

        Replots the data using the given kwargs

        Inputs:
            path : Filepath to JSON or dict of eval times produced by depth_est_evaltime
            xlabel : X-axis label. Default Window Size
            avg_ylabel : Y-axis label for average plot. Default Average
            var_ylabel : Y-axis label for variance plot. Default variance
            avgtitle : Axis title for average plot. Default Average
            vartitle : Axis title for variance plot. Default Variance

        Returns figure and axes objects
    '''
            
    if isinstance(path,str):
        data= json.load(open(path,'r'))
    else:
        data = path

    fv,axv = plt.subplots(ncols=2,constrained_layout=True)
    axv[0].plot(data.keys(),[np.mean(tt) for tt in data.values()],'rx')
    axv[0].set(xlabel=kwargs.get("xlabel","Window Size"),ylabel=kwargs.get("avg_ylabel","Average Evaluation Time (s)"),title=kwargs.get("avgtitle","Average"))
    
    axv[1].plot(data.keys(),[np.var(tt) for tt in data.values()],'rx')
    axv[1].set(xlabel=kwargs.get("xlabel","Window Size"),ylabel=kwargs.get("var_ylabel","Variance Evaluation Time (s)"),title=kwargs.get("vartitle","Variance"))

    fv.suptitle(kwargs.get("title","Mean and Variance of Evaluation Time"))

    return fv,axv

def plot_win(win,N=51):
    from scipy.fft import fft, fftshift
    window = get_window(win,N)
    f,ax = plt.subplots(nrows=2,constrained_layout=True)
    ax[0].plot(window)
    ax[0].set(xlabel="Index",ylabel="Amplitude",title=f"{win} Window")

    A = fft(window,2048) / (len(window)/2.0)
    freq = np.linspace(-0.5,0.5,len(A))
    response = np.abs(fftshift(A / abs(A).max()))
    response = 20 * np.log10(np.maximum(response, 1e-10))
    ax[1].plot(freq,response)
    ax[1].axis([-0.5, 0.5, -120, 0])
    ax[1].set_title(f"Frequency response of the {win} window")
    ax[1].set_ylabel("Normalized magnitude [dB]")
    ax[1].set_xlabel("Normalized frequency [cycles per sample]")

    return f,ax

# for a series of windows and their freq response
def plot_windows():
    # what windows to plot
    for win in ['boxcar','triang','blackman','hamming','hann','bartlett','flattop','parzen','bohman','blackmanharris','nuttall','barthann','cosine','exponential','tukey','taylor']:
        f,ax = plot_win(win)
        f.savefig(f"{win}-plot.png")
        plt.close(f)

# plot the tukey window at different alpha values
# alpha controls the width of the window
def plot_tukey(alpha,N=51):
    f,ax = plt.subplots()
    for a in alpha:
        win = tukey(N,a)
        ax.plot(win,label=f"a={a}")
    ax.legend()
    return f,ax

############################# RUNS #############################
def rolling_gradient_gui(path):
    from matplotlib.widgets import Slider
    # load the data file
    data = dp.loadSetitecXls(path)[-1]
    xdata = data["Position (mm)"]
    f,ax = plt.subplots(nrows=2,ncols=2)
    ax[0,0].plot(xdata,data["I Thrust (A)"].values)
    ax[0,1].plot(xdata,data["I Torque (A))"].values + data["I Torque Empty (A))"].values)
    # create twin axes for plotting gradient
    cax = [ax[0,0].twinx(),ax[0,1].twinx()]
    # create gradient lines to update
    lines = []
    # calculate slope for torque
    slope = rolling_gradient(data["I Torque (A))"].values + data["I Torque Empty (A))"].values,10)
    # find the peaks with default values
    pks,_(slope,height=0.15,distance=30)
    # plot the gradient slope and peaks saving plotting options
    lines.append([cax[0].plot(xdata,slope)[0],cax.scatter(xdata[pks],slope[pks],'rx')])
    
    # calculate slope for torque
    slope = rolling_gradient(data["I Thrust (A))"].values,10)
    # find the peaks with default values
    pks,_ = find_peaks(slope,height=0.15,distance=30)
    lines.append([cax[1].plot(xdata,slope)[0],cax[1].scatter(xdata[pks],slope[pks],'rx')])

    # create sliders
    axN = plt.axes([0.25,0.1,0.65,0.03])
    win_slider = Slider(ax=axN,
                        label="Win Size",
                        valmin=10,
                        valmax=200,
                        valinit=10,
                        valstep=5,
                        valfmt="%d")

    ax_amp = plt.axes([0.25,0.15,0.65,0.03])
    amp_slider = Slider(ax=ax_amp,
                        label="Min Amp",
                        valmin=0.0,
                        valmax=1.0,
                        valinit=0.15,
                        valstep=0.01,
                        valfmt="%.2f")
    
    dist_amp = plt.axes([0.25,0.2,0.65,0.03])
    dist_slider = Slider(ax=dist_amp,
                        label="Peak Dist",
                        valmin=0,
                        valmax=len(xdata),
                        valinit=30,
                        valstep=1,
                        valfmt="%d")
    # function for updating lines
    def update(val):
        # get window size
        N = int(win_slider.val)
        # get amp
        amp = amp_slider.val
        # get distance
        dist = dist_slider.val
        # find slope for Torque
        slope = rolling_gradient(ydata["I Torque (A))"].values + ydata["I Torque Empty (A))"].values,N)
        # find peaks
        pks,_ = find_peaks(slope,height=amp,distance=dist)
        # update line
        lines[0].set_ydata(slope)

        # find slope for Torque
        slope = rolling_gradient(ydata["I Thrust (A))"].values,N)
        # find peaks
        pks,_ = find_peaks(slope,height=amp,distance=dist)
        # update line
        lines[1].set_ydata(slope)
        return lines

    win_slider.set_update(update)
    amp_slider.set_update(update)
    dist_slider.set_update(update)
    plt.show()

def full_run(N):
    # iterate over tools
    for tool in ["UC","UD"]:
        print("tool ",tool)
        # number of files
        print("counting files...")
        nf = len(glob(f"xls/{tool}*.xls"))
        # iterate over sorted files
        print("processing...")
        for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
            # get coupon and hole for plotting
            c,h = dp.get_hole_coupon(fn,tool)
            # load file
            data = dp.loadSetitecXls(fn)[-1]
            # iterate over variables
            for var in ["Thrust","Torque"]:
                ydata = data[f"I {var} (A)"]
                if var == "Torque":
                    ydata += data["I Torque Empty (A)"]
                ## iterate over each window size
                for nn in N:
                    # create plotting axes
                    f,ax = plt.subplots(ncols=2,constrained_layout=True)
                    # plot original data
                    ax[0].plot(ydata,'b-')
                    # create twin axes
                    cax = ax[0].twinx()
                    # calculate rolling slope of the signal
                    slope = rolling_gradient(ydata,nn)
                    # plot data
                    cax.plot(slope,'r-',label="Slope")
                    # find peaks in the slope + plot
                    pks,_ = find_peaks(slope)
                    cax.plot(pks,slope[pks],'mx',label="All peaks",markersize=10,linewidth=6)
                    pks,_ = find_peaks(slope,height=0.15,distance=30)
                    cax.plot(pks,slope[pks],'kx',label="Marked Peaks",markersize=10,linewidth=6)
                    # integrate back up
                    integ = rolling_gradient_integ(ydata,nn)
                    ax[1].plot(integ,label=f"N={N}")

                    # set the labels
                    ax[1].set(xlabel="Index",ylabel=f"{var} (A)",title=f"Integrated Signal")
                    ax[0].set(xlabel="Index",ylabel=f"{var} (A)",title="Data + Rolling Gradient")
                    cax.set_ylabel("Normed Rolling Gradient")
                    plt.legend()
                    f.suptitle(f"{tool} {var} Coupon {c} Hole {h} N={nn}")
                    f.savefig(f"{tool}-coupon-{c}-hole-{h}-{var}-rolling-gradient-data-and-integrated-N-{nn}.png")
                    plt.close(f)

def depth_est(N,targets=[0,-1],dnominal=21.0):
    # iterate over tools
    for tool in ["UC","UD"]:
        print("tool ",tool)
        # number of files
        print("counting files...")
        nf = len(glob(f"xls/{tool}*.xls"))
        torque_est = {}
        thrust_est = {}
        # iterate over sorted files
        print("processing...")
        for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
            # get coupon and hole for plotting
            c,h = dp.get_hole_coupon(fn,tool)
            # load file
            data = dp.loadSetitecXls(fn)[-1]
            xdata = np.abs(data["Position (mm)"].values.flatten())
            # iterate over variables
            for var,dest in zip(["Thrust","Torque"],[thrust_est,torque_est]):
                ydata = data[f"I {var} (A)"]
                if var == "Torque":
                    ydata += data["I Torque Empty (A)"]
                ## iterate over each window size
                for nn in N:
                    if nn not in dest:
                        dest[nn] = []
                    # calculate rolling slope of the signal
                    slope = rolling_gradient(ydata,nn)
                    # find the peaks
                    pks,_ = find_peaks(slope,height=0.15,distance=30)
                    # find the depth estimate 
                    dest[nn].append(abs(xdata[pks[targets[0]]]-xdata[pks[targets[1]]]))
        # iterate over depth estimates saved
        for nn,dest in torque_est.items():
            f,ax = plt.subplots(constrained_layout=True)
            ax.plot(dest)
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{tool} Torque Depth Est from Gradient Peaks N={nn}")
            f.savefig(f"{tool}-coupon-{c}-hole-{h}-Torque-rolling-gradient-depth-est-N-{nn}.png")
            plt.close(f)

        for nn,dest in thrust_est.items():
            f,ax = plt.subplots(constrained_layout=True)
            ax.plot(dest)
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{tool} Thrust Depth Est from Gradient Peaks N={nn}, minH=0.15, dist=30")
            f.savefig(f"{tool}-coupon-{c}-hole-{h}-Thrust-rolling-gradient-depth-est-N-{nn}.png")
            plt.close(f)

def spline_smooth_run():
    for tool in ["UC","UD"]:
        print("tool ",tool)
        # iterate over sorted files
        print("processing...")
        for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
            # get coupon and hole for plotting
            c,h = dp.get_hole_coupon(fn,tool)
            # load file
            data = dp.loadSetitecXls(fn)[-1]
            #xdata = np.abs(data["Position (mm)"].values.flatten())
            # iterate over variables
            for var in ["Thrust","Torque"]:
                ydata = data[f"I {var} (A)"]
                # set parameters for filtering
                if var == "Torque":
                    ydata += data["I Torque Empty (A)"]
                    s=10.0
                    ht = 0.1
                else:
                    s = 0.02
                    ht=0.005
                # convert xdata to index rather than position
                # for some reason
                xdata = np.arange(len(ydata))
                # create spline representation of the data
                tck = interp.splrep(xdata,ydata,s=s,k=3)
                # new x data
                # over same range as x just different resolution
                x_new = np.linspace(min(xdata),max(xdata),200)
                # evaluate over new x data using spline
                yhat = interp.BSpline(*tck)(x_new)
                # if the spline fails to evaluate then it returns all NaNs
                # cause is related to oversmoothing
                if np.isnan(yhat).all():
                    print(f"B-spline interp is all NaNs")
                # plot the gradient of spline data
                grad = np.abs(np.gradient(yhat,x_new))
                # find peaks in the gradient + plot
                pks,_ = find_peaks(grad,height=0.1*grad.max(),distance=5)
                # create 2 axes
                f,[ax,cax] = plt.subplots(nrows=2,sharex=True,constrained_layout=True)
                # plot original data
                ax.plot(xdata,ydata,'b-',label="Data")
                # plot spline smoothed data
                ax.plot(x_new,yhat,'r-',label="Spline")
                # find the accompanying y values for where the peaks occur
                # draw a line between them
                # represents pwlf function
                ax.plot(xdata[[np.abs(xdata-x_new[pp]).argmin() for pp in pks]],ydata[[np.abs(xdata-x_new[pp]).argmin() for pp in pks]],'mx--',linewidth=3,label="Peaks Line")
                ax.legend()
                # plot the gradient of spline on 2nd axes
                cax.plot(x_new,grad,'k-',label="Spline Grad")
                # mark location of peaks on spline
                cax.plot(x_new[pks],grad[pks],'gx',markersize=10,label="Grad Peaks")
                # set labels
                ax.set(xlabel="Index",ylabel=f"{var} (A)")
                f.suptitle(f"{tool} {var} Coupon {c} Hole {h} B-Spline Smooth + Gradient k=3")
                cax.set(ylabel="Gradient of Spline")
                cax.legend()
                # save and close
                f.savefig(f"{tool}-coupon-{c}-hole-{h}-{var}-bspline-k-3-gradient.png")
                plt.close(f)

# perform rolling gradient on the data from each file and integrate back up to form peaks
def grad_integ_smooth_peaks(N=5,ds=False):
    for tool in ["UC","UD"]:
        print("tool ",tool)
        # iterate over sorted files
        print("processing...")
        for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
            # get coupon and hole for plotting
            c,h = dp.get_hole_coupon(fn,tool)
            # load file
            data = dp.loadSetitecXls(fn)[-1]
            xdata = np.abs(data["Position (mm)"].values.flatten())
            # downsample
            if ds:
                xdata = xdata[::2]
            # iterate over variables
            for var in ["Thrust","Torque"]:
                ydata = data[f"I {var} (A)"].values.flatten()
                if var == "Torque":
                    ydata += data["I Torque Empty (A)"].values.flatten()
                # downsample data
                if ds:
                    ydata = ydata[::2]
                # smooth via rolling gradient
                smooth = rolling_gradient_integ(pd.Series(ydata),N)
                # calculate numerical gradient of smooth data
                #grad = np.abs(np.gradient(smooth,xdata[:-1]))
                grad = np.gradient(smooth,xdata[:-1])
                grad[grad<0] = 0.0
                # remove invalid 
                np.nan_to_num(grad,copy=False,nan=0.0,posinf=0.0)
                # find peaks in gradient
                pks,_ = find_peaks(grad,height=0.1*grad.max(),distance=10)
                ## plot
                f,[ax,cax] = plt.subplots(nrows=2,sharex=True,constrained_layout=True)
                # plot original data
                ax.plot(xdata,ydata,'r-',label="Data")
                # plot smooth data
                # as it's from gradient it's N-1 points as opposed to N
                ax.plot(xdata[:-1],smooth,'b-',label="Smooth")
                # connect the dots
                ax.plot(xdata[:-1][pks],ydata[pks],'o--')
                ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)",title="Original + Smoothed")
                ax.legend()
                # plot the gradient
                cax.plot(xdata[:-1],grad,'k-',label="G-Smooth")
                #cax.set_ylim(bottom=grad.min(),top=grad.max())
                # plot the peaks
                cax.plot(xdata[:-1][pks],grad[pks],'mx',markersize=10,linewidth=6,label="Grad Pks")
                # set the labels
                cax.set(xlabel="Position (mm)",ylabel="Gradient",title="Rolling Gradient")
                cax.legend()
                f.suptitle(f"{tool} {var} coupon {c} hole {h} Smoothed Integrated Rolling Gradient N={N}")
                f.savefig(f"{tool}-coupon-{c}-hole-{h}-{var}-smooth-integ-rolling-gradient-peaks-N-{N}.png")
                plt.close(f)

def full_run_savgol(N=30):
    from scipy.signal import savgol_filter
    for tool in ["UC","UD"]:
        print("tool ",tool)
        # iterate over sorted files
        print("processing...")
        for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
            # get coupon and hole for plotting
            c,h = dp.get_hole_coupon(fn,tool)
            # load file
            data = dp.loadSetitecXls(fn)[-1]
            xdata = np.abs(data["Position (mm)"].values.flatten())
            # iterate over variables
            for var in ["Thrust","Torque"]:
                ydata = data[f"I {var} (A)"].values.flatten()
                if var == "Torque":
                    ydata += data["I Torque Empty (A)"].values.flatten()
                # filter the data using savgol_filter
                for o in [1,2,3,4,5]:
                    # filter using savgol_filter
                    filt = savgol_filter(ydata,N,o,deriv=0)
                    # filter using weighted_savgol_filter
                    filt_weight = weighted_savgol_filter(ydata,N,o,deriv=0, window="hann")
                    # create axes
                    f,ax = plt.subplots(ncols=2,sharex=True,constrained_layout=True,figsize=(8,5))
                    # plot original data
                    ax[0].plot(xdata,ydata,'b-',label="Original")
                    # plot savgol_filter data
                    ax[0].plot(xdata,filt,'r-',label="Filtered")
                    cax = ax[0].twinx()
                    grad = np.gradient(filt,xdata)
                    cax.plot(xdata,grad,'k-')
                    cax.set_ylabel("Gradient")
                    ax[0].legend()
                    # plot weighted_savgol_filter data
                    # plot original data
                    ax[1].plot(xdata,ydata,'b-',label="Original")
                    ax[1].plot(xdata,filt_weight,'r-',label="Weighted")
                    cax = ax[1].twinx()
                    grad = np.gradient(filt_weight,xdata)
                    cax.plot(xdata,grad,'k-')
                    cax.set_ylabel("Gradient")
                    ax[1].legend()
                    # set labels
                    ax[0].set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    ax[1].set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    # set title
                    f.suptitle(f"{tool} {var} Coupon {c} Hole {h} Savitzky-Golay Filter Order {o} N={N}")
                    
                    f.savefig(f"{tool}-coupon-{c}-hole-{h}-{var}-savgol-order-{o}-N-{N}.png")
                    plt.close(f)

def full_run_weiner(N):
    from scipy.signal import wiener
    for tool in ["UC","UD"]:
        print("tool ",tool)
        # iterate over sorted files
        print("processing...")
        for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
            # get coupon and hole for plotting
            c,h = dp.get_hole_coupon(fn,tool)
            # load file
            data = dp.loadSetitecXls(fn)[-1]
            xdata = np.abs(data["Position (mm)"].values.flatten())
            # iterate over variables
            for var in ["Thrust","Torque"]:
                ydata = data[f"I {var} (A)"].values.flatten()
                if var == "Torque":
                    ydata += data["I Torque Empty (A)"].values.flatten()
                # filter using savgol_filter
                filt = wiener(ydata,N)
                # create axes
                f,ax = plt.subplots(constrained_layout=True,figsize=(8,5))
                # plot original data
                ax.plot(xdata,ydata,'b-',label="Original")
                # plot wiener data
                ax.plot(xdata,filt,'r-',label="Filtered")
                ax.legend()
                cax = ax.twinx()
                # plot the gradient of the filtered data
                grad = np.gradient(filt,xdata)
                cax.plot(xdata,grad,'k-')
                cax.set_ylabel("Gradient")
                # set labels
                ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)",title=f"{tool} {var} Coupon {c} Hole {h} Wiener Filter N={N}")
                # save figure
                f.savefig(f"{tool}-coupon-{c}-hole-{h}-{var}-wiener-N-{N}.png")
                plt.close(f)

def depth_est_methods_air(km=0.1,dist=20):
    for tool in ["UC","UD"]:
        # wrapped methods
        for method,name in [(lambda x : weighted_savgol_filter(x,N,1,deriv=0, window="hann"),"Weighted Savgol"),
                            (lambda x : wiener(ydata,N),"Weiner")]:
            for filt in ["peak-prominence","height-distance"]:
                # for different window sizes
                for N in [10,20,30]:
                    # depth estimates for variables
                    dest_thrust = []
                    dest_torque = []
                    # iterate over files
                    for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
                        # find coupon and hole
                        c,h = dp.get_hole_coupon(fn,tool)
                        # load file
                        data = dp.loadSetitecXls(fn)[-1]
                        # get x data
                        xdata = np.abs(data['Position (mm)'].values.flatten())
                        # for each variables
                        for var,dest in zip(["Thrust","Torque"],[dest_thrust,dest_torque]):
                            if var == "Thrust":
                                ydata = data['I Thrust (A)'].values.flatten()
                            else:
                                ydata = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
                            # filter using weighted_savgol_filter
                            filt_weight = method(ydata)
                            grad = np.gradient(filt_weight[xdata<=10.0],xdata[xdata<=10.0])
                            win = get_window('hann',len(grad))
                            grad *= win
                            grad = np.abs(grad)
                            np.nan_to_num(grad,False,posinf=0.0)
                            pks,_ = find_peaks(grad)
                            prom,_,_  = peak_prominences(grad,pks)
                            ii = prom.argmax()
                            pkA = pks[ii]

                            xmax = xdata.max()-10.0
                            grad = np.gradient(filt_weight[xdata>=xmax],xdata[xdata>=xmax])
                            win = get_window('hann',len(grad))
                            grad *= win
                            grad = np.abs(grad)
                            np.nan_to_num(grad,False,posinf=0.0)
                            pks,_ = find_peaks(grad)
                            prom,_,_  = peak_prominences(grad,pks)
                            ii = prom.argmax()
                            pkB = pks[ii]
                            dest.append(xdata[xdata>=xmax][pkB] - xdata[xdata<=10.0][pkA])
                # create axes
                f,ax = plt.subplots()
                # plot depth estimate under current settings for thrust
                ax.plot(dest_thrust,'bx',label="Thrust")
                # plot the depth estimate for torque
                ax.plot(dest_torque,'rx',label="Torque")
                # plot nominal depth lines
                ax.plot(len(dest_thrust)*[21.0,],'k-',label="Nominal")
                # set labels
                ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)")
                ax.legend()
                f.suptitle(f"{tool} Depth Estimate From First & Last Periods\nWeighted Gradient of {name} Filter N={N}")
                f.savefig(f"{tool}-{'-'.join(name.lower().split(' '))}-depth-estimate-N-{N}-air-est.png")
                plt.close(f)

def depth_est_methods(km=0.1,dist=20):
    for tool in ["UC","UD"]:
        # wrapped methods
        for method,name in [(lambda x : weighted_savgol_filter(x,N,1,deriv=0, window="hann"),"Weighted Savgol"),
                            (lambda x : wiener(ydata,N),"Weiner")]:
            for filt in ["peak-prominence","height-distance"]:
                # for different window sizes
                for N in [10,20,30]:
                    # depth estimates for variables
                    dest_thrust = []
                    dest_torque = []
                    # iterate over files
                    for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
                        # find coupon and hole
                        c,h = dp.get_hole_coupon(fn,tool)
                        # load file
                        data = dp.loadSetitecXls(fn)[-1]
                        # get x data
                        xdata = np.abs(data['Position (mm)'].values.flatten())
                        # for each variables
                        for var,dest in zip(["Thrust","Torque"],[dest_thrust,dest_torque]):
                            if var == "Thrust":
                                ydata = data['I Thrust (A)'].values.flatten()
                            else:
                                ydata = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
                            # filter using weighted_savgol_filter
                            filt_weight = method(ydata)
                            # calculate gradient
                            grad = np.gradient(filt_weight,xdata)
                            # get window to remove edge artifacts
                            win = get_window('hann',len(grad))
                            grad *= win
                            # absolute gradient
                            grad = np.abs(grad)
                            # remove invalid values
                            np.nan_to_num(grad,False,posinf=0.0)
                            # if filtering by height and distance
                            if filt == "height-distance":
                                pks,_ = find_peaks(grad,height=km*grad.max(),distance=dist)
                            else:
                                # find all peaks
                                pks,_ = find_peaks(grad)
                                # calculate prominence
                                prom,_,_ = peak_prominences(grad,pks)
                                # filter to those above fraction of max
                                ii = np.where(prom>=(km*prom.max()))[0]
                                pks = pks[ii]
                            # estimate depth using last and first    
                            dest.append(xdata[pks[-1]]-xdata[pks[0]])
                    # create axes
                    f,ax = plt.subplots()
                    # plot depth estimate under current settings for thrust
                    ax.plot(dest_thrust,'bx',label="Thrust")
                    # plot the depth estimate for torque
                    ax.plot(dest_torque,'rx',label="Torque")
                    # plot nominal depth lines
                    ax.plot(len(dest_thrust)*[21.0,],'k-',label="Nominal")
                    # set labels
                    ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)")
                    ax.legend()
                    f.suptitle(f"{tool} Depth Estimate From {filt.upper()}\nWeighted Gradient of {name} Filter N={N}")
                    f.savefig(f"{tool}-{'-'.join(name.lower().split(' '))}-depth-estimate-N-{N}-{filt}.png")
                    plt.close(f)

def first_last_peaks_singleN(NN = [10,20,30,40,50], xstart=10.0,xendA=7.0,xendB=4.0):
    for tool in ["UC","UD"]:
        # wrapped methods
        for method,name in [(lambda x : weighted_savgol_filter(x,N,1,deriv=0, window="hann"),"Weighted Savgol"),
                            (lambda x : wiener(ydata,N),"Weiner")]:
            dest_var_thrust = []
            dest_var_torque  = []
            
            dest_mean_thrust = []
            dest_mean_torque  = []
            # for different window sizes
            for N in NN:
                # depth estimates for variables
                dest_thrust = []
                dest_torque = []
                # iterate over files
                for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
                    c,h = dp.get_hole_coupon(fn,tool)
                    # load file
                    data = dp.loadSetitecXls(fn)[-1]
                    xdata = np.abs(data['Position (mm)'].values.flatten())
                    for var,dest in zip(["Thrust","Torque"],[dest_thrust,dest_torque]):
                        if var == "Thrust":
                            ydata = data['I Thrust (A)'].values.flatten()
                        else:
                            ydata = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
                        # filter using weighted_savgol_filter
                        filt_weight = method(ydata)
                        grad = rolling_gradient(filt_weight,N)
                        f,ax = plt.subplots()
                        ax.plot(xdata,filt_weight)
                        cax = ax.twinx()                      
                        cax.plot(xdata,grad,'r-')
                        pks,_ = find_peaks(grad,height=0.1*grad.max())
                        cax.plot(xdata[pks],grad[pks],'kx',markersize=10,markeredgewidth=4)
                        #plt.show()
                        #cax.set_yscale('log')
                        ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                        cax.set_ylabel("Rolling Gradient")
                        f.suptitle(f"{tool} Coupon {c} Hole {h} Rolling Gradient of {var} using {name} Filter N={N}")
                        f.savefig(f"{tool}-{var}-coupon-{c}-hole-{h}-{'-'.join(name.lower().split(' '))}-rolling-gradient-N-{N}.png")
                        plt.close(f)

                        ## find peaks within first and last 10mm
                        pks,_ = find_peaks(grad[xdata <= xstart])
                        pkA = grad[xdata <= 10.0][pks].argmax()
                        xA = xdata[xdata <= 10.0][pks][pkA]

                        xmax = xdata.max()-7.0
                        pks,_ = find_peaks(grad[(xdata >= (xdata.max()-xendA))&(xdata <= (xdata.max()-xendB))])
                        pkB = grad[(xdata >= (xdata.max()-xendA))&(xdata <= (xdata.max()-xendB))][pks].argmax()
                        xB = xdata[(xdata >= (xdata.max()-xendA))&(xdata <= (xdata.max()-xendB))][pks][pkB]

                        #print(xA,xB)

                        dest.append(xB-xA)
                f,ax = plt.subplots()
                ax.plot(dest_thrust,'bx',label="Thrust")
                ax.plot(dest_torque,'rx',label="Torque")
                # plot nominal depth lines
                ax.plot(len(dest_thrust)*[21.0,],'k-',label="Nominal")
                # set labels
                ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)")
                ax.legend()
                f.suptitle(f"{tool} Depth Estimate From Rolling Gradient of {name.upper()} Filter N={N}")
                f.savefig(f"{tool}-{'-'.join(name.lower().split(' '))}-depth-estimate-rolling-gradient-N-{N}.png")
                plt.close(f)

                dest_var_thrust.append(np.var(dest_thrust))
                dest_var_torque.append(np.var(dest_torque))

                dest_mean_thrust.append(np.mean(dest_thrust))
                dest_mean_torque.append(np.mean(dest_torque))
            # plot variance in the statistics
            f,ax = plt.subplots(ncols=2,constrained_layout=True)
            ax[0].plot(NN,dest_var_thrust,'bx',label="Thrust")
            ax[0].plot(NN,dest_var_torque,'rx',label="Torque")
            ax[0].set(xlabel="Window Size",ylabel="Depth Est. Variance",title="Variance")
            ax[0].legend()
            
            ax[1].plot(NN,dest_mean_thrust,'bx',label="Thrust")
            ax[1].plot(NN,dest_mean_torque,'rx',label="Torque")
            ax[1].set(xlabel="Window Size",ylabel="Depth Est. Mean",title="Mean")
            ax[1].legend()
            f.suptitle(f"{tool} Statistics of Depth Estimate\nFrom Rolling Gradient of {name.upper()}")
            f.savefig(f"{tool}-{'-'.join(name.lower().split(' '))}-depth-estimate-statistics.png")
            plt.close(f)
            
def first_last_peaks_multiN(NN=[10,20,30,40,50],xstart=10.0,xendA=7.0,xendB=4.0):
    # iterate over each tool
    for tool in ["UC","UD"]:
        # define stat dicts for tool
        dest_thrust_var = {}
        dest_torque_var = {}
        dest_thrust_mean = {}
        dest_torque_mean = {}
        # iterate over tool window sizes
        for NA in NN:
            # iterate over methods
            for method,name in [(lambda x : weighted_savgol_filter(x,NA,1,deriv=0, window="hann"),"Weighted Savgol"),(lambda x : wiener(ydata,NA),"Weiner")]:
                # if the grids haven't been initialized
                if not name in dest_thrust_var:
                    dest_thrust_var[name] = []
                    dest_torque_var[name] = []

                    dest_thrust_mean[name] = []
                    dest_torque_mean[name] = []
                # add entry for current window size
                dest_thrust_var[name].append([])
                dest_torque_var[name].append([])

                dest_thrust_mean[name].append([])
                dest_torque_mean[name].append([])
                
                # for different window sizes
                for NB in NN:
                    # depth estimates for variables
                    dest_thrust = []
                    dest_torque = []
                    # iterate over files
                    for ii,fn in enumerate(sorted(glob(f"xls/{tool}*.xls"),key=lambda x : dp.get_hole_coupon(x,tool)),start=1):
                        c,h = dp.get_hole_coupon(fn,tool)
                        # load file
                        data = dp.loadSetitecXls(fn)[-1]
                        xdata = np.abs(data['Position (mm)'].values.flatten())
                        for var,dest in zip(["Thrust","Torque"],[dest_thrust,dest_torque]):
                            if var == "Thrust":
                                ydata = data['I Thrust (A)'].values.flatten()
                            else:
                                ydata = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
                            # filter using weighted_savgol_filter
                            filt_weight = method(ydata)
                            grad = rolling_gradient(filt_weight,NB)

                            # plot the data and gradient for the current sizes
                            f,ax = plt.subplots()
                            ax.plot(xdata,filt_weight)
                            cax = ax.twinx()                      
                            cax.plot(xdata,grad,'r-')
                            pks,_ = find_peaks(grad,height=0.1*grad.max())
                            cax.plot(xdata[pks],grad[pks],'kx',markersize=10,markeredgewidth=4)
                            #plt.show()
                            #cax.set_yscale('log')
                            ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                            cax.set_ylabel("Rolling Gradient")
                            f.suptitle(f"{tool} Coupon {c} Hole {h} Rolling Gradient of {var} using {name} Filter N=({NA},{NB})")
                            f.savefig(f"{tool}-{var}-coupon-{c}-hole-{h}-{'-'.join(name.lower().split(' '))}-rolling-gradient-Nmeth-{NA}-Nwin-{NB}.png")
                            plt.close(f)

                            ## find peaks within first and last 10mm
                            # filter for max
                            hlim = grad[xdata <= 10.0].max()
                            pks,_ = find_peaks(grad[xdata <= 10.0], height=0.1*hlim)
                            pkA = grad[xdata <= 10.0][pks].argmax()
                            xA = xdata[xdata <= 10.0][pks][pkA]

                            xmax = xdata.max()-7.0
                            hlim = grad[(xdata >= (xdata.max()-7.0))&(xdata <= (xdata.max()-4.0))].max()
                            pks,_ = find_peaks(grad[(xdata >= (xdata.max()-7.0))&(xdata <= (xdata.max()-4.0))],height=0.1*hlim)
                            pkB = grad[(xdata >= (xdata.max()-7.0))&(xdata <= (xdata.max()-4.0))][pks].argmax()
                            xB = xdata[(xdata >= (xdata.max()-7.0))&(xdata <= (xdata.max()-4.0))][pks][pkB]

                            #print(xA,xB)
                            # append depth estimate
                            dest.append(xB-xA)

                    f,ax = plt.subplots()
                    ax.plot(dest_thrust,'bx',label="Thrust")
                    ax.plot(dest_torque,'rx',label="Torque")
                    # plot nominal depth lines
                    ax.plot(len(dest_thrust)*[21.0,],'k-',label="Nominal")
                    # set labels
                    ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)")
                    ax.legend()
                    f.suptitle(f"{tool} Depth Estimate From Rolling Gradient of {name.upper()} Filter N=({NA},{NB})")
                    f.savefig(f"{tool}-{'-'.join(name.lower().split(' '))}-depth-estimate-rolling-gradient-Nmeth-{NA}-Nwin-{NB}.png")
                    plt.close(f)
                    
                    # update statistics
                    dest_thrust_var[name][-1].append(np.var(dest_thrust))
                    dest_torque_var[name][-1].append(np.var(dest_torque))

                    dest_thrust_mean[name][-1].append(np.mean(dest_thrust))
                    dest_torque_mean[name][-1].append(np.mean(dest_torque))
        # assign N values
        dest_thrust_var['NB'] = NN
        dest_torque_var['NB'] = NN

        dest_thrust_mean['NB'] = NN
        dest_torque_mean['NB'] = NN

        dest_thrust_var['NA'] = NN
        dest_torque_var['NA'] = NN

        dest_thrust_mean['NA'] = NN
        dest_torque_mean['NA'] = NN

        json.dump(dest_thrust_var,open(f"{tool}-Thrust-depth-estimate-filter-rolling-gradient-var.json",'w'))
        json.dump(dest_torque_var,open(f"{tool}-Torque-depth-estimate-filter-rolling-gradient-var.json",'w'))
        json.dump(dest_thrust_mean,open(f"{tool}-Thrust-depth-estimate-filter-rolling-gradient-mean.json",'w'))
        json.dump(dest_torque_mean,open(f"{tool}-Thrust-depth-estimate-filter-rolling-gradient-mean.json",'w'))

        for var,stat_dict in zip(["Thrust","Torque"],[dest_thrust_var,dest_torque_var]):
            for name in ['Weighted Savgol','Weiner']:
                stat = stat_dict[name]
                f,ax = plot_depthest(np.array(stat),xlabel="Rolling Gradient Window Size",ylabel="Smoothing Window Size",xticks=NN,yticks=NN,cmap="magma")
                f.suptitle(f"{tool} {var} Variance of Depth Estimate\nFrom Rolling Gradient of {name.upper()}")
                f.savefig(f"{tool}-{var}-{'-'.join(name.lower().split(' '))}-depth-estimate-filter-rolling-gradient-var.png")
                plt.close(f)

        for var,stat_dict in zip(["Thrust","Torque"],[dest_thrust_mean,dest_torque_mean]):
            for name in ['Weighted Savgol','Weiner']:
                stat = stat_dict[name]
                f,ax = plot_depthest(np.array(stat),xlabel="Rolling Gradient Window Size",ylabel="Smoothing Window Size",xticks=NN,yticks=NN,cmap="magma")
                f.suptitle(f"{tool} {var} Mean of Depth Estimate\nFrom Rolling Gradient of {name.upper()}")
                f.savefig(f"{tool}-{var}-{'-'.join(name.lower().split(' '))}-depth-estimate-filter-rolling-gradient-mean.png")
                plt.close(f)

def depth_est_evaltime(path="xls/UC*.xls",choice=0,NN=[10,20,30,40,50],its=1000):
    '''
        Evaluate the time to estimate depth and plot the results

        Runs time_depth_est_rolling at the given settings and plots the resutls

        Generates a scatter plot for all iterations at different window sizes and
        a scatter plot of the average and variance at difference window sizes

        Tool is extracted from the first two characters of the path

        The filenames for the plots are generated as following

        "{tool}-weiner-rolling-gradient-evaluation-time.png"

        "{tool}-savgol-rolling-gradient-evaluation-time.png"

        "{tool}-weiner-rolling-gradient-evaluation-time-statistics.png"

        "{tool}-savgol-rolling-gradient-evaluation-time-statistics.png"

        Inputs:
            path : Wildcard path to choose from
            choice : Index or string indicating which path to use. If index, used to index list of unsorted files. If string is "random", then a random one is chosen using numpy.random.choice.
            NN : Range of window sizes to try
            its : Number of iterations per window size
    '''
    # choose the file to use
    if isinstance(choice,int):
        fn = list(glob(path))[0]
    elif choice == "random":
        fn = np.random.choice(list(glob(path)),1)[0]
    # get the tool from the first two characters
    tool = os.path.splitext(os.path.basename(fn))[0][:2]
    # load file
    data = dp.loadSetitecXls(fn)[-1]
    # get x and y data
    xdata = np.abs(data['Position (mm)'].values.flatten())
    ydata = data['I Torque (A)'].values.flatten()
    # perform time estimation getting the full dictionary
    weiner_timer,savgol_timer = time_depth_est_rolling(xdata,ydata,NN=NN,its=its)

    ## plot the full eval time dataset
    f,ax= replot_evaltime(weiner_timer,xlabel="Window Size",ylabel="Evaluation Time (s)",title=f"Weiner Rolling Depth Est Evaluation Time its=1000")
    f.savefig(f"{tool}-weiner-rolling-gradient-evaluation-time.png")
    plt.close(f)
    
    f,ax= replot_evaltime(savgol_timer,xlabel="Window Size",ylabel="Evaluation Time (s)",title=f"Savgol Rolling Depth Est Evaluation Time its=1000")
    f.savefig(f"{tool}-savgol-rolling-gradient-evaluation-time.png")
    plt.close(f)

    ## plot the average and variance in the eval time across different window sizes
    f,ax=replot_mean_evaltime(weiner_timer,title="Weiner Rolling Depth Est Evaluation Time its=1000")
    f.savefig(f"{tool}-weiner-rolling-gradient-evaluation-time-statistics.png")
    plt.close(f)

    f,ax=replot_mean_evaltime(savgol_timer,title="Savgol Rolling Depth Est Evaluation Time its=1000")
    f.savefig(f"{tool}-savgol-rolling-gradient-evaluation-time-statistics.png")
    plt.close(f)

def depth_est_evaltime_all(NN=[10,20,30,40,50],its=1000):
    '''
        Evaluate time of each file in the path
    '''
    # create dictionaries to hold the eval times at different window sizes
    times_weiner = {N:len(PATHS)*[None,] for N in NN}
    times_savgol = {N:len(PATHS)*[None,] for N in NN}
    # iterate over each of the paths
    for fi,fn in enumerate(PATHS):
        # get the data
        data = dp.loadSetitecXls(fn)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten())
        ydata = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
        # evaluate its times
        weiner_timer,savgol_timer = time_depth_est_rolling(xdata,ydata,NN=NN,its=its)
        # append to the dictionary
        for N,tt in weiner_timer.items():
            times_weiner[N][fi]=tt
        for N,tt in savgol_timer.items():
            times_savgol[N][fi]=tt
    # return dicts
    return times_weiner, times_savgol

##def mt_depth_eval(N,cfact=None,dexp=32.0,win=3.0):
##    from multiprocessing import Pool
##    return Pool(5).starmap(calc_depth,[(x,N,cfact,dexp,win) for x in PATHS])

##def calc_depth(path,N,cf=None,depth_exp=32.0,win=3.0):
##    data = dp.loadSetitecXls(path)[-1]
##    xdata = np.abs(data['Position (mm)'].values.flatten())
##    ydata = data['I Torque (A)'].values.flatten()
##    if not (cf is None):
##        return depth_est_rolling(ydata,xdata,NA=N,xstart=10.0,depth_exp=depth_exp,depth_win=win,default=True,fact=cf)
##    else:
##        return depth_est_rolling(ydata,xdata,NA=N,xstart=10.0,depth_exp=depth_exp,depth_win=win,default=True)

def tt(y,x,N,dexp,winp):
    from timeit import default_timer as timer
    start = timer()
    depth_est_rolling(y,x,NA=N,depth_exp=dexp,depth_win=winp)
    end = timer()
    return end-start

# evaluate the time it takes to perform a depth estimate
# the depth est is performed with Torque + Empty
# a multiprocessing Pool is used to speed up the estimations for each iteration
def eval_time_mt(path,N,depth_exp=32.0,win=3.0,its=1000):
    from multiprocessing import Pool
    # load file
    data = dp.loadSetitecXls(path)[-1]
    # get data
    xdata = np.abs(data['Position (mm)'].values.flatten())
    ydata = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
    # split the iterations across a processing pool
    return Pool(5).starmap(tt,[(ydata,xdata,N,depth_exp,win) for _ in range(its)])

# evaluate the time it takes to perform a depth estimate
# the depth estimate is performed its number of times
# returns a dictionary of list-of-list representing all times for each file at different window sizes
def eval_time_mt_all(paths=PATHS,NN=[10,20,30,40,50],depth_exp=32.0,win=3.0,its=100):
    return {N : [eval_time_mt(fn,N,depth_exp,win,its) for fn in paths] for N in NN}

# calculate the depth estimate using multithreading
# finds percentage of files within the target range set by acc
def depth_est_eval_acc(paths=PATHS,exp=32.0,win=3.0,acc=(1/16)*25.4,NN=[10,20,30,40,50]):
    dest_dict = {N : [float(v) for v in mt_depth_eval(N).tolist()] for N in NN}
    json.dump(dest_dict,open(f"depth-estimate-exp-{exp}-win-{win}.json",'w'))
    nf = float(len(paths))
    acc_perc = []
    acc_dist = []
    for N,dest in dest_dict.items():
        # find percentage of values within target range
        ii = np.where((dest >= (exp-acc)) & (dest <= (exp+acc)))[0]
        # store number of files that are within the target range
        acc_perc.append(float(ii)/nf)
        # calculate distance from expected distance
        acc_dist.append(np.abs(exp-dest))
        # plot histogram of depth estimates
        f,ax = plt.subplots()
        ax.hist(dest,bins=20)
        ax.set(xlabel="Depth Estimate (mm)",ylabel="Population",title=f"Histogram of Depth Estimate N={N}")
        f.savefig(f"depth-estimate-histogram-N-{N}.png")
        plt.close(f)
        # plot histogram of distance to expected depths
        f,ax = plt.subplots()
        ax.hist(acc_dist[-1],bins=20)
        ax.set(xlabel="Distance from Expected Depth (mm)",ylabel="Population",title=f"Histogram of Distance from Expected Depth {exp}\nWindow Size N={N}")
        f.savefig(f"depth-estimate-distance-histogram-exp-{exp}-N-{N}.png")
        plt.close(f)
        
    # plot the percentage of files within range
    f,ax = plt.subplots()
    ax.plot(NN,acc_perc,'x')
    ax.set(xlabel="Window Size",ylabel="Percentage of Files within Range (%)")
    f.suptitle(f"Percentage of Files Within Range {exp} +/- {acc:.2f}")
    f.savefig(f"acc-files-within-range-perc-exp-{exp}-win-{win}-acc-{acc:.2f}.png")
    plt.close(f)
    ## plot the distance from expected
    f,ax = plt.subplots(ncols=2,nrows=2,constrained_layout=True)
    ax[1,0].plot(NN,[np.mean(dd) for dd in acc_dist])
    ax[1,0].set(xlabel="Window Size",ylabel="Average Distance",title="Average Distance")
    ax[1,1].plot(NN,[np.var(dd) for dd in acc_dist])
    ax[1,1].set(xlabel="Window Size",ylabel="Variance Distance",title="Distance Variance")
    ax[0,0].plot(NN,[np.min(dd) for dd in acc_dist])
    ax[0,0].set(xlabel="Window Size",ylabel="Minimum Distance",title="Minimum Distance")
    ax[0,1].plot(NN,[np.mean(dd) for dd in acc_dist])
    ax[0,1].set(xlabel="Window Size",ylabel="Maximum Distance",title="Maximum Distance")
    f.suptitle(f"Statistics of Distance from {exp}")
    f.savefig(f"acc-statistics-exp-{exp}-win-{win}-acc-{acc:.2f}.png")
    plt.close(f)
    
def plot_filtered_kistler(path):
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get thrust data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        for var in ["Torque","Thrust"]:
            ydata = data[f'I {var} (A)'].values.flatten()
            if var == "Torque":
                ydata += data['I Torque Empty (A)'].values.flatten()
            # create axes
            f,ax = plt.subplots()
            # plot original data
            ax.plot(xdata,ydata,'b-')
            ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)",title=f"{fname} Wiener Filter")
            # plot data at different window sizes
            for N in [10,20,30,40,50]:
                filt_weight = wiener(ydata,N)
                np.nan_to_num(filt_weight,False)
                ax.plot(xdata,filt_weight,label=f"N={N}")
            ax.legend()
            f.savefig(f"{fname}-{var}-wiener-different-window-sizes.png")
            plt.close()

        for var in ["Torque","Thrust"]:
            ydata = data[f'I {var} (A)'].values.flatten()
            if var == "Torque":
                ydata += data['I Torque Empty (A)'].values.flatten()
            for o in [1,2,3,4,5]:
                # create axes
                f,ax = plt.subplots()
                # plot original data
                ax.plot(xdata,ydata,'b-')
                ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)",title=f"{fname}\nWeighted Savgol Filter Order={o}")
                # plot data at different window sizes
                for N in [10,20,30,40,50]:
                    filt_weight = weighted_savgol_filter(ydata,N,o)
                    np.nan_to_num(filt_weight,False)
                    ax.plot(xdata,filt_weight,label=f"N={N}")
                ax.legend()
                f.savefig(f"{fname}-{var}-weighted-savgol-different-window-sizes-order-{o}.png")
                plt.close()

def plot_filt_rolling_gradient_kistler(path):
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get thrust data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over variable
        for var in ["Torque","Thrust"]:
            ydata = data[f'I {var} (A)'].values.flatten()
            if var == "Torque":
                ydata += data['I Torque Empty (A)'].values.flatten()
            # plot data at different window sizes
            for N in [10,20,30,40,50]:
                for method,name in zip([lambda y,N=N : wiener(y,N), lambda y,N=N: weighted_savgol_filter(y,N)],["Wiener","Weighted Savgol"]):
                    # create axis
                    f,ax = plt.subplots()
                    # plot original data
                    ax.plot(xdata,ydata,'b-',label="Original")
                    # set axis labels
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    # methods to try
                    # update title
                    ax.set_title(f"{fname} {var}\n{name} Filter N={N}")
                    # filter data
                    filt_weight = method(ydata,N)
                    np.nan_to_num(filt_weight,False)
                    # calculate rolling gradient
                    grad = rolling_gradient(filt_weight,N,False)
                    # plot the filtered data
                    ax.plot(xdata,filt_weight,'r-',label="Filtered")
                    ax.legend()
                    # create twin axis
                    cax = ax.twinx()
                    # plot rolling gradient
                    cax.plot(xdata,grad,'k-')
                    cax.set_ylabel("Rolling Gradient")
                    # save fig
                    f.savefig(f"{fname}-{var}-{'-'.join(name.lower().split(' '))}-rolling-gradient-N-{N}.png")
                    plt.close(f)

def plot_kistler_rolling_gradient_win(path):
    from scipy.signal.windows import tukey
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get thrust data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over variable
        for var in ["Torque","Thrust"]:
            ydata = data[f'I {var} (A)'].values.flatten()
            if var == "Torque":
                ydata += data['I Torque Empty (A)'].values.flatten()
            # plot data at different window sizes
            for N in [20,30]:
                for method,name in zip([lambda y,N=N : wiener(y,N), lambda y,N=N: weighted_savgol_filter(y,N)],["Wiener","Weighted Savgol"]):
                    # create axis
                    f,ax = plt.subplots(constrained_layout=True)
                    # plot original data
                    ax.plot(xdata,ydata,'b-',label="Original")
                    # set axis labels
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    # filter data
                    filt_weight = method(ydata,N)
                    np.nan_to_num(filt_weight,False)
                    # plot the filtered data
                    ax.plot(xdata,filt_weight,'r-',label="Filtered")
                    #ax.legend()
                    # create twin axis
                    cax = ax.twinx()
                    # calculate rolling gradient
                    grad = rolling_gradient(filt_weight,N,False)
                    # create a symmetrical window
                    win = tukey(len(grad),0.1,True)
                    # multiply gradient with windlw
                    grad_win = grad*win
                    # plot orignal rolling gradient
                    cax.plot(xdata,grad,'k-',label="R. Grad")
                    # plot windowed rolling gradient
                    cax.plot(xdata,grad_win,'m-',label="Win. R. Grad")
                    cax.set_ylabel("Rolling Gradient")
                    #cax.legend()

                    # combine legends together
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = cax.get_legend_handles_labels()
                    cax.legend(lines + lines2, labels + labels2, loc=0)
                    
                    # set figure title
                    f.suptitle(f"{fname} {name}\n{var} Tukey Rolling Gradient N={N}")
                    # save fig
                    f.savefig(f"{fname}-{var}-{'-'.join(name.lower().split(' '))}-win-tukey-rolling-gradient-N-{N}.png")
                    plt.close(f)    

def plot_all_kistler(path,cmap=None):
    from matplotlib import cm
    if isinstance(cmap,str):
        cmap = cm.get_cmap(cmap)

    ax = {}
    nf = len(sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]))
    for fi,fn in enumerate(sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]),1):
        fname = os.path.splitext(os.path.basename(fn))[0]
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get thrust data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over variable
        for var in ["Torque","Thrust"]:
            if not (var in ax):
                ax[var] = plt.subplots(constrained_layout=True)[1]
            ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            if cmap is None:
                ax[var].plot(xdata,ydata)
            else:
                ax[var].plot(xdata,ydata,c=cmap(fi/nf))

    for k,aa in ax.items():
        aa.set(xlabel="Position (mm)",ylabel=k,title=f"Kestler Seitec {k}")
        aa.figure.savefig(f"kestler-setitec-{k}-all.png")
        plt.close(aa.figure)

# vector of tool lengths
kistler_sides = [1.2,3,5,35.8,2.02]
# estimate the depth from the Kistler setitec data
def kistler_depth_run(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],plot_steps=True,add_thrust_empty=False,use_signs=True,xstart=10.0,end_ref='end',xendA=30.0,xendB=35.0,default=False,pselect='argmax'):
    # create dictionary to hold depth estimate
    depth_est = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        print(fn)
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            print(var)
            #print(f"Processing {var}")
            # add entry for variable to depth estimate
            if not (var in depth_est):
                depth_est[var] = {}
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            # for a given window size
            for N in NN:
                print(N)
                if plot_steps:
                    # smooth using wiener
                    filt_weight = wiener(ydata,N)
                    # perform rolling gradient on smoothed data
                    grad = rolling_gradient(filt_weight,N,False,use_signs)
                    # weight to remove edge artifacts
                    win = tukey(len(grad),0.1,True)
                    grad *= win
                    pks,_ = find_peaks(grad,height=grad.max()*0.1)
                    # plot data
                    f,ax = plt.subplots(constrained_layout=True)
                    ax.plot(xdata,ydata,'b-',label='Original')
                    ax.plot(xdata,filt_weight,'r-',label="Filtered")
                    cax = ax.twinx()
                    cax.plot(xdata,grad,'k-',label="Rolling Gradient")
                    cax.plot(xdata[pks],grad[pks],'gx',markersize=10,markeredgewidth=4,label="Peaks")
                    ## process gradient as it would in depth estimaton
                    # first window
                    mask = xdata <= xstart
                    grad_mask = grad[mask]
##                    if kwargs.get('use_signs',True):
##                        grad_mask *= 1.0
                    # plot modded gradient in first window
                    cax.plot(xdata[mask],grad_mask,'m-',label="1st Window")
                    ## 2nd window
                    # check teference point for 2nd window period
                    # if referencing the end
                    if end_ref=='end':
                        mask = (xdata >= (xdata.max()-xendA)) & (xdata <= (xdata.max()-xendB))
                    # if referencing the start
                    elif end_ref=='start':
                        mask = (xdata >= (xdata.min()+xendA)) & (xdata <= (xdata.min()+xendB))
                    # mask the gradient
                    grad_mask = grad[mask]
                    # if keeping signs invert to focus on negatives
                    if use_signs:
                        grad_mask *= 1.0
                        
                    cax.plot(xdata[mask],grad_mask,'c-',label="2nd Window")
                    # set labels
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    cax.set_ylabel("Rolling Gradient")
                    # combine legends together
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = cax.get_legend_handles_labels()
                    cax.legend(lines + lines2, labels + labels2, loc=0)
                    #plt.show()
                    if add_thrust_empty:
                        f.suptitle(f"{fname} {var}\nWiener Filtered + Rolling Gradient N={N} All Empty Added")
                        f.savefig(f"{fname}-{var}-wiener-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}-all-empty.png")
                        
                    else:
                        f.suptitle(f"{fname} {var}\nWiener Filtered + Rolling Gradient N={N}")
                        f.savefig(f"{fname}-{var}-wiener-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}.png")
                #print(f"Processing win size {N}")
                # calculate the depth estimate
                dest = depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,xendA=xendA,xendB=xendB,default=default,end_ref=end_ref,pselect=pselect)
                #print(dest)
                # save result in dictionary
                if not (N in depth_est[var]):
                    depth_est[var][N] = []
                depth_est[var][N].append(dest)
                #print(len(depth_est[var][N]))
                plt.close('all')
    # iterate over each window size    
    for N in NN:
        # create axes
        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Torque"][N],'bx',label="Torque")
        # plot thrust depth estimate with red X's
        ax.plot(depth_est["Thrust"][N],'rx',label="Thrust")
        # draw a black line for nominal depth
        ax.plot(len(depth_est["Thrust"][N])*[32.0,],'k-',label="Nominal")
        # create legend
        ax.legend()
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate N={N} with Retry")
        # save figure
        f.savefig(f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-with-retry.png")
        plt.close(f)
    return depth_est

def kistler_depth_run_dexp(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],plot_steps=True,add_thrust_empty=False,use_signs=False,xstart=10.0,end_ref='end',depth_exp=32.0,depth_win=4.0,default=True,pselect='argmax',opath='',ref_line=None):
    # create dictionary to hold depth estimate
    depth_est = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        print(fn)
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            print(var)
            #print(f"Processing {var}")
            # add entry for variable to depth estimate
            if not (var in depth_est):
                depth_est[var] = {}
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            # for a given window size
            for N in NN:
                print(N)
                if plot_steps:
                    # smooth using wiener
                    filt_weight = wiener(ydata,N)
                    # perform rolling gradient on smoothed data
                    grad = rolling_gradient(filt_weight,N,False,use_signs)
                    # weight to remove edge artifacts
                    win = tukey(len(grad),0.1,True)
                    grad *= win
                    pks,_ = find_peaks(grad,height=grad.max()*0.1)
                    # plot data
                    f,ax = plt.subplots(constrained_layout=True)
                    ax.plot(xdata,ydata,'b-',label='Original')
                    ax.plot(xdata,filt_weight,'r-',label="Filtered")
                    cax = ax.twinx()
                    cax.plot(xdata,grad,'k-',label="Rolling Gradient")
                    cax.plot(xdata[pks],grad[pks],'gx',markersize=10,markeredgewidth=4,label="Peaks")
                    ## process gradient as it would in depth estimaton
                    # first window
                    mask = xdata <= xstart
                    grad_mask = grad[mask]
##                    if kwargs.get('use_signs',True):
##                        grad_mask *= 1.0
                    # plot modded gradient in first window
                    cax.plot(xdata[mask],grad_mask,'m-',label="1st Window")
                    ## 2nd window
                    mask= (xdata >= (xdata.min()+depth_exp-(win/2)))&(xdata <= (xdata.min()+depth_exp+(win/2)))
                    # mask the gradient
                    grad_mask = grad[mask]
                    # if keeping signs invert to focus on negatives
                    if use_signs:
                        grad_mask *= 1.0
                        
                    cax.plot(xdata[mask],grad_mask,'c-',label="2nd Window")
                    # set labels
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    cax.set_ylabel("Rolling Gradient")
                    # combine legends together
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = cax.get_legend_handles_labels()
                    cax.legend(lines + lines2, labels + labels2, loc=0)
                    #plt.show()
                    if add_thrust_empty:
                        f.suptitle(f"{fname} {var}\nWiener Filtered + Rolling Gradient N={N} All Empty Added")
                        f.savefig(os.path.join(opath,f"{fname}-{var}-wiener-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}-all-empty-depth-exp-{depth_exp}-window-{depth_win}.png"))
                        
                    else:
                        f.suptitle(f"{fname} {var}\nWiener Filtered + Rolling Gradient N={N}")
                        f.savefig(os.path.join(opath,f"{fname}-{var}-wiener-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}-depth-exp-{depth_exp}-window-{depth_win}.png"))
                #print(f"Processing win size {N}")
                # calculate the depth estimate
                #print("expected depth",depth_exp)
                dest = depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect)
                #print(dest)
                # save result in dictionary
                if not (N in depth_est[var]):
                    depth_est[var][N] = []
                depth_est[var][N].append(dest)
                #print(len(depth_est[var][N]))
                plt.close('all')
    
    fstats,axstats = plt.subplots(nrows=2,ncols=2,sharex=True,constrained_layout=True)
    thrust_mean = []
    thrust_var = []
    thrust_std = []
    torque_mean = []
    torque_var = []
    torque_std = []
    
    data = {'NN':NN,'depth_est' : {'Thrust' : [], 'Torque': []}, 'mean' : {'Thrust' : [], 'Torque': []}, 'var' : {'Thrust' : [], 'Torque': []}, 'std' : {'Thrust' : [], 'Torque': []}}
    # iterate over each window size    
    for N in NN:
        # create axes
        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Torque"][N],'bx',label="Torque")
        torque_mean.append(np.mean(depth_est["Torque"][N]))
        torque_var.append(np.var(depth_est["Torque"][N]))
        torque_std.append(np.std(depth_est["Torque"][N]))
        # plot thrust depth estimate with red X's
        ax.plot(depth_est["Thrust"][N],'rx',label="Thrust")
        thrust_mean.append(np.mean(depth_est["Thrust"][N]))
        thrust_var.append(np.var(depth_est["Thrust"][N]))
        thrust_std.append(np.std(depth_est["Thrust"][N]))
        # draw a black line for nominal depth
        if not (ref_line is None):
            ax.plot(len(depth_est["Thrust"][N])*[ref_line,],'k-',label="Nominal")
        # create legend
        ax.legend()
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate N={N} with Retry")
        # save figure
        f.savefig(os.path.join(opath,f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-depth-exp-{depth_exp}-window-{depth_win}-with-retry-grad-v2.png"))
        plt.close(f)

        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Torque"][N],'bx')
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate (Torque) N={N} with Retry")
        f.savefig(os.path.join(opath,f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-depth-exp-{depth_exp}-window-{depth_win}-with-retry-torque-only-grad-v2.png"))
        plt.close(f)

        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Thrust"][N],'rx')
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate (Thrust) N={N} with Retry")
        f.savefig(os.path.join(opath,f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-depth-exp-{depth_exp}-window-{depth_win}-with-retry-thrust-only-grad-v2.png"))
        plt.close(f)

        data['depth_est']['Torque'].append([float(x) for x in depth_est["Torque"][N]])
        data['depth_est']['Thrust'].append([float(x) for x in depth_est["Thrust"][N]])
        data['mean']['Torque'].append(float(torque_mean[-1]))
        data['var']['Torque'].append(float(torque_var[-1]))
        data['std']['Torque'].append(float(torque_std[-1]))
        data['mean']['Thrust'].append(float(thrust_mean[-1]))
        data['var']['Thrust'].append(float(thrust_var[-1]))
        data['std']['Thrust'].append(float(thrust_std[-1]))
    # save data
    json.dump(data,open(os.path.join(opath,f"depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-grad-v2.json"),'w'))

    # plot the statistics of each window size
    axstats[0,0].plot(NN,torque_mean,'b-')
    axstats[0,1].plot(NN,torque_var,'r-')
    axstats[1,0].plot(NN,thrust_mean,'b-')
    axstats[1,1].plot(NN,thrust_var,'r-')

    axstats[0,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Torque)")
    axstats[0,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Torque)")
    axstats[1,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Thrust)")
    axstats[1,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Thrust)")
    fstats.suptitle(f"Stats about Depth Est. using depth_exp={depth_exp}, window={depth_win}")
    fstats.savefig(os.path.join(opath,f"depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-grad-v2.png"))
    return depth_est

def kistler_depth_run_gmin(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],add_thrust_empty=True,use_signs=True,xstart=10.0,end_ref='end',default=True,pselect='argmax',opath=''):
    # create dictionary to hold depth estimate
    depth_est = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        print(fn)
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            print(var)
            #print(f"Processing {var}")
            # add entry for variable to depth estimate
            if not (var in depth_est):
                depth_est[var] = {}
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            # for a given window size
            for N in NN:
                print(N)
                #print(f"Processing win size {N}")
                # calculate the depth estimate
                #print("expected depth",depth_exp)
                dest = depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,try_gmin=True,default=default,end_ref=end_ref,pselect=pselect)
                #print(dest)
                # save result in dictionary
                if not (N in depth_est[var]):
                    depth_est[var][N] = []
                depth_est[var][N].append(dest)
                #print(len(depth_est[var][N]))
                plt.close('all')
    
    fstats,axstats = plt.subplots(nrows=2,ncols=2,sharex=True,constrained_layout=True)
    thrust_mean = []
    thrust_var = []
    thrust_std = []
    torque_mean = []
    torque_var = []
    torque_std = []
    
    data = {'NN':NN,'depth_est' : {'Thrust' : [], 'Torque': []}, 'mean' : {'Thrust' : [], 'Torque': []}, 'var' : {'Thrust' : [], 'Torque': []}, 'std' : {'Thrust' : [], 'Torque': []}}
    # iterate over each window size    
    for N in NN:
        # create axes
        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Torque"][N],'bx',label="Torque")
        torque_mean.append(np.mean(depth_est["Torque"][N]))
        torque_var.append(np.var(depth_est["Torque"][N]))
        torque_std.append(np.std(depth_est["Torque"][N]))
        # plot thrust depth estimate with red X's
        ax.plot(depth_est["Thrust"][N],'rx',label="Thrust")
        thrust_mean.append(np.mean(depth_est["Thrust"][N]))
        thrust_var.append(np.var(depth_est["Thrust"][N]))
        thrust_std.append(np.std(depth_est["Thrust"][N]))
        # draw a black line for nominal depth
        #ax.plot(len(depth_est["Thrust"][N])*[32.0,],'k-',label="Nominal")
        # create legend
        ax.legend()
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate N={N} with Retry")
        # save figure
        f.savefig(os.path.join(opath,f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-try-gmin.png"))
        plt.close(f)

        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Torque"][N],'bx')
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate (Torque) N={N} with Retry")
        f.savefig(os.path.join(opath,f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-try-gmin-torque-only.png"))
        plt.close(f)

        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Thrust"][N],'rx')
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate (Thrust) N={N} with Retry")
        f.savefig(os.path.join(opath,f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-try-gmin-thrust-only.png"))
        plt.close(f)

        data['depth_est']['Torque'].append([float(x) for x in depth_est["Torque"][N]])
        data['depth_est']['Thrust'].append([float(x) for x in depth_est["Thrust"][N]])
        data['mean']['Torque'].append(float(torque_mean[-1]))
        data['var']['Torque'].append(float(torque_var[-1]))
        data['std']['Torque'].append(float(torque_std[-1]))
        data['mean']['Thrust'].append(float(thrust_mean[-1]))
        data['var']['Thrust'].append(float(thrust_var[-1]))
        data['std']['Thrust'].append(float(thrust_std[-1]))
    # save data
    json.dump(data,open(os.path.join(opath,f"depth-estimate-stats-exp-try-gmin.json"),'w'))

    # plot the statistics of each window size
    axstats[0,0].plot(NN,torque_mean,'b-')
    axstats[0,1].plot(NN,torque_var,'r-')
    axstats[1,0].plot(NN,thrust_mean,'b-')
    axstats[1,1].plot(NN,thrust_var,'r-')

    axstats[0,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Torque)")
    axstats[0,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Torque)")
    axstats[1,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Thrust)")
    axstats[1,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Thrust)")
    fstats.suptitle(f"Stats about Depth Est. using Min. Gradient")
    fstats.savefig(os.path.join(opath,f"depth-estimate-stats-exp-try-gmin.png"))
    return depth_est

def kistler_depth_run_cfact(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],plot_steps=True,add_thrust_empty=False,use_signs=True,xstart=10.0,end_ref='end',depth_exp=32.0,depth_win=4.0,default=False,pselect='argmax',cfact=0.99):
    # create dictionary to hold depth estimate
    depth_est = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        print(fn)
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            print(var)
            #print(f"Processing {var}")
            # add entry for variable to depth estimate
            if not (var in depth_est):
                depth_est[var] = {}
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            # for a given window size
            for N in NN:
                print(N)
                if plot_steps:
                    # smooth using wiener
                    filt_weight = wiener(ydata,N)
                    # perform rolling gradient on smoothed data
                    grad = rolling_gradient(filt_weight,N,False,use_signs)
                    # weight to remove edge artifacts
                    win = tukey(len(grad),0.1,True)
                    grad *= win
                    pks,_ = find_peaks(grad,height=grad.max()*0.1)
                    results_full = peak_widths(grad,pks,rel_height=cfact)
                    # plot data
                    f,ax = plt.subplots(constrained_layout=True)
                    # plot original data
                    ax.plot(xdata,ydata,'b-',label='Original')
                    # plot smoothed data
                    ax.plot(xdata,filt_weight,'r-',label="Filtered")
                    cax = ax.twinx()
                    # plot the rolling gradient of smoothed signal
                    cax.plot(xdata,grad,'k-',label="Rolling Gradient")
                    # plot the peaks
                    cax.plot(xdata[pks],grad[pks],'gx',markersize=10,markeredgewidth=4,label="Peaks")
                    cax.hlines(results_full[1],xmin=xdata[results_full[2].astype("int32")],xmax=xdata[results_full[3].astype("int32")],colors='C0',linestyles='solid',label="Peak Widths")
                    ## process gradient as it would in depth estimaton
                    # first window
                    mask = xdata <= xstart
                    grad_mask = grad[mask]
##                    if kwargs.get('use_signs',True):
##                        grad_mask *= 1.0
                    # plot modded gradient in first window
                    cax.plot(xdata[mask],grad_mask,'m-',label="1st Window")
                    ## 2nd window
                    mask= (xdata >= (xdata.min()+depth_exp-(win/2)))&(xdata <= (xdata.min()+depth_exp+(win/2)))
                    # mask the gradient
                    grad_mask = grad[mask]
                    # if keeping signs invert to focus on negatives
                    if use_signs:
                        grad_mask *= 1.0
                        
                    cax.plot(xdata[mask],grad_mask,'c-',label="2nd Window")
                    # set labels
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    cax.set_ylabel("Rolling Gradient")
                    # combine legends together
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = cax.get_legend_handles_labels()
                    cax.legend(lines + lines2, labels + labels2, loc=0)
                    #plt.show()
                    if add_thrust_empty:
                        f.suptitle(f"{fname} {var}\nWiener Filtered + Rolling Gradient N={N} All Empty Added")
                        f.savefig(f"{fname}-{var}-wiener-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}-all-empty-depth-exp-{depth_exp}-window-{depth_win}-cfact-{cfact:.2f}.png")
                        
                    else:
                        f.suptitle(f"{fname} {var}\nWiener Filtered + Rolling Gradient N={N}")
                        f.savefig(f"{fname}-{var}-wiener-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}-depth-exp-{depth_exp}-window-{depth_win}-cfact-{cfact:.2f}.png")
                #print(f"Processing win size {N}")
                # calculate the depth estimate
                #print("expected depth",depth_exp)
                dest = depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect,cfact=cfact)
                #print(dest)
                # save result in dictionary
                if not (N in depth_est[var]):
                    depth_est[var][N] = []
                depth_est[var][N].append(dest)
                #print(len(depth_est[var][N]))
                plt.close('all')
    fstats,axstats = plt.subplots(nrows=2,ncols=3,sharex=True,constrained_layout=True)
    thrust_mean = []
    thrust_var = []
    thrust_std = []
    torque_mean = []
    torque_var = []
    torque_std = []
    
    data = {'NN':NN,'depth_est' : {'Thrust' : [], 'Torque': []}, 'mean' : {'Thrust' : [], 'Torque': []}, 'var' : {'Thrust' : [], 'Torque': []}, 'std' : {'Thrust' : [], 'Torque': []}}
    # iterate over each window size    
    for N in NN:
        # create axes
        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Torque"][N],'bx',label="Torque")
        torque_mean.append(np.mean(depth_est["Torque"][N]))
        torque_var.append(np.var(depth_est["Torque"][N]))
        torque_std.append(np.std(depth_est["Torque"][N]))
        # plot thrust depth estimate with red X's
        ax.plot(depth_est["Thrust"][N],'rx',label="Thrust")
        thrust_mean.append(np.mean(depth_est["Thrust"][N]))
        thrust_var.append(np.var(depth_est["Thrust"][N]))
        thrust_std.append(np.std(depth_est["Thrust"][N]))
        # draw a black line for nominal depth
        ax.plot(len(depth_est["Thrust"][N])*[32.0,],'k-',label="Nominal")
        # create legend
        ax.legend()
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate N={N} with Retry")
        # save figure
        f.savefig(f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-depth-exp-{depth_exp}-window-{depth_win}-cfact-{cfact:.2f}.png")
        plt.close(f)

        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Torque"][N],'bx')
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate (Torque) N={N} with Retry")
        f.savefig(f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-depth-exp-{depth_exp}-window-{depth_win}-cfact-{cfact:.2f}-torque-only.png")
        plt.close(f)

        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Thrust"][N],'rx')
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate (Thrust) N={N} with Retry")
        f.savefig(f"depth-estimate-kestler-all-var-N-{N}-pselect-argmax-{'using-sign-' if use_signs else ''}{'-all-empty' if add_thrust_empty else ''}-depth-exp-{depth_exp}-window-{depth_win}-cfact-{cfact:.2f}-thrust-only.png")
        plt.close(f)

        data['depth_est']['Torque'].append([float(x) for x in depth_est["Torque"][N]])
        data['depth_est']['Thrust'].append([float(x) for x in depth_est["Thrust"][N]])
        data['mean']['Torque'].append(float(torque_mean[-1]))
        data['var']['Torque'].append(float(torque_var[-1]))
        data['std']['Torque'].append(float(torque_std[-1]))
        data['mean']['Thrust'].append(float(thrust_mean[-1]))
        data['var']['Thrust'].append(float(thrust_var[-1]))
        data['std']['Thrust'].append(float(thrust_std[-1]))
    # save data
    json.dump(data,open(f"depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-cfact-{cfact:.2f}.json",'w'))

    # plot the statistics of each window size
    axstats[0,0].plot(NN,torque_mean,'b-')
    axstats[0,1].plot(NN,torque_var,'r-')
    axstats[0,2].plot(NN,torque_std,'r-')
    axstats[1,0].plot(NN,thrust_mean,'b-')
    axstats[1,1].plot(NN,thrust_var,'r-')
    axstats[1,2].plot(NN,thrust_std,'r-')
    
    axstats[0,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Torque)")
    axstats[0,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Torque)")
    axstats[0,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Torque)")
    axstats[1,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Thrust)")
    axstats[1,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Thrust)")
    axstats[1,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Thrust)")

    fstats.suptitle(f"Stats about Depth Est. using depth_exp={depth_exp}, window={depth_win}, cfact={cfact}")
    fstats.savefig(f"depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-cfact-{cfact}.png")
    return depth_est

def kistler_depth_run_multiN(path="AirbusData/Seti-Tec data files life test/*.xls",NA=[10,20,30,40,50],NB=None,add_thrust_empty=False):
    if (NB is None) and (NA is not None):
        NB = NA
    elif (NB is not None) and (NA is None):
        NA = NB
        
    # create dictionary to hold depth estimate
    depth_est = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            #print(f"Processing {var}")
            if not (var in depth_est):
                depth_est[var] = [[None,]*len(NA)]
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            # for a given window size
            for na in NA:
                for nb in NB:
                    # smooth using wiener
                    filt_weight = wiener(ydata,na)
                    # perform rolling gradient on smoothed data
                    grad = rolling_gradient(filt_weight,nb)
                    # weight to remove edge artifacts
                    win = tukey(len(grad),0.1,True)
                    grad *= win
                    pks,_ = find_peaks(grad,height=grad.max()*0.1)
                    # plot data
                    f,ax = plt.subplots(constrained_layout=True)
                    ax.plot(xdata,ydata,'b-',label='Original')
                    ax.plot(xdata,filt_weight,'r-',label="Filtered")
                    cax = ax.twinx()
                    cax.plot(xdata,grad,'k-',label="Rolling Gradient")
                    cax.plot(xdata[pks],grad[pks],'gx',markersize=10,linewidth=6,label="Peaks")
                    
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    cax.set_ylabel("Rolling Gradient")
                    f.suptitle(f"{fname} {var}\nWiener Filtered + Rolling Gradient NA={na}, NB={nb}")
                    f.savefig(f"{fname}-{var}-wiener-tukey-rolling-gradient-pks-NA-{na}-NB-{nb}-pselect-limit.png")
                    plt.close(f)

                    #print(f"Processing win size {N}")
                    # calculate the depth estimate
                    try:
                        dest = depth_est_rolling(ydata,xdata,NA=N,xendB = 0.0,xendA=30.0,use_signs=True)
                    except ValueError:
                        print(f"Failed to process {fname} at N={N}! Setting dest to -1")
                        dest = -1
                        #plt.show()
                    #print(dest)
                    # save result in dictionary
                    if not (N in depth_est[var]):
                        depth_est[var][N] = []
                    depth_est[var][N].append(dest)
                    #print(len(depth_est[var][N]))
                    plt.close(f)

    ## plot the results
    # iterate over the dictionary for each variable
    for var,dd in depth_est.items():
        # iterate over the depth estimate for the different window sizes
        for N,dest in dd.items():
            # create axes
            f,ax = plt.subplots(constrained_layout=True)
            # plot depth estimate
            ax.plot(dest,'bx')
            # set labels
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"Depth Estimate using {var} N={N}")
            # save and cleanup
            f.savefig(f"depth-estimate-kestler-{var}-N-{N}-pselect-limit.png")
            plt.close(f)
    return depth_est

def kistler_signal_energy(path = "AirbusData/Seti-Tec data files life test/*.xls",add_thrust_empty = True):
    from scipy.signal import welch
    energy = {}
    psd = {}
    spectrum = {}
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            #print(f"Processing {var}")
            # add entry for variable to depth estimate
            if not (var in energy):
                energy[var] = []
                psd[var] = []
                spectrum[var] = []
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            energy[var].append(np.sum(np.abs(ydata)**2.0))
            psd[var].append(welch(ydata,fs=100.0,scaling='density')[1])
            spectrum[var].append(welch(ydata,fs=100.0,scaling='spectrum')[1])
    for var,e in energy.items():
        f,ax = plt.subplots()
        ax.plot(e)
        ax.set(xlabel="Hole Number",ylabel="Energy",title=f"{var} Signal Energy")
        f.savefig(f"kistler-{var}-signal-energy.png")
        plt.close(f)
    
    for var,e in psd.items():
        f,ax = plt.subplots()
        ax.plot(e)
        ax.set(xlabel="Freq (Hz)",ylabel="PSD (V**2/Hz)",title=f"{var} PSD")
        f.savefig(f"kistler-{var}-psd.png")
        plt.close(f)

    for var,e in psd.items():
        f,ax = plt.subplots()
        ax.plot(e)
        ax.set(xlabel="Freq (Hz)",ylabel="Power Spectrum (V**2)",title=f"{var} Power Spectrum")
        f.savefig(f"kistler-{var}-power-spectrum.png")
        plt.close(f)

# evaluate the depth estimate of each file
# calculate the distance to nominal depth
# return a dictionary of the closest files and their respective distance
def kistler_find_best_est(path="AirbusData/Seti-Tec data files life test/*.xls",N=30,dnom=32.0,num=5,add_thrust_empty=False,use_signs=False):
    # create dictionary to hold depth estimate
    depth_est = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            #print(f"Processing {var}")
            # add entry for variable to depth estimate
            if not (var in depth_est):
                depth_est[var] = []
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()   
            # calculate the depth estimate
            dest = depth_est_rolling(ydata,xdata,NA=N,xendB = 0.0,xendA=30.0,use_signs=use_signs)
##            try:
##                dest = depth_est_rolling(ydata,xdata,NA=N,xendB = 0.0,xendA=30.0,use_signs=use_signs)
##            except ValueError:
##                print(f"Failed to process {fname} at N={N}! Setting dest to -1")
##                dest = -1
            depth_est[var].append(dest)
    paths_list = list(sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]))
    # find distance to nominal
    dest_best = {}
    for var,dest in depth_est.items():
        # find abs difference to nominal
        dest_dnom = np.abs(np.asarray(dest)-dnom)
        # sort ascending + select num best
        idx = dest_dnom.argsort()[:num]
        dest_best[var] = [(paths_list[ii],dest_dnom[ii]) for ii in idx]
    # plot Torque data
    f,ax = plt.subplots()
    for p,_ in dest_best['Torque']:
        data = dp.loadSetitecXls(p)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten()) 
        ydata = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
        ax.plot(xdata,ydata,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
    ax.legend()
    ax.set(xlabel="Position (mm)",ylabel="Torque (A)",title=f"Torque Files Closest to Nominal 32.0mm")
    f.savefig(f"best-nominal-files-torque.png")
    plt.close(f)

    # plot Thrust data
    f,ax = plt.subplots()
    for p,_ in dest_best['Thrust']:
        data = dp.loadSetitecXls(p)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten()) 
        ydata = data['I Thrust (A)'].values.flatten()
        ax.plot(xdata,ydata,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
    ax.legend()
    ax.set(xlabel="Position (mm)",ylabel="Thrust (A)",title=f"Thrust Files Closest to Nominal 32.0mm")
    f.savefig(f"best-nominal-files-thrust.png")

def kistler_find_outliers(path="AirbusData/Seti-Tec data files life test/*.xls",N=30,dnom=32.0,num=5,add_thrust_empty=False,use_signs=False,dlim=7.0):
    # create dictionary to hold depth estimate
    depth_est = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            #print(f"Processing {var}")
            # add entry for variable to depth estimate
            if not (var in depth_est):
                depth_est[var] = []
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()   
            # calculate the depth estimate
            dest = depth_est_rolling(ydata,xdata,NA=N,xendB = 0.0,xendA=30.0,use_signs=use_signs)
            depth_est[var].append(dest)
    paths_list = list(sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]))
    dest_best = {}
    for var,dest in depth_est.items():
        # find abs difference to nominal
        dest_dnom = np.abs(np.asarray(dest)-dnom)
        # find those whose distance from norm is greater than dlim
        idx = np.where(dest_dnom>=dlim)[0][:num]
        dest_best[var] = [(paths_list[ii],dest_dnom[ii]) for ii in idx]
    # plot Torque data
    f,ax = plt.subplots(nrows=2,sharex=True,constrained_layout=True)
    for p,_ in dest_best['Torque']:
        data = dp.loadSetitecXls(p)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten()) 
        ydata = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
        ax[0].plot(xdata,ydata,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
        grad = rolling_gradient(ydata,N,keep_signs=True)
        ax[1].plot(xdata,grad,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
    ax[0].legend()
    ax[1].legend()
    ax[1].set(xlabel="Position (mm)",ylabel="Rolling Gradient")
    ax[0].set(xlabel="Position (mm)",ylabel="Torque (A)")
    f.suptitle(f"Torque Files more than {dlim}mm from nom Nominal {dnom}mm")
    f.savefig(f"dnom-files-farther-than-dlim-{str(dlim).replace('.','-')}-torque.png")
    plt.close(f)

    # plot Thrust data
    f,ax = plt.subplots(nrows=2,sharex=True,constrained_layout=True)
    for p,_ in dest_best['Thrust']:
        data = dp.loadSetitecXls(p)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten()) 
        ydata = data['I Thrust (A)'].values.flatten() + data['I Thrust Empty (A)'].values.flatten()
        ax[0].plot(xdata,ydata,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
        grad = rolling_gradient(ydata,N,keep_signs=True)
        ax[1].plot(xdata,grad,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
    ax[0].legend()
    ax[1].legend()
    ax[1].set(xlabel="Position (mm)",ylabel="Rolling Gradient")
    ax[0].set(xlabel="Position (mm)",ylabel="Thrust (A)")
    f.suptitle(f"Thrust Files more than {dlim}mm from nom Nominal {dnom}mm")
    f.savefig(f"dnom-files-farther-than-dlim-{str(dlim).replace('.','-')}-thrust.png")
    plt.close(f)

def kistler_find_overest(path="AirbusData/Seti-Tec data files life test/*.xls",N=30,dnom=32.0,num=5,add_thrust_empty=False,use_signs=True):
    # create dictionary to hold depth estimate
    depth_est = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            #print(f"Processing {var}")
            # add entry for variable to depth estimate
            if not (var in depth_est):
                depth_est[var] = []
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()   
            # calculate the depth estimate
            try:
                dest = depth_est_rolling(ydata,xdata,NA=N,xendB = 0.0,xendA=30.0,use_signs=use_signs)
            except ValueError:
                print(f"Failed to process {fname} at N={N}! Setting dest to -1")
                dest = -1
            depth_est[var].append(dest)
    paths_list = list(sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]))

    dest_best = {}
    for var,dest in depth_est.items():
        # find abs difference to nominal
        dest_dnom = dnom - np.array(dest)
        # find those whose distance from norm is greater than dlim
        idx = np.where(dest_dnom<0)[0][:num]
        #print(var, np.where(dest_dnom<0))
        dest_best[var] = [(paths_list[ii],dest_dnom[ii]) for ii in idx]
    # plot Torque data
    f,ax = plt.subplots(nrows=2,sharex=True,constrained_layout=True)
    for p,_ in dest_best['Torque']:
        data = dp.loadSetitecXls(p)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten()) 
        ydata = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
        ax[0].plot(xdata,ydata,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
        grad = rolling_gradient(ydata,N,keep_signs=True)
        ax[1].plot(xdata,grad,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
    ax[0].legend()
    ax[1].legend()
    ax[1].set(xlabel="Position (mm)",ylabel="Rolling Gradient")
    ax[0].set(xlabel="Position (mm)",ylabel="Torque (A)")
    f.suptitle(f"Torque Files more than Nominal {dnom}mm")
    f.savefig(f"dnom-files-over-est-torque.png")
    plt.close(f)

    # plot Thrust data
    f,ax = plt.subplots(nrows=2,sharex=True,constrained_layout=True)
    for p,_ in dest_best['Thrust']:
        data = dp.loadSetitecXls(p)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten()) 
        ydata = data['I Thrust (A)'].values.flatten() + data['I Thrust Empty (A)'].values.flatten()
        ax[0].plot(xdata,ydata,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
        grad = rolling_gradient(ydata,N,keep_signs=True)
        ax[1].plot(xdata,grad,'-',label=os.path.splitext(os.path.basename(p))[0].split('_')[3])
    ax[0].legend()
    ax[1].legend()
    ax[1].set(xlabel="Position (mm)",ylabel="Rolling Gradient")
    ax[0].set(xlabel="Position (mm)",ylabel="Thrust (A)")
    f.suptitle(f"Thrust Files more than Nominal {dnom}mm")
    f.savefig(f"dnom-files-over-est-thrust.png")
    plt.close(f)

def kistler_rolling_variance(path="AirbusData/Seti-Tec data files life test/*.xls",add_thrust_empty=False,NN=[10,20,30,40,50]):
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            for N in NN:
                rvar = rolling_variance(ydata,N)
                f,ax = plt.subplots(constrained_layout=True)
                ax.plot(xdata,ydata,'b-')
                ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)",title=f"{fname}\n{var} Rolling Variance N={N}")
                cax = ax.twinx()
                cax.plot(xdata,rvar,'r-')
                cax.set_ylabel("Rolling Variance")
                f.savefig(f"{fname}-{var}-rolling-variance-N-{N}.png")
                plt.close(f)

def kistler_complete_variance(path="AirbusData/Seti-Tec data files life test/*.xls",add_thrust_empty=False):
    sigvar = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            if not (var in sigvar):
                sigvar[var] = []
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            sigvar[var].append(np.var(ydata))
    for var,svar in sigvar.items():
        f,ax = plt.subplots()
        ax.plot(svar)
        ax.set(xlabel="Hole Number",ylabel="Variance",title=f"{var} Complete Signal Variance")
        f.savefig(f"kistler-{var}-complete-variance.png")
        plt.close(f)

def kistler_complete_variance_and_energy(path="AirbusData/Seti-Tec data files life test/*.xls",add_thrust_empty=False):
    sigvar = {}
    energy = {}
    # iterte over each of the files
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            if not (var in sigvar):
                sigvar[var] = []
                energy[var] = []
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            sigvar[var].append(np.var(ydata))
            energy[var].append(np.sum(np.abs(ydata)**2.0))
    for var,svar in sigvar.items():
        f,ax = plt.subplots()
        ax.plot(svar,'b-')
        cax = ax.twinx()
        cax.plot(energy[var],'r-')
        cax.set_ylabel("Signal Energy")
        ax.set(xlabel="Hole Number",ylabel="Variance",title=f"{var} Complete Signal Variance")
        f.savefig(f"kistler-{var}-complete-variance-and-energy.png")
        plt.close(f)

def kistler_rolling_gradient_stats(path="AirbusData/Seti-Tec data files life test/*.xls",add_thrust_empty=False):
    stats = {}
    path="AirbusData/Seti-Tec data files life test/*.xls"
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var in ["Torque","Thrust"]:
            ydata = data[f'I {var} (A)'].values.flatten()
            if var == "Torque":
                ydata += data[f'I {var} Empty (A)'].values.flatten()
            grad = rolling_gradient(ydata,N=30,keep_signs=True)
            if not (var in stats):
                stats[var] = {'min': [],'max': [],'mean': [],'var': []}
            stats[var]['min'].append(grad.min())
            stats[var]['max'].append(grad.max())
            stats[var]['mean'].append(grad.mean())
            stats[var]['var'].append(grad.var())
    for var,st in stats.items():
        f,ax = plt.subplots(constrained_layout=True)
        for mm,dd in st.items():
            ax.plot(dd,label=mm)
        ax.set(xlabel="Hole Number",ylabel="Rolling Gradient",title=f"Kistler {var} Rolling Gradient Stats")
        ax.legend()
        f.savefig(f"kistler-{var}-stats-rolling-gradient.png")
        plt.close(f)

def kistler_first_and_last_peaks(path="AirbusData/Seti-Tec data files life test/*.xls"):
    xstart = {'first': [],'last': [], 'max': []}
    xend = {'first': [],'last': [], 'max': []}
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        ydata = data[f'I Torque (A)'].values.flatten() +data[f'I Torque Empty (A)'].values.flatten()
        A,B = depth_est_rolling_pks(ydata,xdata,NA=30,xendB = 0.0,xendA=25.0,use_signs=True,pselect='max')
        xstart['max'].append(A)
        xend['max'].append(B)
        A,B = depth_est_rolling_pks(ydata,xdata,NA=30,xendB = 0.0,xendA=25.0,use_signs=True,pselect='first')
        xstart['first'].append(A)
        xend['first'].append(B)
        A,B = depth_est_rolling_pks(ydata,xdata,NA=30,xendB = 0.0,xendA=25.0,use_signs=True,pselect='last')
        xstart['last'].append(A)
        xend['last'].append(B)

    f,ax = plt.subplots()
    for k,v in xstart.items():
        ax.plot(v,'x',label=k)
    ax.set(xlabel="Hole Number",ylabel="X-Position (mm)",title="Kistler 1st Position Location Using Methods")
    ax.legend()
    f.savefig("kistler-first-position-methods.png")

    f,ax = plt.subplots()
    for k,v in xend.items():
        ax.plot(v,'x',label=k)
    ax.set(xlabel="Hole Number",ylabel="X-Position (mm)",title="Kistler 2nd Position Location Using Methods")
    ax.legend()
    f.savefig("kistler-end-position-methods.png")

    f,ax = plt.subplots()
    for k in xstart.keys():
        ax.plot(np.array(xend[k])-np.array(xstart[k]),'x',label=k)
    ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title="Kistler Depth Estimate Using Methods")
    ax.legend()
    f.savefig("kistler-depth-est-methods.png")

def kister_plot_torque(path="AirbusData/Seti-Tec data files life test/*.xls"):
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        ydata = data[f'I Torque (A)'].values.flatten() +data[f'I Torque Empty (A)'].values.flatten()
        f,ax = plt.subplots(constrained_layout=True)
        ax.plot(xdata,ydata,'b-')
        filt_weight = wiener(ydata,30)
        grad = rolling_gradient(filt_weight,N=30,keep_signs=True)
        ax.plot(xdata,filt_weight,'r-')
        cax = ax.twinx()
        cax.plot(xdata,grad,'k-')
        pks,_ = find_peaks(grad)
        cax.plot(xdata[pks],grad[pks],'gx')

        ax.set(xlabel="Position (mm)",ylabel=f"Torque (A)",title=fname)
        cax.set_ylabel("Rolling Gradient")
        plt.show()

def _depthest(x,*args):
    depth_est = []
    for fn in PATHS:
        data = dp.loadSetitecXls(fn)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten())
        ydata = data['I Torque (A)'].values.flatten()
        try:
            depth_est.append(depth_est_rolling(ydata,xdata,NA=20,xstart=10.0,depth_exp=x[0],depth_win=x[1],default=False,end_ref='end',pselect='argmax'))
        except MissingPeaksException:
            depth_est.append(40.0)
    return abs(32.2-np.mean(depth_est))

def _depthestfull(x,*args):
    depth_est = []
    for fn in PATHS:
        data = dp.loadSetitecXls(fn)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten())
        ydata = data['I Torque (A)'].values.flatten()
        try:
            depth_est.append(depth_est_rolling(ydata,xdata,NA=int(x[2]),xstart=10.0,depth_exp=x[0],depth_win=x[1],default=False,end_ref='end',pselect='argmax'))
        except MissingPeaksException:
            depth_est.append(40.0)
    return ((32.2-np.mean(depth_est))**2.0 +(0.1-np.std(depth_est)**2.0))**0.5

def _depthestjustwindow(x,*args):
    depth_est = []
    for fn in PATHS:
        data = dp.loadSetitecXls(fn)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten())
        ydata = data['I Torque (A)'].values.flatten()
        try:
            depth_est.append(depth_est_rolling(ydata,xdata,NA=20,xstart=10.0,depth_exp=args[0],depth_win=x[0],default=True,end_ref='end',pselect='argmax'))
        except MissingPeaksException:
            depth_est.append(40.0)
    return ((32.2-np.mean(depth_est))**2.0 +(0.1-np.std(depth_est)**2.0))**0.5

def _depthestjustexpected(x,*args):
    depth_est = []
    for fn in PATHS:
        data = dp.loadSetitecXls(fn)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten())
        ydata = data['I Torque (A)'].values.flatten()
        try:
            depth_est.append(depth_est_rolling(ydata,xdata,NA=20,xstart=10.0,depth_exp=x[0],depth_win=args[0],default=False,end_ref='end',pselect='argmax'))
        except MissingPeaksException:
            depth_est.append(40.0)
    return abs(32.2-np.mean(depth_est))

def depth_est_optimize_mean_std(mode='all'):
    from scipy.optimize import minimize
    if mode == 'all':
        xopts = minimize(_depthest,x0=[25.0,1.0],method='Nelder-Mead',bounds=[(25.0,35.0),(1.0,5.0)],options={'maxiter':20000})
    elif mode == 'exp':
        xopts = minimize(_depthestjustexpected,x0=[25.0],args=(3.0),method='Nelder-Mead',bounds=[(25.0,35.0)],options={'maxiter':20000})
    elif mode == 'win':
        xopts = minimize(_depthestjustwindow,x0=[1.0],args=(25.0),method='Nelder-Mead',bounds=[(1.0,5.0)],options={'maxiter':20000})
    elif mode == 'full':
        xopts = minimize(_depthestfull,x0=[25.0,1.0,20],method='Nelder-Mead',bounds=[(25.0,35.0),(1.0,5.0),(10,100)],options={'maxiter':20000})
    return xopts

def plot_json_cfact(folder="",mean=31.0,win_size=5.0,cmap="binary"):
    from matplotlib import cm
    
    search = os.path.join(folder,f"depth-estimate-stats-exp-{mean:.1f}-window-{win_size:.1f}*.json")
    paths = glob(search)
    if len(paths)==0:
        raise ValueError(f"Failed to find any JSONS using {search}!")
    if len(paths)<3:
        raise ValueError(f"Fewer than 3 JSONS. Not a very interesting plot for {mean} {win_size}")
    if not (cmap is None):
        cmap = cm.get_cmap(cmap)
        nf = len(paths)
    # get the window sizes used
    # same for all files as they're generated in batches
    data = json.load(open(paths[0],'r')) 
    NN = data['NN']
    # create axes
    f,ax = plt.subplots(nrows=2,ncols=3,figsize=(12,10))
    cf_plotted = set()
    # sort by correction factor
    for fi,path in enumerate(sorted(paths,key=lambda x : float(os.path.splitext(x)[0].split('-')[-1])),1):
        # load data
        data = json.load(open(path,'r'))
        # get correction factor
        cf = os.path.splitext(os.path.basename(path))[0].split('-')[-1]
        if (float(cf)>1) or ('cfact' not in path):
            print(f"Skipping {path} as the CF is > 1 or doesn't contain cfact in path")
            continue
        # plot data labelling it using cf
        if cmap is None:
            ax[0,0].plot(NN,data['mean']['Torque'],label=cf if not (cf in cf_plotted) else "")
            ax[1,0].plot(NN,data['mean']['Thrust'],label=cf if not (cf in cf_plotted) else "")
            ax[0,1].plot(NN,data['var']['Torque'],label=cf if not (cf in cf_plotted) else "")
            ax[1,1].plot(NN,data['var']['Thrust'],label=cf if not (cf in cf_plotted) else "")
            if 'std' in data:
                ax[0,2].plot(NN,data['std']['Torque'],label=cf if not (cf in cf_plotted) else "")
                ax[1,2].plot(NN,data['std']['Thrust'],label=cf if not (cf in cf_plotted) else "")
            else:
                ax[0,2].plot(NN,[np.std(x) for x in data['depth_est']['Torque']],label=cf if not (cf in cf_plotted) else "")
                ax[1,2].plot(NN,[np.std(x) for x in data['depth_est']['Thrust']],label=cf if not (cf in cf_plotted) else "")
        else:
            ax[0,0].plot(NN,data['mean']['Torque'],c=cmap(fi/nf),label=cf if not (cf in cf_plotted) else "")
            ax[1,0].plot(NN,data['mean']['Thrust'],c=cmap(fi/nf),label=cf if not (cf in cf_plotted) else "")
            ax[0,1].plot(NN,data['var']['Torque'],c=cmap(fi/nf),label=cf if not (cf in cf_plotted) else "")
            ax[1,1].plot(NN,data['var']['Thrust'],c=cmap(fi/nf),label=cf if not (cf in cf_plotted) else "")
            if 'std' in data:
                ax[0,2].plot(NN,data['std']['Torque'],c=cmap(fi/nf),label=cf if not (cf in cf_plotted) else "")
                ax[1,2].plot(NN,data['std']['Thrust'],c=cmap(fi/nf),label=cf if not (cf in cf_plotted) else "")
            else:
                ax[0,2].plot(NN,[np.std(x) for x in data['depth_est']['Torque']],c=cmap(fi/nf),label=cf if not (cf in cf_plotted) else "")
                ax[1,2].plot(NN,[np.std(x) for x in data['depth_est']['Thrust']],c=cmap(fi/nf),label=cf if not (cf in cf_plotted) else "")
        cf_plotted.add(cf)
    f.legend()
    ax[0,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Torque)")
    ax[0,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Torque)")
    ax[0,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Torque)")
    ax[1,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Thrust)")
    ax[1,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Thrust)")
    ax[1,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Thrust)")

    f.suptitle(f"Stats about Depth Est. using depth_exp={mean}, window={win_size}")
    return f

def plot_json_depthest_compare(folder="",mean=31.0,win_size=5.0):
    # find all corrected jsons
    search =os.path.join(folder, f"depth-estimate-stats-exp-{mean:.1f}-window-{win_size:.1f}*.json")
    paths = glob(search)
    if len(paths)==0:
        raise ValueError(f"Failed to find any JSONS using {search}!")
    # find base json
    base_search = f"depth-estimate-stats-exp-{mean:.1f}-window-{win_size:.1f}.json"
    base_path = glob(search)
    if len(base_path)==0:
        raise ValueError(f"Failed to find base JSONS using {base_search}!")
    base_path = base_path[0]
    data = json.load(open(base_path,'r'))
    base_depth_est = data['depth_est']['Torque']
    # get the window sizes used
    # same for all files as they're generated in batches
    data = json.load(open(paths[0],'r')) 
    NN = data['NN']
    for fi,path in enumerate(sorted(paths,key=lambda x : float(os.path.splitext(x)[0].split('-')[-1])),1):
        # load data
        data = json.load(open(path,'r'))
        # get correction factor
        cf = os.path.splitext(os.path.basename(path))[0].split('-')[-1]
        if (float(cf)>1) or ('cfact' not in path):
            print(f"Skipping {path} as the CF is > 1 or does not contain cfact in path")
            continue
        for N,dest,base in zip(NN,data['depth_est']['Torque'],base_depth_est):
            f,ax = plt.subplots()
            ax.plot(base,'bx',label="Base")
            ax.plot(dest,'rx',label="Corrected")
            ax.legend()
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)")
            f.suptitle(f"Depth Estimate Compare Expected {mean}, Window {win_size},c={cf}, N={N}")
            f.savefig(f"depth-estimate-corrected-compare-N-{N}-exp-{mean}-window-{win_size}-cfact-{cf}.png")
            plt.close(f)

def plot_all_json_cfact(search="*.json"):
    paths = glob(search)
    if len(paths)==0:
        raise ValueError(f"No JSON files found in {search}!")
    # find unique mean values
    means = set()
    for x in paths:
        try:
            pts = float(os.path.splitext(os.path.basename(x))[0].split('-')[4])
        except (ValueError,IndexError):
            continue
        means.add(pts)
    # find unique window values
    wins = set()
    for x in paths:
        try:
            pts = float(os.path.splitext(os.path.basename(x))[0].split('-')[6])
        except (ValueError,IndexError):
            continue
        wins.add(pts)
    print(means,wins)
    for mn in means:
        for w in wins:
            try:
                f = plot_json_cfact(os.path.dirname(search),mn,w)
            except ValueError:
                continue
            #f = plot_json_cfact(os.path.dirname(search),mn,w)
            f.savefig(f"depth-estimate-stats-exp-{mn}-window-{w}-all-cfact.png")
            plt.close(f)

            try:
                plot_json_depthest_compare(os.path.dirname(search),mn,w)
            except ValueError:
                continue

def plot_peak_widths(path,N=20,height=0.7,units="index"):
    data = dp.loadSetitecXls(path)[-1]
    xdata = np.abs(data['Position (mm)'].values.flatten())
    ydata = data['I Torque (A)'].values.flatten()
    
    filt_weight = wiener(ydata,N)
    # perform rolling gradient on smoothed data
    grad = np.abs(rolling_gradient(filt_weight,N,keep_signs=True))
    win = tukey(len(grad),0.1,True)
    grad *= win
    hlim = grad.max()
    # find peaks ignoring those below the target threshold
    f,ax = plt.subplots()
    pks,_ = find_peaks(grad, height=0.1*hlim)
    xpks = xdata[pks]
    widths = peak_widths(grad,pks,height)
    ax.set_xlabel("Peak Location (mm)")
    if units=="index":
        ax.plot(xpks,widths[0],'x')
        ax.set(ylabel="Width (Index")
    elif units=="dist":
        ax.plot(xpks,[abs(xdata[int(A)]-xdata[int(B)]) for A,B in zip(widths[2],widths[3])],'x')
        ax.set(ylabel="Width (mm)")
    f.suptitle(f"Peak Widths for {os.path.splitext(os.path.basename(path))[0]}\nwhere width is {units.capitalize()} and height={height}")
    return f

def plot_all_peak_widths(path,N=20,height=0.7,units="index",cmap="binary"):
    from matplotlib import cm
    cmap = cm.get_cmap(cmap)
    paths = glob(path)
    nf = len(paths)
    f,ax = plt.subplots()
    for fi,fn in enumerate(sorted(paths,key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]),1):
        data = dp.loadSetitecXls(fn)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten())
        ydata = data['I Torque (A)'].values.flatten()
        
        filt_weight = wiener(ydata,N)
        # perform rolling gradient on smoothed data
        grad = np.abs(rolling_gradient(filt_weight,N,keep_signs=True))
        win = tukey(len(grad),0.1,True)
        grad *= win
        hlim = grad.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad, height=0.1*hlim)
        xpks = xdata[pks]
        widths = peak_widths(grad,pks,height)
        
        if units=="index":
            ax.plot(xpks,widths[0],'x',label=os.path.splitext(os.path.basename(fn))[0].split('_')[3],c=cmap(fi/nf))
        elif units=="dist":
            ax.plot(xpks,[abs(xdata[int(A)]-xdata[int(B)]) for A,B in zip(widths[2],widths[3])],'x',label=os.path.splitext(os.path.basename(fn))[0].split('_')[3],c=cmap(fi/nf))
    f.suptitle(f"Airbus peak widths where width is {units.capitalize()} and height={height}")
    ax.set_xlabel("Peak Location (mm)")
    #ax.legend(loc="center right")
    if units=="index":
        ax.set(ylabel="Width (Index)")
    elif units=="dist":
        ax.set(ylabel="Width (mm)")
    return f

def plot_peak_widths_stats(path,N=20,height=0.7,units="index"):
    from matplotlib import cm
    paths = glob(path)
    nf = len(paths)
    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    for fi,fn in enumerate(sorted(paths,key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3])):
        data = dp.loadSetitecXls(fn)[-1]
        xdata = np.abs(data['Position (mm)'].values.flatten())
        ydata = data['I Torque (A)'].values.flatten()
        
        filt_weight = wiener(ydata,N)
        # perform rolling gradient on smoothed data
        grad = np.abs(rolling_gradient(filt_weight,N,keep_signs=True))
        win = tukey(len(grad),0.1,True)
        grad *= win
        hlim = grad.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad, height=0.1*hlim)
        xpks = xdata[pks]
        widths = peak_widths(grad,pks,height)
        
        if units=="index":
            widths = widths[0]
            ax[0].plot(fi,np.min(widths),'bx')
            ax[1].plot(fi,np.max(widths),'rx')
            ax[2].plot(fi,np.mean(widths),'kx')
        elif units=="dist":
            widths = [abs(xdata[int(A)]-xdata[int(B)]) for A,B in zip(widths[2],widths[3])]
            ax[0].plot(fi,np.min(widths),'bx')
            ax[1].plot(fi,np.max(widths),'rx')
            ax[2].plot(fi,np.mean(widths),'kx')
    ax[0].set(xlabel="Hole Number",ylabel=f"Min Peak Width {units.capitalize()}",title=f"Min Peak Width {units.capitalize()}")
    ax[1].set(xlabel="Hole Number",ylabel=f"Max Peak Width {units.capitalize()}",title=f"Max Peak Width {units.capitalize()}")
    ax[2].set(xlabel="Hole Number",ylabel=f"Mean Peak Width {units.capitalize()}",title=f"Meann Peak Width {units.capitalize()}")
    f.suptitle(f"Peak Widths Statistics")
    return f

def search_jsons(target_mean=32.2,target_std=0.1):
    for fn in glob("*.json"):
        targets = []
    for fn in glob("*.json"):
        data = json.load(open(fn,'r'))
        if not ('NN' in data):
            continue
        means = data['mean']['Torque']
        NN = data['NN']
        tg = {}
        for mi,m in enumerate(means):
            if (m>(target_mean-0.2)) and (m<(target_mean+0.2)):
                if not (fn in tg):
                    tg[fn] = []
                tg[fn].append(NN[mi])
        if tg:
            targets.append(tg)

def calc_depth(path,N,cf=None,depth_exp=32.0,win=3.0):
    data = dp.loadSetitecXls(path)[-1]
    xdata = np.abs(data['Position (mm)'].values.flatten())
    ydata = data['I Torque (A)'].values.flatten()
    if not (cf is None):
        return depth_est_rolling(ydata,xdata,NA=N,xstart=10.0,depth_exp=depth_exp,depth_win=win,default=True,fact=cf)
    else:
        return depth_est_rolling(ydata,xdata,NA=N,xstart=10.0,depth_exp=depth_exp,depth_win=win,default=True)

def mt_depth_eval(N,cfact=None,dexp=32.0,win=3.0):
    from multiprocessing import Pool
    return Pool(5).starmap(calc_depth,[(x,N,cfact,dexp,win) for x in PATHS])

def correction_factor_gui():
    from matplotlib.widgets import Slider, Button

    f,ax = plt.subplots(ncols=2)
    ax[0].set(xlabel="Hole Number",ylabel="Depth Estimate (mm)")
    ax[1].set(xlabel="Hole Number",ylabel="Depth Estimate (mm)")
    plt.subplots_adjust(left=0.25, bottom=0.25)
    # define the base depth estimate
    axcf = plt.axes([0.25, 0.15, 0.65, 0.03])
    cf_slider = Slider(
        ax=axcf,
        label='Correction Slider',
        valmin=0.01,
        valmax=0.99,
        valstep=0.01,
        valinit=0.80,
    )

    axwin = plt.axes([0.25, 0.1, 0.65, 0.03])
    win_slider = Slider(
        ax=axwin,
        label='Window Size',
        valmin=10,
        valmax=50,
        valstep=10,
        valinit=20,
    )

    depthest = mt_depth_eval(int(win_slider.val))
    mind = min(depthest)
    maxd = max(depthest)
    IDX = list(range(len(depthest)))
    ax[0].scatter(IDX,depthest,c='b',label="Base")
    scat = ax[1].scatter(IDX,depthest,c='r',label="Corrected")
    #ax.legend()

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Calculate', hovercolor='0.975')
    
    def update(event):
        print("crunching")
        dest = mt_depth_eval(int(win_slider.val),cf_slider.val)
        print("updating")
        scat.set_offsets(np.vstack((IDX,dest)).T)
        f.canvas.draw_idle()
        print("adjusting y axes")
        #ax[1].set_ylim(min([min(dest),mind]),max([max(dest),maxd]))
        
    button.on_clicked(update)
    plt.show()

def kistler_depth_run_rpca(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],add_thrust_empty=False,xstart=10.0,end_ref='end',depth_exp=32.0,depth_win=4.0,default=False,pselect='argmax',ref_line=None):
    from r_pca import R_pca
    # create dictionary to hold depth estimate
    depth_est = {}
    # lists of data vectors
    dt_torque = []
    dt_thrust = []
    # list of pos
    pos = []
    # number of files
    nf = len(PATHS)
    # minimum length
    min_len = None
    print("Creating data files")
    # iterte over each of the files
    for fn in glob(path):
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        pos.append(xdata)
        if min_len is None:
            min_len = xdata.shape[0]
        else:
            min_len = min([min_len,xdata.shape[0]])
        # iterate over the variables
        for var,dt in zip(["Torque","Thrust"],[dt_torque,dt_thrust]):
            #print(f"Processing {var}")
            # add entry for variable to depth estimate
            if not (var in depth_est):
                depth_est[var] = {}
            # if adding empty to thrust
            # set y data to normal + empty
            if add_thrust_empty:
                ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # if just adding empty to torque
            else:
                ydata = data[f'I {var} (A)'].values.flatten()
                if var == "Torque":
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            dt.append(ydata)
    # clip arrays
    print(f"clipping arrays to {min_len}")
    for i in range(nf):
        if dt_torque[i].shape[0]>min_len:
            dt_torque[i] = dt_torque[i][:min_len]
            dt_thrust[i] = dt_torque[i][:min_len]
            pos[i] = pos[i][:min_len]    
    # stack them to form a 2D matrix
    dt_torque_arr = np.column_stack(dt_torque)
    dt_thrust_arr = np.column_stack(dt_thrust)
    # process using RPCA
    print("processing using RPCA")
    L_torque,S_torque = R_pca(dt_torque_arr).fit(max_iter=10000)
    L_thrust,S_thrust = R_pca(dt_thrust_arr).fit(max_iter=10000)
    print(f"finished ",L_torque.shape,L_thrust.shape)
    # for a given variable
    for var,L in zip(["Torque","Thrust"],[L_torque,L_thrust]):
        # iterate over each column
        for i in range(nf):
            # get data vectors
            xdata = pos[i]
            ydata = L[:,i]
            # iterate over different window sizes
            for N in NN:
                # estimate depth
                dest = depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect)
                # save result in dictionary
                if not (N in depth_est[var]):
                    depth_est[var][N] = []
                depth_est[var][N].append(dest)
    # create axes for statistics    
    fstats,axstats = plt.subplots(nrows=2,ncols=3,sharex=True,constrained_layout=True,figsize=(14,12))
    # create lists to process
    thrust_mean = []
    thrust_var = []
    thrust_std = []
    torque_mean = []
    torque_var = []
    torque_std = []
    # define JSON to ave
    data = {'NN':NN,'depth_est' : {'Thrust' : [], 'Torque': []}, 'mean' : {'Thrust' : [], 'Torque': []}, 'var' : {'Thrust' : [], 'Torque': []}, 'std' : {'Thrust' : [], 'Torque': []}}
    # iterate over each window size    
    for N in NN:
        # create axes
        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Torque"][N],'bx',label="Torque")
        torque_mean.append(np.mean(depth_est["Torque"][N]))
        torque_var.append(np.var(depth_est["Torque"][N]))
        torque_std.append(np.std(depth_est["Torque"][N]))
        # plot thrust depth estimate with red X's
        ax.plot(depth_est["Thrust"][N],'rx',label="Thrust")
        thrust_mean.append(np.mean(depth_est["Thrust"][N]))
        thrust_var.append(np.var(depth_est["Thrust"][N]))
        thrust_std.append(np.std(depth_est["Thrust"][N]))
        # draw a black line for nominal depth
        if not (ref_line is None):
            ax.plot(nf*[ref_line,],'k-',label="Nominal")
        # create legend
        ax.legend()
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"RPCA Depth Estimate N={N} with Retry")
        # save figure
        f.savefig(f"depth-estimate-kestler-all-var-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-rpca.png")
        plt.close(f)

        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Torque"][N],'bx')
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"RPCA Depth Estimate (Torque) N={N} with Retry")
        f.savefig(f"depth-estimate-kestler-all-var-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-rpca-torque-only.png")
        plt.close(f)

        f,ax = plt.subplots(constrained_layout=True)
        # plot torque depth estimate with blue X's
        ax.plot(depth_est["Thrust"][N],'rx')
        ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"RPCA Depth Estimate (Thrust) N={N} with Retry")
        f.savefig(f"depth-estimate-kestler-all-var-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-rpca-thrust-only.png")
        plt.close(f)

        # process and add results to the JSON
        data['depth_est']['Torque'].append([float(x) for x in depth_est["Torque"][N]])
        data['depth_est']['Thrust'].append([float(x) for x in depth_est["Thrust"][N]])
        
        data['mean']['Torque'].append(float(torque_mean[-1]))
        data['var']['Torque'].append(float(torque_var[-1]))
        data['std']['Torque'].append(float(torque_std[-1]))
        
        data['mean']['Thrust'].append(float(thrust_mean[-1]))
        data['var']['Thrust'].append(float(thrust_var[-1]))
        data['std']['Thrust'].append(float(thrust_std[-1]))
    # save data
    json.dump(data,open(f"depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-rpca.json",'w'))

    # plot the statistics of each window size
    axstats[0,0].plot(NN,torque_mean,'b-')
    axstats[0,1].plot(NN,torque_var,'r-')
    axstats[0,2].plot(NN,torque_std,'k-')
    axstats[1,0].plot(NN,thrust_mean,'b-')
    axstats[1,1].plot(NN,thrust_var,'r-')
    axstats[1,2].plot(NN,thrust_std,'k-')
    
    axstats[0,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Torque)")
    axstats[0,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Torque)")
    axstats[0,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Torque)")
    axstats[1,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Thrust)")
    axstats[1,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Thrust)")
    axstats[1,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Thrust)")
    fstats.suptitle(f"Stats about RPCA Depth Est. using depth_exp={depth_exp}, window={depth_win}")
    fstats.savefig(f"depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-rpca.png")
    return depth_est

def compare_grad():
    def calc_grad(fn,N,v="1"):
        data = dp.loadSetitecXls(fn)[-1]
        y = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
        if v=="1":
            return rolling_gradient(y,N)
        elif v=="2":
            return rolling_gradient_v2(y,N)
    grad = []
    grad_2 = []
    for N in [10,20,30,40,50]:
        f,ax = plt.subplots(ncols=2)
        for fn in PATHS:
            ax[0].plot(calc_grad(fn,N))
            ax[1].plot(calc_grad(fn,N,"2"))
        ax[0].set(xlabel="Index",ylabel="Rolling Gradient",title="Current")
        ax[1].set(xlabel="Index",ylabel="Rolling Gradient",title="New")
        f.suptitle(f"Comparison of Rolling Gradients under Different Algorithms\nN={N}")
        f.savefig(f"rolling-grad-compare-N-{N}.png")
        plt.close(f)

def load_all_files(path=PATHS,form='parts'):
    if form == 'parts':
        return [dp.loadSetitecXls(fn)[-1] for fn in path]
    elif form == 'whole':
        frames = []
        for fn in path:
            data = dp.loadSetitecXls(fn)[-1]
            data['HOLENUM'] = int(os.path.splitext(os.path.basename(fn))[0].split('_')[3])
            frames.append(data)
        return pd.concat(frames)

def windowPeakWarnings(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],xstart=10.0,end_ref='end',depth_exp=40.0,depth_win=5.0,default=True,pselect='argmax',opath=''):
    import warnings
    warnings.filterwarnings("error")
    counts_torque = {N : 0 for N in NN}
    counts_thrust = {N : 0 for N in NN}
    for fn in sorted(glob(path),key=lambda x : os.path.splitext(os.path.basename(x))[0].split('_')[3]):
        fname = os.path.splitext(os.path.basename(fn))[0]
        #print(f"Processing {fname}")
        # load data file
        data = dp.loadSetitecXls(fn)[-1]
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        # iterate over the variables
        for var,counts in zip(["Torque","Thrust"],[counts_torque,counts_thrust]):
            ydata = data[f'I {var} (A)'].values.flatten() + data[f'I {var} Empty (A)'].values.flatten()
            # for a given window size
            for N in NN:
                #print(f"Processing win size {N}")
                # calculate the depth estimate
                #print("expected depth",depth_exp)
                try:
                    depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect)
                except MissingPeaksWarning:
                    counts[N] += 1
                except:
                    continue
    figs = []
    f,ax = plt.subplots()
    ax.bar(NN,counts_torque.values(),width=10,linewidth=4,edgecolor='k')
    ax.set(xlabel="Window Size",ylabel="Number of Warnings",title="Missing Peak Warnings vs Window Size (Torque)")
    f.suptitle(os.path.dirname(path) + f" {len(glob(path))} files")
    if opath:
        f.savefig(os.path.join(opath,f"{os.path.dirname(path).split('/')[-1]}-depth-peakwarning-count-torque.png"))
        plt.close(f)
    else:
        figs.append(f)
    
    f,ax = plt.subplots()
    ax.bar(NN,counts_thrust.values(),width=10,linewidth=4,edgecolor='k')
    ax.set(xlabel="Window Size",ylabel="Number of Warnings",title="Missing Peak Warnings vs Window Size (Thrust)")
    f.suptitle(os.path.dirname(path) + f" {len(glob(path))} files")
    if opath:
        f.savefig(os.path.join(opath,f"{os.path.dirname(path).split('/')[-1]}-depth-peakwarning-count-thrust.png"))
        plt.close(f)
    else:
        figs.append(f)
    return figs

def kistler_depth_run_dtw(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],add_empty=False,xstart=10.0,end_ref='end',depth_exp=32.0,depth_win=4.0,default=False,pselect='argmax',ref_line=None,tool_dims=[],opath=''): 
    from modelling import depth_est_rolling as depth_est_rolling_test, R_pca
    from time import perf_counter
    # create dictionary to hold depth estimate for different types of data
    depth_est = {'rpca':{},'rpca_dtw':{},'normal':{},'dtw':{}}
    eval_time = {'rpca':{},'rpca_dtw':{},'normal':{},'dtw':{}}
    # lists of data vectors
    dt_torque = []
    dt_thrust = []
    # list of pos
    pos = []
    # list of AVs
    av = []
    # number of files
    nf = len(glob(path))
    per = 0.6*math.fsum(tool_dims)
    # minimum length
    min_len = None
    print("Creating data files")
    # iterte over each of the files
    for fn in glob(path):
        # load data file
        data_all = dp.loadSetitecXls(fn,"auto")
        # get feed rate
        av.append(dp.getAV(data_all)[-1])
        # get run data
        data = data_all.pop(-1)
        # delete everything else
        del data_all
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        pos.append(xdata)
        # update clipping length
        min_len = xdata.shape[0] if min_len is None else min([min_len,xdata.shape[0]])
        # iterate over the variables
        for var,dt in zip(["Torque","Thrust"],[dt_torque,dt_thrust]):
            # add entry for variable to depth estimate dict
            for kk in depth_est.keys():
                depth_est[kk][var] = {}
                eval_time[kk][var] = {}
            # get base data
            ydata = data[f'I {var} (A)'].values.flatten()
            # add empty
            if add_empty:
                if f'I {var} Empty (A)' in data:
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            # add to array
            dt.append(ydata)
    # make copies of arrays
    dt_torque_clip = dt_torque.copy()
    dt_thrust_clip = dt_thrust.copy()
    pos_clip = pos.copy()
    # clip arrays
    print(f"clipping arrays to {min_len}")
    for i in range(nf):
        if dt_torque_clip[i].shape[0]>min_len:
            dt_torque_clip[i] = dt_torque_clip[i][:min_len]
            dt_thrust_clip[i] = dt_thrust_clip[i][:min_len]
            pos_clip[i] = pos_clip[i][:min_len]    
    # stack them to form a 2D matrix
    dt_torque_arr = np.column_stack(dt_torque_clip)
    dt_thrust_arr = np.column_stack(dt_thrust_clip)
    # process using RPCA
    print("processing using RPCA")
    L_torque,_ = R_pca(dt_torque_arr).fit(max_iter=10000)
    L_thrust,_ = R_pca(dt_thrust_arr).fit(max_iter=10000)
    print(f"finished ",L_torque.shape,L_thrust.shape)
    # for a given variable
    for var,L in zip(["Torque","Thrust"],[L_torque,L_thrust]):
        # iterate over each signal
        for i in range(nf):
            # get data vectors
            xdata = pos_clip[i]
            ydata = L[:,i]
            # update data to unclipped, original data
            ogxdata = pos[i]
            ogydata = dt_torque[i] if var == 'Torque' else dt_thrust[i]
            # iterate over different window sizes
            for N in NN:
                #print(f"Processing {var} RPCA {N}")
                # estimate depth using RPCA data
                start = perf_counter()
                dest = depth_est_rolling_test(ydata,xdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect)
                end = perf_counter() - start
                # save result in dictionary
                if not (N in depth_est['rpca'][var]):
                    depth_est['rpca'][var][N] = []
                    eval_time['rpca'][var][N] = []
                depth_est['rpca'][var][N].append(dest)
                eval_time['rpca'][var][N].append(end)
                # calculate attempting to use DTW on RPCA data
                start = perf_counter()
                dest = depth_est_rolling_test(ydata,xdata,NA=N,xstart=xstart,try_dtw=True,tool_dims=tool_dims,feed_rate=av[i],per=per,default=default,end_red=end_ref,pselect=pselect)
                end = perf_counter() - start
                # save result in dictionary
                if not (N in depth_est['rpca_dtw'][var]):
                    depth_est['rpca_dtw'][var][N] = []
                    eval_time['rpca_dtw'][var][N] = []
                depth_est['rpca_dtw'][var][N].append(dest)
                eval_time['rpca_dtw'][var][N].append(end)
                #print(f"Processing {var} NORMAL {N}")
                # estimate depth using ORIGINAL data
                start = perf_counter()
                dest = depth_est_rolling_test(ogydata,ogxdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect)
                end = perf_counter() - start
                # save result in dictionary
                if not (N in depth_est['normal'][var]):
                    depth_est['normal'][var][N] = []
                    eval_time['normal'][var][N] = []
                depth_est['normal'][var][N].append(dest)
                eval_time['normal'][var][N].append(end)
                # calculate attempting to use DTW on ORIGINAL data
                start = perf_counter()
                dest = depth_est_rolling_test(ogydata,ogxdata,NA=N,xstart=xstart,try_dtw=True,tool_dims=tool_dims,feed_rate=av[i],per=per,default=default,end_red=end_ref,pselect=pselect)
                end = perf_counter() - start
                # save result in dictionary
                if not (N in depth_est['dtw'][var]):
                    depth_est['dtw'][var][N] = []
                    eval_time['dtw'][var][N] = []
                depth_est['dtw'][var][N].append(dest)
                eval_time['dtw'][var][N].append(end)
    # iterate over data type
    for dtype,vals in depth_est.items():
        print(dtype)
        # iterate over variable and window sizes
        for key,wins in vals.items():
            print(f"\t{key}")
            for N,dest in wins.items():
                print(f"\t\t{N}:{len(dest)}")
    json.dump(depth_est,open(os.path.join(opath,f"depth-estimates-dtw-full-exp-{depth_exp}-window-{depth_win}-av-auto-per-{per:.2f}.json"),'w'),default=str)
    json.dump(eval_time,open(os.path.join(opath,f"depth-estimates-dtw-full-eval-time-exp-{depth_exp}-window-{depth_win}-av-auto-per-{per:.2f}.json"),'w'),default=str)
    # create lists to process
    thrust_mean = []
    thrust_var = []
    thrust_std = []
    torque_mean = []
    torque_var = []
    torque_std = []
    # define JSON to save
    data = {kk : {'NN':NN,'depth_est' : {'Thrust' : [], 'Torque': []}, 'mean' : {'Thrust' : [], 'Torque': []}, 'var' : {'Thrust' : [], 'Torque': []}, 'std' : {'Thrust' : [], 'Torque': []}} for kk in depth_est.keys()}
    # iterate over each type of estimate
    for kk,dest in depth_est.items():
        print(f"Plotting {kk} results")
        # iterate over each window size
        for N in NN:
            # get torque and thrust depth estimates for window size
            dest_torque = dest["Torque"][N]
            dest_thrust = dest["Thrust"][N]
            # create axes
            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_torque,'bx',label="Torque")
            torque_mean.append(np.mean(dest_torque))
            torque_var.append(np.var(dest_torque))
            torque_std.append(np.std(dest_torque))
            # plot thrust depth estimate with red X's
            ax.plot(dest_thrust,'rx',label="Thrust")
            thrust_mean.append(np.mean(dest_thrust))
            thrust_var.append(np.var(dest_thrust))
            thrust_std.append(np.std(dest_thrust))
            # draw a black line for nominal depth
            if ref_line:
                ax.plot(nf*[ref_line,],'k-',label="Nominal")
            # create legend
            ax.legend()
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate N={N}")
            # save figure
            f.savefig(os.path.join(opath,f"{kk}-depth-estimates-dtw-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-per-{per:.2f}.png"))
            plt.close(f)

            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_torque,'bx')
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate (Torque) N={N}")
            f.savefig(os.path.join(opath,f"{kk}-depth-estimates-dtw-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-per-{per:.2f}-torque-only.png"))
            plt.close(f)

            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_thrust,'rx')
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate (Thrust) N={N}")
            f.savefig(os.path.join(opath,f"{kk}-estimates-dtw-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-per-{per:.2f}-auto-thrust-only.png"))
            plt.close(f)

            # process and add results to the JSON
            data[kk]['depth_est']['Torque'].append([float(x) for x in dest_torque])
            data[kk]['depth_est']['Thrust'].append([float(x) for x in dest_thrust])
            
            data[kk]['mean']['Torque'].append(float(torque_mean[-1]))
            data[kk]['var']['Torque'].append(float(torque_var[-1]))
            data[kk]['std']['Torque'].append(float(torque_std[-1]))
            
            data[kk]['mean']['Thrust'].append(float(thrust_mean[-1]))
            data[kk]['var']['Thrust'].append(float(thrust_var[-1]))
            data[kk]['std']['Thrust'].append(float(thrust_std[-1]))
        # create axes for statistics    
        fstats,axstats = plt.subplots(nrows=2,ncols=3,sharex=True,constrained_layout=True,figsize=(14,12))
        # plot the statistics of each window size
        axstats[0,0].plot(NN,data[kk]['mean']['Torque'],'b-')
        axstats[0,1].plot(NN,data[kk]['var']['Torque'],'r-')
        axstats[0,2].plot(NN,data[kk]['std']['Torque'],'k-')
        axstats[1,0].plot(NN,data[kk]['mean']['Thrust'],'b-')
        axstats[1,1].plot(NN,data[kk]['var']['Thrust'],'r-')
        axstats[1,2].plot(NN,data[kk]['std']['Thrust'],'k-')
        
        axstats[0,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Torque)")
        axstats[0,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Torque)")
        axstats[0,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Torque)")
        axstats[1,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Thrust)")
        axstats[1,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Thrust)")
        axstats[1,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Thrust)")
        fstats.suptitle(f"{kk.capitalize()} Depth Est. Stats using dexp={depth_exp}, win={depth_win}, av=auto, per={per:.2f}")
        fstats.savefig(os.path.join(opath,f"{kk}-depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-av-auto-per-{per:.2f}.png"))
    # save data
    json.dump(data,open(os.path.join(opath,f"depth-estimate-stats-dtw-full-exp-{depth_exp}-window-{depth_win}-av-auto-per-{per:.2f}.json"),'w'),default=str)

def dtw_depth_est(path,tool_dims,per='all',win_mode='nov',N=20,**kwargs):
    from modelling import ToolShapelet,RollingPosWindow
    from tslearn.metrics import dtw
    # create shapelet data
    tool = ToolShapelet(tool_dims)
    # if per is None, default to 60%
    if per is None:
        per = [0.6 * tool.tlength(),]
    # if per is all, generate values along the length of the tool
    elif per == 'all':
        per = np.linspace(0.1,tool.tlength(),N).tolist()
    else:
        per = [per,]
    paths = glob(path)
    # dictionary of values
    dest = {'Torque':{pp:[] for pp in per},'Thrust':{pp:[] for pp in per},'fns' : paths, 'per':per}

    def _score(x,shapelet):
        ''' Score given Pandas window using DTW '''
        x.dropna()
        if isinstance(x,pd.DataFrame):
            x = x['signal']
        return dtw(shapelet,x)
    
    def applyPer(search,pp):
        # get min max and first values used in scaling
        fsc = search.signal[0]
        lsc = search.signal[len(search)-1]
        smin = search.signal.min()
        smax = search.signal.max()
        # group by position
        if win_mode == 'nov':
            gps = search.groupby(search['pos'].apply(lambda x : pp*round(x/pp)))
            pos = np.abs(list(gps.indices.keys()))+pp/2
        elif win_mode == 'ov':
            gps = search.rolling(RollingPosWindow(search.pos,width=pp))
            pos = search.pos.copy()
        
        # generate forward shapelet
        scale = (smax-smin)/2
        shapelet = tool.generate(av,scale,mirror=False)[-1]+fsc
        score = gps.apply(lambda x : _score(x,shapelet)).values.flatten()
        #print(len(search),len(score),np.argmin(score))
        #plt.plot(pos,score[::2])
        #plt.show()
        if win_mode == 'nov':
            xA = pos[np.argmin(score)]
        else:
            xA = pos[np.argmin(score[::2])]

        # generate backward shapelet
        scale = smax - fsc
        if scale==0:
            scale = smax-lsc
            if scale == 0:
                print(f"Scale is 0 for {fn} {key} @ {pp:.2f}mm! Signal ({smin,smax,fsc,lsc}) Setting depth est to -1!")
                return -1
    
        shapelet = tool.generate(av,scale,mirror=True)[-1]+min([fsc,lsc])
        score = gps.apply(lambda x : _score(x,shapelet)).values.flatten()
        #print(len(search),len(score),np.argmin(score))
        if win_mode == 'nov':
            xB = pos[np.argmin(score)]
        else:
            xB = pos[np.argmin(score[::2])]
        return abs(xB-xA)
    # for torque and thrust
    for fn in paths:
        # load data
        data_all = dp.loadSetitecXls(fn,"auto")
        av = dp.getAV(data_all)[-1]
        data = data_all.pop(-1)
        del data_all
        # get position data
        data_pos = np.abs(data['Position (mm)'].values.flatten())
        # get target signal data
        for key in ['Torque','Thrust']:
            signal = data[f"I {key.capitalize()} (A)"].values.flatten()
            if f"I {key.capitalize()} Empty (A)" in data:
                signal += data[f"I {key.capitalize()} Empty (A)"].values.flatten()
            # form into dataframe
            search = pd.DataFrame.from_dict({'pos':data_pos,'signal':signal})
            # apply each target period
            for pp in per:
                dest[key][pp].append(applyPer(search,pp))
    json.dump(dest,open(f"dtw-only-depth-est-per-N-{len(per)}-av-auto-win_mode-{win_mode}.json",'w'))

    # make plots for the depth estimates for each file
    f,ax = plt.subplots(ncols=2,sharex=True,constrained_layout=True)
    for aa,var in zip(ax,["Torque","Thrust"]):
        for pp,dd in dest[var].items():
            #print(per,type(per))
            aa.plot(dd,'x',label=f"per={pp:.2f}mm")
        aa.legend()
        aa.set(xlabel="Hole Number",ylabel="Estimted Depth (mm)",title=var)
    f.suptitle(kwargs.get("title","DTW Only Depth Estimate"))
    # plot statistics
    fstats,axstats = plt.subplots(nrows=2,ncols=2,constrained_layout=True)
    # itertate over each key
    # get the periods used
    periods = dest['per']
    # plot stats about the torque
    axstats[0,0].plot(periods,[np.mean(p) for p in dest['Torque'].values()],'b-')
    axstats[0,0].set(xlabel="Period Size (mm)",ylabel="Mean Depth Est (mm)", title="Mean Depth Estimate (Torque)")
    axstats[0,1].plot(periods,[np.var(p) for p in dest['Torque'].values()],'r-')
    axstats[0,1].set(xlabel="Period Size (mm)",ylabel="Var Depth Est (mm)", title="Var Depth Estimate (Torque)")
    # plot stats about the thrust
    axstats[1,0].plot(periods,[np.mean(p) for p in dest['Thrust'].values()],'b-')
    axstats[1,0].set(xlabel="Period Size (mm)",ylabel="Mean Depth Est (mm)", title="Mean Depth Estimate (Thrust)")
    axstats[1,1].plot(periods,[np.var(p) for p in dest['Thrust'].values()],'r-')
    axstats[1,1].set(xlabel="Period Size (mm)",ylabel="Var Depth Est (mm)", title="Var Depth Estimate (Thrust)")
    fstats.suptitle(kwargs.get("stats_title","DTW Only Depth Estimate Statistics"))
    if kwargs.get('opath',None):
        fstats.savefig(os.path.join(kwargs.get('opath'),f"dtw-only-depth-est-stats-per-N-{len(per)}-av-auto-win_mode-{win_mode}.png"))
        f.savefig(os.path.join(kwargs.get('opath'),f"dtw-only-depth-est-per-N-{len(per)}-av-auto-win_mode-{win_mode}.png"))
    return f,fstats

def kistler_depth_run_roll_corrected(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],add_empty=False,xstart=10.0,end_ref='end',depth_exp=32.0,depth_win=4.0,default=False,pselect='argmax',ref_line=None,tool_dims=[],opath='',plot_steps=False): 
    '''
        Estimate depth of files both with and without the rolling depth correction

        Example of use
        -------------------------
        kistler_depth_run_roll_corrected(path='8B Life Test/*.xls',NN=[10,20,30,40,50],plot_steps=True,add_empty=True,xstart=20.0,end_ref='end',depth_exp=40.0,depth_win=5.0,default=True,pselect='argmax',opath='8B Life Test/plots/rolling_correct',ref_line=40.0)
    '''
    
    from modelling import depth_est_rolling as depth_est_rolling_test, R_pca
    from time import perf_counter
    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.patches as mpatches
    # create dictionary to hold depth estimate for different types of data
    depth_est = {'normal':{},'normal_c':{},'rpca':{},'rpca_c':{}}
    eval_time = {'normal':{},'normal_c':{},'rpca':{},'rpca_c':{}}
    # lists of data vectors
    dt_torque = []
    dt_thrust = []
    # list of step indicies
    sc_mins = []
    # list of pos
    pos = []
    # number of files
    nf = len(glob(path))
    print(f"total {nf} files")
    per = 0.6*math.fsum(tool_dims)
    # minimum length
    min_len = None
    print("Creating data files")
    if plot_steps:
        paths_list = list(glob(path))
    # iterte over each of the files
    for fn in glob(path):
        # load data file
        data = dp.loadSetitecXls(fn,"auto_data")
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        pos.append(xdata)
        # get min transition index
        sc = np.unique(data['Step (nb)'])
        if sc.shape[0]<2:
            warnings.warn(f"Skipping {fn} as it only has one step nc={sc.shape}!")
            sc_mins.append(None)
        else:
            sc_mins.append(data[data['Step (nb)'] == sc[1]].index.min())
        # update clipping length
        min_len = xdata.shape[0] if min_len is None else min([min_len,xdata.shape[0]])
        # iterate over the variables
        for var,dt in zip(["Torque","Thrust"],[dt_torque,dt_thrust]):
            # add entry for variable to depth estimate dict
            for kk in depth_est.keys():
                depth_est[kk][var] = {}
                eval_time[kk][var] = {}
            # get base data
            ydata = data[f'I {var} (A)'].values.flatten()
            # add empty
            if add_empty:
                if f'I {var} Empty (A)' in data:
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            # add to array
            dt.append(ydata)
    if (np.asarray(sc_mins)==None).all():
        warnings.warn("Not performing depth estimate as all step code transitions are None indicating that all the data files only have 1 step!")
        return
    # make copies of arrays
    dt_torque_clip = dt_torque.copy()
    dt_thrust_clip = dt_thrust.copy()
    pos_clip = pos.copy()
    #print(dt_torque_clip.shape,dt_thrust_clip.shape)
    # clip arrays
    print(f"clipping arrays to {min_len}")
    for i in range(nf):
        if dt_torque_clip[i].shape[0]>min_len:
            dt_torque_clip[i] = dt_torque_clip[i][:min_len]
            dt_thrust_clip[i] = dt_thrust_clip[i][:min_len]
            pos_clip[i] = pos_clip[i][:min_len]    
    # stack them to form a 2D matrix
    dt_torque_arr = np.column_stack(dt_torque_clip)
    dt_thrust_arr = np.column_stack(dt_thrust_clip)
    # process using RPCA
    print("processing using RPCA")
    L_torque,_ = R_pca(dt_torque_arr).fit(max_iter=10000)
    L_thrust,_ = R_pca(dt_thrust_arr).fit(max_iter=10000)
    print(f"finished ",L_torque.shape,L_thrust.shape)
    # for a given variable
    for var,L in zip(["Torque","Thrust"],[L_torque,L_thrust]):
        print(f"Processing {var}")
        # iterate over each signal
        for i in range(nf):
            # get data vectors
            xdata = pos_clip[i]
            ydata = L[:,i]
            # update data to unclipped, original data
            ogxdata = pos[i]
            ogydata = dt_torque[i] if var == 'Torque' else dt_thrust[i]
            # plot the data and intermediate steps of correcting the data
            if plot_steps:
                sc_min = sc_mins[i]
                if sc_min is None:
                    continue
                fname = os.path.splitext(os.path.basename(paths_list[i]))[0]
                # make figures for plotting
                # regular data
                f,ax = plt.subplots(constrained_layout=True,ncols=2)
                tax = ax[0].twinx()
                
                # get where the program changes
                dep = ogxdata[sc_min]
                # mask to small window
                mask = (ogxdata>= (dep-1.0)) & (ogxdata<= (dep+1.0))
                xfilt = ogxdata[mask]
                # get torque values for target range
                data_filt = ogydata[mask]
                # plot the target window
                ax[0].plot(xfilt,data_filt,'b-')
                # add vertical line for transition point
                ax[0].vlines(dep,0.9*data_filt.min(),1.1*data_filt.max(),colors=['k'])
                all_dist = []
                # for each window size
                arr = None
                for N in NN:
                    # filter + smooth
                    torque_filt = wiener(data_filt,N)
                    grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
                    dist = dep-xfilt[grad.argmax()]
                    # plot the gradient
                    tax.plot(xfilt,grad,'r-')
                    # draw an arrow going from the peak in the gradient that would be used to where the new location is
                    arr = tax.arrow(x=xfilt[grad.argmax()],y=grad.max(),dx=dist,dy=0,color='green',head_width=0.05,shape='full',length_includes_head=False)
                    all_dist.append(abs(dist))
                # get first lines from each
                lines = [ax[0].lines[0],tax.lines[0],arr]
                ax[0].legend(lines,['Torque','Gradient','Correction'],handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})
                ax[0].set(xlabel='Position (mm)',ylabel='Torque (A)')
                
                # plot the distances
                ax[1].plot(NN,all_dist,'x',markersize=10,markeredgewidth=5)
                ax[1].set_xticks(NN)
                ax[1].set(xlabel="Rolling Window Size (Samples)",ylabel="Correction Distance (mm)")
                f.suptitle(fname)
                f.savefig(os.path.join(opath,f"{fname}-depth-correct-sc.png"))
                plt.close(f)

                ## for RPCA data
                f,ax = plt.subplots(ncols=2,constrained_layout=True)
                tax = ax[0].twinx()
                # mask to small window
                mask = (xdata>= (dep-1.0)) & (xdata<= (dep+1.0))
                xfilt = xdata[mask]
                # get torque values for target range
                data_filt = ydata[mask]
                ax[0].plot(xfilt,data_filt,'b-')
                ax[0].vlines(dep,0.9*data_filt.min(),1.1*data_filt.max(),colors=['k'])
                all_dist.clear()
                for N in NN:
                    # filter + smooth
                    torque_filt = wiener(data_filt,N)
                    grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
                    dist = dep-xfilt[grad.argmax()]
                    # plot the gradient
                    tax.plot(xfilt,grad,'r-')
                    # draw an arrow going from the peak in the gradient that would be used to where the new location is
                    arr = tax.arrow(x=xfilt[grad.argmax()],y=grad.max(),dx=dist,dy=0,color='green',head_width=0.05,shape='full',length_includes_head=False)
                    all_dist.append(abs(dist))

                lines = [ax[0].lines[0],tax.lines[0],arr]
                ax[0].legend(lines,['Torque','Gradient','Correction'],handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})
                ax[0].set(xlabel='Position (mm)',ylabel='Torque (A)')
                # plot the distances
                ax[1].plot(NN,all_dist,'x',markersize=10,markeredgewidth=5)
                ax[1].set_xticks(NN)
                ax[1].set(xlabel="Rolling Window Size (Samples)",ylabel="Correction Distance (mm)")
                f.suptitle(fname + " (RPCA)")
                f.savefig(os.path.join(opath,f"{fname}-rpca-depth-correct-sc.png"))
                plt.close(f)
            # iterate over different window sizes
            for N in NN:
                # estimate depth using RPCA data
                start = perf_counter()
                dest = depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect)
                end = perf_counter() - start
                # save result in dictionary
                if not (N in depth_est['rpca'][var]):
                    depth_est['rpca'][var][N] = []
                    eval_time['rpca'][var][N] = []
                depth_est['rpca'][var][N].append(float(dest))
                eval_time['rpca'][var][N].append(float(end))
                # estimate depth using RPCA data + correction
                start = perf_counter()
                dest = depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect,
                                         correct_dist=True,change_idx=sc_mins[i])
                end = perf_counter() - start
                # save result in dictionary
                if not (N in depth_est['rpca_c'][var]):
                    depth_est['rpca_c'][var][N] = []
                    eval_time['rpca_c'][var][N] = []
                depth_est['rpca_c'][var][N].append(float(dest))
                eval_time['rpca_c'][var][N].append(float(end))
                #print(f"Processing {var} NORMAL {N}")
                # estimate depth using ORIGINAL data
                start = perf_counter()
                dest = depth_est_rolling(ogydata,ogxdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect)
                end = perf_counter() - start
                # save result in dictionary
                if not (N in depth_est['normal'][var]):
                    depth_est['normal'][var][N] = []
                    eval_time['normal'][var][N] = []
                depth_est['normal'][var][N].append(float(dest))
                eval_time['normal'][var][N].append(float(end))
                # estimate depth using ORIGINAL data with correction
                start = perf_counter()
                dest = depth_est_rolling(ogydata,ogxdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect,
                                              correct_dist=True,change_idx=sc_mins[i])
                end = perf_counter() - start
                # save result in dictionary
                if not (N in depth_est['normal_c'][var]):
                    depth_est['normal_c'][var][N] = []
                    eval_time['normal_c'][var][N] = []
                depth_est['normal_c'][var][N].append(float(dest))
                eval_time['normal_c'][var][N].append(float(end))

    # iterate over data type
##    for dtype,vals in depth_est.items():
##        print(dtype)
##        # iterate over variable and window sizes
##        for key,wins in vals.items():
##            print(f"\t{key}")
##            for N,dest in wins.items():
##                print(f"\t\t{N}:{len(dest)}")
##                print(dest)
##    input()        
    json.dump(depth_est,open(os.path.join(opath,f"depth-estimates-rolling-correct-full-exp-{depth_exp}-window-{depth_win}-av-auto-per-{per:.2f}.json"),'w'),default=str)
    json.dump(eval_time,open(os.path.join(opath,f"depth-estimates-rolling-correct-full-eval-time-exp-{depth_exp}-window-{depth_win}-av-auto-per-{per:.2f}.json"),'w'),default=str)
    # create lists to process
    thrust_mean = []
    thrust_var = []
    thrust_std = []
    torque_mean = []
    torque_var = []
    torque_std = []
    # define JSON to save
    data = {kk : {'NN':NN,'depth_est' : {'Thrust' : [], 'Torque': []}, 'mean' : {'Thrust' : [], 'Torque': []}, 'var' : {'Thrust' : [], 'Torque': []}, 'std' : {'Thrust' : [], 'Torque': []}} for kk in depth_est.keys()}
    data_corr = {kk : {'NN':NN,'depth_est' : {'Thrust' : [], 'Torque': []}, 'mean' : {'Thrust' : [], 'Torque': []}, 'var' : {'Thrust' : [], 'Torque': []}, 'std' : {'Thrust' : [], 'Torque': []}} for kk in depth_est.keys()}
    # iterate over each type of estimate
    for kk,dest in depth_est.items():
        print(f"Plotting {kk} results")
        # iterate over each window size
        for N in NN:
            # get torque and thrust depth estimates for window size
            dest_torque = dest["Torque"][N]
            dest_thrust = dest["Thrust"][N]
            # create axes
            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_torque,'bx',label="Torque")
            torque_mean.append(np.mean(dest_torque))
            torque_var.append(np.var(dest_torque))
            torque_std.append(np.std(dest_torque))
            # plot thrust depth estimate with red X's
            ax.plot(dest_thrust,'rx',label="Thrust")
            thrust_mean.append(np.mean(dest_thrust))
            thrust_var.append(np.var(dest_thrust))
            thrust_std.append(np.std(dest_thrust))
            # draw a black line for nominal depth
            if ref_line:
                ax.plot(nf*[ref_line,],'k-',label="Nominal")
            # create legend
            ax.legend()
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate N={N}")
            # save figure
            f.savefig(os.path.join(opath,f"{kk}-depth-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-per-{per:.2f}.png"))
            plt.close(f)

            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_torque,'bx')
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate (Torque) N={N}")
            f.savefig(os.path.join(opath,f"{kk}-depth-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-per-{per:.2f}-torque-only.png"))
            plt.close(f)

            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_thrust,'rx')
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate (Thrust) N={N}")
            f.savefig(os.path.join(opath,f"{kk}-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-per-{per:.2f}-auto-thrust-only.png"))
            plt.close(f)

            # process and add results to the JSON
            data[kk]['depth_est']['Torque'].append([float(x) for x in dest_torque])
            data[kk]['depth_est']['Thrust'].append([float(x) for x in dest_thrust])
            
            data[kk]['mean']['Torque'].append(float(torque_mean[-1]))
            data[kk]['var']['Torque'].append(float(torque_var[-1]))
            data[kk]['std']['Torque'].append(float(torque_std[-1]))
            
            data[kk]['mean']['Thrust'].append(float(thrust_mean[-1]))
            data[kk]['var']['Thrust'].append(float(thrust_var[-1]))
            data[kk]['std']['Thrust'].append(float(thrust_std[-1]))
        # create axes for statistics    
        fstats,axstats = plt.subplots(nrows=2,ncols=3,sharex=True,constrained_layout=True,figsize=(14,12))
        # plot the statistics of each window size
        axstats[0,0].plot(NN,data[kk]['mean']['Torque'],'b-')
        axstats[0,1].plot(NN,data[kk]['var']['Torque'],'r-')
        axstats[0,2].plot(NN,data[kk]['std']['Torque'],'k-')
        axstats[1,0].plot(NN,data[kk]['mean']['Thrust'],'b-')
        axstats[1,1].plot(NN,data[kk]['var']['Thrust'],'r-')
        axstats[1,2].plot(NN,data[kk]['std']['Thrust'],'k-')
        
        axstats[0,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Torque)")
        axstats[0,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Torque)")
        axstats[0,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Torque)")
        axstats[1,0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Thrust)")
        axstats[1,1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Thrust)")
        axstats[1,2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Thrust)")
        fstats.suptitle(f"{kk.capitalize()} Depth Est. Stats using dexp={depth_exp}, win={depth_win}, av=auto, per={per:.2f}")
        fstats.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-av-auto-per-{per:.2f}.png"))
    plt.close('all')
    # save data
    json.dump(data,open(os.path.join(opath,f"depth-estimate-stats-rolling-correct-full-exp-{depth_exp}-window-{depth_win}-av-auto-per-{per:.2f}.json"),'w'),default=str)

def eval_time(fn,ogydata,ogxdata,**kwargs):
    start = perf_counter()
    dest = depth_est_rolling(ogydata,ogxdata,**kwargs)
    end = perf_counter() - start
    return float(end)

# record evaluation time of full and slim version of depth estimation function
def evaluate_full_vs_slim(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],add_empty=False,xstart=10.0,end_ref='end',depth_exp=32.0,depth_win=4.0,default=False,pselect='argmax',ref_line=None,tool_dims=[],opath='',plot_steps=False): 
    '''
        Record and compare how long the full and slim version of depth_est_rolling takes to run

        Example of use
        -----------------------
        evaluate_full_vs_slim(path='8B Life Test/*.xls',NN=[10,20,30,40,50],plot_steps=True,add_empty=True,xstart=20.0,end_ref='end',depth_exp=40.0,depth_win=5.0,default=True,pselect='argmax',opath='8B Life Test/plots/rolling_correct',ref_line=40.0)
    '''
    import multiprocessing as mp
    from modelling import R_pca
    from time import perf_counter
    eval_time = {'normal' : {win : [] for win in NN},'slim' : {win : [] for win in NN}}
    # lists of data vectors
    dt_torque = []
    dt_thrust = []
    # list of step indicies
    sc_mins = []
    # list of pos
    pos = []
    # number of files
    nf = len(glob(path))
    print(f"total {nf} files")
    # minimum length
    min_len = None
    print("Creating data files")
    # iterte over each of the files
    for fn in glob(path):
        # load data file
        data = dp.loadSetitecXls(fn,"auto_data")
        # get position data data
        xdata = np.abs(data['Position (mm)'].values.flatten())
        pos.append(xdata)
        # get min transition index
        sc = np.unique(data['Step (nb)'])
        if sc.shape[0]<2:
            warnings.warn(f"Skipping {fn} as it only has one step nc={sc.shape}!")
            sc_mins.append(None)
        else:
            sc_mins.append(data[data['Step (nb)'] == sc[1]].index.min())
        # iterate over the variables
        for var,dt in zip(["Torque","Thrust"],[dt_torque,dt_thrust]):
            # get base data
            ydata = data[f'I {var} (A)'].values.flatten()
            # add empty
            if add_empty:
                if f'I {var} Empty (A)' in data:
                    ydata += data[f'I {var} Empty (A)'].values.flatten()
            # add to array
            dt.append(ydata)
    if (np.asarray(sc_mins)==None).all():
        warnings.warn("Not performing depth estimate as all step code transitions are None indicating that all the data files only have 1 step!")
        return
    times = []
    with mp.Pool(6) as pool:
        # for a given variable
        for var,L in zip(["Torque","Thrust"],[dt_torque,dt_thrust]):
            print(f"Processing {var}")
            # iterate over each signal
            for i in range(nf):
                # update data to unclipped, original data
                ogxdata = pos[i]
                ogydata = dt_torque[i] if var == 'Torque' else dt_thrust[i]
                # iterate over different window sizes
                for N in NN:
                    # estimate depth using ORIGINAL data with correction
                    times.clear()
                    for _ in range(1):
                        start = perf_counter()
                        dest = depth_est_rolling(ogydata,ogxdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect,
                                                      correct_dist=True,change_idx=sc_mins[i])
                        end = perf_counter() - start
                        times.append(float(end))
                    eval_time['normal'][N].append(times)

                    times.clear()
                    for _ in range(1):
                        start = perf_counter()
                        dest = depth_est_rolling_slim(ogydata,ogxdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect,
                                                      correct_dist=True,change_idx=sc_mins[i])
                        end = perf_counter() - start
                        times.append(float(end))
                    eval_time['slim'][N].append(times)

    for ftype,wins in eval_time.items():
        print(ftype)
        for N,times in wins.items():
            print("\n{N}:{np.asarray(times).shape}")

    json.dump(eval_time,open(os.path.join(opath,f"depth-estimates-rolling-correct-eval-time-full-vs-slim.json"),'w'),default=str)
    slim_res = {}
    for ftype,vals in eval_time.items():
        f,ax = plt.subplots(constrained_layout=True)
        times = [np.mean(t) for t in vals.values()]
        ax.plot(vals.keys(),times,'bx')
        ax.set(xlabel="Rolling Window Size (samples)",ylabel="Average Evaluation Time (s)",title=f"{ftype} Average Evaluation Time")
        #f.savefig(os.path.join(opath,f"{ftype}-average-eval-time.png"))
        #plt.close(f)

    f,ax = plt.subplots(constrained_layout=True)
    # expecting regular to take longer
    bins = np.linspace(min(min(eval_time['normal'][N]) for N in NN),max(max(eval_time['normal'][N]) for N in NN),1.0)
    ax.hist(np.array(list(eval_time['normal'].values())).flatten(), bins, alpha=0.5, label='normal')
    ax.hist(np.array(list(eval_time['slim'].values())).flatten(), bins, alpha=0.5, label='slim')
    ax.set(xlabel="Evaluation Time (s)",ylabel="Count",title="Evaluation Time normal vs slim")
    ax.legend()
    #f.savefig(os.path.join(opath,f"{ftype}-all-eval-time.png"))
    #plt.close('all')

def forwardBackwardGradient(path,NN,win=1.0):
    from matplotlib.legend_handler import HandlerPatch
    import matplotlib.patches as mpatches
    data = dp.loadSetitecXls(path,'auto_data')
    # get filename to use as axis titles
    fname = os.path.splitext(os.path.basename(path))[0]
    xdata = np.abs(data['Position (mm)'].values)
    ydata = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    # check there are multiple unique step codes
    sc = data['Step (nb)'].values
    sc_uq = np.unique(data['Step (nb)'])
    # check that there's at least step codes
    if sc.shape[0]<2:
        warnings.warn("Data does not contain multiple step codes so cannot correct!")
        return None
    # get index where the step code changes
    sc_min = data[sc == sc_uq[1]].index.min()
    # make plots
    f,ax = plt.subplots(ncols=2,constrained_layout=True)
    tax = ax[0].twinx()
    # get where the program changes
    dep = xdata[sc_min]
    # add vertical line for transition point
    ax[0].vlines(dep,0.0,10.0,colors=['k'])
    # due to rolling window, peaks in the gradient occur after the event
    # only search for peaks that come after
    mask = (xdata >= (dep-win)) & (xdata <= (dep+win))
    xdata = xdata[mask]
    torque_filt_o = ydata[mask]
    all_dist = []
    all_dist_bwd = []
    # for each window size
    arr = None
    arr_bwd = None
    for N in NN:
        # filter + smooth
        torque_filt = wiener(torque_filt_o,N)
        grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
        # get the correction distance
        dist = dep-xdata[grad.argmax()]
        # plot data
        ax[0].plot(xdata,torque_filt,'b-')
        # plot gradient
        tax.plot(xdata,grad,'r-')
        arr = tax.arrow(x=xdata[grad.argmax()],y=grad.max(),dx=dist,dy=0,color='green',head_width=0.05,shape='full',length_includes_head=False)

        # calculate rolling gradient of torque in reverse
        grad = rolling_gradient(torque_filt[::-1],N) * tukey(len(torque_filt),0.1,True)
        tax.plot(xdata[::-1],grad,'k-')
        # as the gradient is in reverse the signs are in reverse too
        # to a normal peak is now negative
        dist_bwd = dep-xdata[::-1][grad.argmin()]
        # draw arrow showing correction distance
        arr_bwd = tax.arrow(x=xdata[::-1][grad.argmin()],y=grad.min(),dx=dist_bwd,dy=0,color='magenta',head_width=0.05,shape='full',length_includes_head=False)
        # store distance
        all_dist.append(abs(dist))
        all_dist_bwd.append(abs(dist_bwd))
    # get first lines from each
    lines = [ax[0].lines[0],tax.lines[0],tax.lines[1],arr,arr_bwd]
    ax[0].legend(lines,['Torque','Gradient (Fwd)','Gradient (Bwd)','Correction'],handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})
    ax[0].set(xlabel='Position (mm)',ylabel='Torque (A)',title='Data')
    f.suptitle(fname)

    ax[1].plot(NN,all_dist,'x')
    ax[1].plot(NN,all_dist_bwd,'kx')
    ax[1].set(xlabel="Window Size",ylabel="Distance (mm)",title="Window Distance")
    return f
    
###### TARGETS ########
##Depth Est = 31.89
##Std = 0.06
##Var = 0.003
#######################
### 4B tool lengths ###
def tool_4B():
    ts1 = np.tan(np.radians(67.5))*(6.324-5.79)
    dims = [1.2,3,ts1,8-3-ts1]
    av=3.0
    return dims
### 8B tool lengths ###
def tool_8B():
    ts1 = np.tan(np.radians(67.5))*(12.68-11.57)
    dims=[np.tan(np.radians(22.5))*(11.57/2),3.0,ts1,8-3-ts1]
    av=3.0
    return dims

def depth_est_segmented(file):
    """
    Estimate the depth using segmented keypoint recognition gradient.

    Parameters:
    file (str): The file path of the input image.

    Returns:
    list: A list of depth estimation results.

    """
    l_result = uos.kp_recognition_gradient(file)
    l_output = []
    for istep, row in enumerate(l_result):
        print(row)
        l_row = [row[0]['medianmax'], row[0]['medianmin']]
        l_output.append(l_row)

    l_output_depths = (l_output[1][0], l_output[2][0], l_output[2][1])
    return l_output_depths

def depth_est_ml(file):
    """
    Estimate the depth using machine learning.

    Args:
        file (str): The file path of the input data.

    Returns:
        list: The estimated depth values.

    """
    l_result = uos.kp_recognition_ml(file)
    return l_result


if __name__ == "__main__":
    import abyss
    abysspath = abyss.__path__[0]
    #windowPeakWarnings('8B Random Life Test/*.xls',xstart=20.0,opath='8B Random Life Test/plots')
    #kistler_depth_run_dexp(path='8B Random Life Test/*.xls',NN=[10,20,30,40,50],plot_steps=False,add_thrust_empty=True,use_signs=True,xstart=20.0,end_ref='end',depth_exp=40.0,depth_win=5.0,default=True,pselect='argmax',opath='8B Random Life Test/plots',ref_line=40.0)
    #kistler_depth_run_dtw(path='8B life test/*.xls',NN=[10,20,30,40,50],add_empty=True,xstart=20.0,depth_exp=40.0,depth_win=5.0,default=True,opath='8B Life Test/plots',ref_line=40.0,tool_dims=tool_8B())
    #dest = dtw_depth_est(path='8B life test/*.xls',tool_dims=tool_8B(),win_mode='nov')
    #dtw_depth_est(path='8B life test/*.xls',tool_dims=tool_8B(),win_mode='ov')
    #kistler_depth_run_roll_corrected(path='8B Random Life Test/*.xls',NN=[10,20,30,40,50],plot_steps=True,add_empty=True,xstart=20.0,end_ref='end',depth_exp=40.0,depth_win=5.0,default=True,pselect='argmax',opath='8B Random Life Test/plots/rolling_correct',ref_line=40.0)
    #evaluate_full_vs_slim(path='8B Random Life Test/*.xls',NN=[10,20,30,40,50],plot_steps=True,add_empty=True,xstart=20.0,end_ref='end',depth_exp=40.0,depth_win=5.0,default=True,pselect='argmax',opath='8B Random Life Test/plots/rolling_correct',ref_line=40.0)

    #c = [depthCorrectSC(fn,10) for fn in glob('8B life test/*.xls')]
    #plotTorqueGradient('AirbusData/Seti-Tec data files life test/*.xls',[10,20,30,40,50])
    #plotDepthCorrectSC('8B Life Test/E00401009F45AF14_18080018_ST_2109_27.xls',close_figs=False)
    #plotDepthCorrectedDist('8B Life Test/*.xls')
    #plotDepthCorrectedDist('8B Random Life Test/*.xls')
    #plotDepthCorrectedDist('COUNTERSPARK/data/4B/*.xls')
    # plt.show()
    filename='17070141_17070141_ST_753_55.xls'
    filename='UNK/2312041_23120041_ST_1378_95.xls'
    abspath = r'C:\Users\NG9374C\Documents\uos-drilling\abyss\test_data\UNK\2312041_23120041_ST_1378_95.xls'
    # result = depth_est_segmented(f'{abysspath}/test_data/{filename}')
    result = depth_est_segmented(abspath)
    print("*********************************************")
    print("Segmented Keypoint Recognition Gradient Results:")
    print("*********************************************")
    print(result)
    print("*********************************************")
    print("Machine Learning Results:")
    print("*********************************************")
    # result_ml = depth_est_ml(f'{abysspath}/test_data/{filename}')
    result_ml = depth_est_ml(abspath)
    print(result_ml)
