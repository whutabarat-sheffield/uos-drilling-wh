import os
from modelling import weighted_savgol_filter, rolling_gradient, EmptyMaskException, MissingPeaksException, MissingPeaksWarning, EmptyMaskWarning, R_pca
from scipy.signal import find_peaks, get_window, peak_prominences, peak_widths, wiener
from scipy.signal.windows import tukey
import abyss.dataparser as dp
from time import perf_counter
import warnings
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import pandas as pd

'''
    This script is for final testing of the algorithm before handing over the final result.
    This is different from the rolling_gradient sandbox script as it has fewer functions to worry about
'''

# from https://stackoverflow.com/a/22349717
def make_legend_arrow(legend, orig_handle,xdescent, ydescent,width, height, fontsize):
    import matplotlib.patches as mpatches
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

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
            correct_dist : Correct the distance estimate by pull-back method. Default True. If True, requires change_index or step_vector.
            change_index : Where the program changes between two program steps, ideally 0->1. Used in pull-back correction.
            step_vector  : Full step code vector from Setitec file. Used to find where the program changes between two program steps. Used in pull-back correction.
            correct_win  : Size of search window used for pull-back method in mm. Default 1.0mm.

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
    grad = rolling_gradient(filt_weight,NB)
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
                if len(set(sc))<2:
                    warnings.warn(f"{fn} data does not contain multiple step codes so cannot correct!")
                else:
                    # find the first index where 2nd step code occurs
                    # IOW where it transtiioned between step codes
                    if isinstance(sc,pd.Series):
                        sc_min = [sc == sc[1]].index.min()
                    else:
                        sc_min = np.where(np.asarray(sc) == sc.unique()[1])[0]
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

def estimate_xstart(path,proc=np.max):
    '''
        Function for estimating the xstart parameter of a dataset

        xstart is the distance from the start where the depth_est_rolling searches for the first reference point

        The first reference point is typically where the torque first increases as it enters the material.

        When it first enters the material, it also typically changes program step (0 -> 1). Therefore, a way to estimate
        the xstart parameter is to look at the distance from the start to the first program step change.
        NOTE (Windo): Hamburg data does not change the program step when it first enters the material. I am not sure on the above assumption

        The input proc is the function applied to the vector of these distances e.g. np.mean

        Inputs:
            path : Wildcard path to Setitec files
            proc : Function to process vector of distances or None. If None, then the vector of distances is returned.
                    Default np.max.

        Returns result of proc or just the vector of distances if proc is None.
    '''
    sc_min = []    
    for fn in glob(path):
        data = dp.loadSetitecXls(fn,"auto_data")
        pos = np.abs(data['Position (mm)'].values)
        # get step codes
        sc = data['Step (nb)']
        uqsc = sc.unique()
        # if there are less then two step codes then we can't find where it transitions
        if uqsc.shape[0]<2:
            warnings.warn(f"{fn} data does not contain multiple step codes so cannot correct!")
            continue
        else:
            # find the first index where 2nd step code occurs
            # IOW where it transtiioned between step codes
            sc_min.append(pos[data[data['Step (nb)'] == uqsc[1]].index.min()]-pos[0])
    return proc(sc_min) if proc else sc_min

def depth_est_run(path="AirbusData/Seti-Tec data files life test/*.xls",NN=[10,20,30,40,50],add_empty=False,xstart=10.0,end_ref='end',depth_exp=32.0,depth_win=4.0,default=False,pselect='argmax',opath='',plot_steps=False): 
    '''
        Estimate depth of files both with & without the rolling depth correction (pull back) and with & without RPCA in different combinations

        If plot_steps is True, the gradient at different stages of filtering is saved creating many more plots.

        The results are saves as JSONs

        Inputs:
            path         Where the data files are
            NN           List of rolling window sizes. Default [10,20,30,40,50]
            plot_steps   Flag to plot the signals after they've been filtered
            add_empty    Flag to add the empty channel to signals
            xstar        Search period from start of the signal to look for first reference point
            depth_exp    Expected depth. used in searching for second reference point
            depth_win    Search window in mm around 2nd reference point
            default      Flag to default to certain values rather than raise exceptions
            opath        Output path where files are saved

        Example of use
        -------------------------
        depth_est_run(path='8B Life Test/*.xls',NN=[10,20,30,40,50],plot_steps=True,add_empty=True,xstart=20.0,depth_exp=40.0,depth_win=5.0,default=True,opath='8B Life Test/plots/rolling_correct')
    '''
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
    if plot_steps:
        input(f"WARNING: With plot_steps True, this function will produce A LOT OF plots in {opath}! Do you still want to continue? If not, reset the IDLE or press CTRL+C to exit now")
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
                f.savefig(os.path.join(opath,f"{fname}-rpca-and-depth-correct-sc.png"))
                plt.close(f)

                ## for orignl data
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
                f.savefig(os.path.join(opath,f"{fname}-depth-correct-sc.png"))
                plt.close(f)

                ### Plot the gradient of the signal
                # smooth using wiener
                for N in NN:
                    filt_weight = wiener(ydata,N)
                    # perform rolling gradient on smoothed data
                    grad = rolling_gradient(filt_weight,N)
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
                    # set labels
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    cax.set_ylabel("Rolling Gradient")
                    # combine legends together
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = cax.get_legend_handles_labels()
                    cax.legend(lines + lines2, labels + labels2, loc=0)
                    
                    f.suptitle(f"{fname} {var}\nWiener Filtered + Rolling Gradient N={N} {'Empty Added' if add_empty else ''}")
                    f.savefig(os.path.join(opath,f"{fname}-{var}-wiener-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}.png"))
                    plt.close(f)

                    # smooth using wiener
                    filt_weight = wiener(ogydata,N)
                    # perform rolling gradient on smoothed data
                    grad = rolling_gradient(filt_weight,N)
                    # weight to remove edge artifacts
                    win = tukey(len(grad),0.1,True)
                    grad *= win
                    pks,_ = find_peaks(grad,height=grad.max()*0.1)
                    # plot data
                    f,ax = plt.subplots(constrained_layout=True)
                    ax.plot(ogxdata,ogydata,'b-',label='Original')
                    ax.plot(ogxdata,filt_weight,'r-',label="Filtered")
                    cax = ax.twinx()
                    cax.plot(ogxdata,grad,'k-',label="Rolling Gradient")
                    cax.plot(ogxdata[pks],grad[pks],'gx',markersize=10,markeredgewidth=4,label="Peaks")
                    # set labels
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    cax.set_ylabel("Rolling Gradient")
                    # combine legends together
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = cax.get_legend_handles_labels()
                    cax.legend(lines + lines2, labels + labels2, loc=0)
                    
                    f.suptitle(f"{fname} {var}\nWiener Filtered + RPCA Rolling Gradient N={N} {'Empty Added' if add_empty else ''}")
                    f.savefig(os.path.join(opath,f"{fname}-{var}-wiener-rpca-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}.png"))
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
    json.dump(depth_est,open(os.path.join(opath,f"depth-estimates-rolling-correct-full-exp-{depth_exp}-window-{depth_win}.json"),'w'),default=str)
    json.dump(eval_time,open(os.path.join(opath,f"depth-estimates-rolling-correct-full-eval-time-exp-{depth_exp}-window-{depth_win}.json"),'w'),default=str)
    # create lists to process
    thrust_mean = []
    thrust_var = []
    thrust_std = []
    torque_mean = []
    torque_var = []
    torque_std = []
    # define JSON to save
    data = {kk : {'depth_est' : {'Thrust' : [], 'Torque': []}, 'mean' : {'Thrust' : [], 'Torque': []}, 'var' : {'Thrust' : [], 'Torque': []}, 'std' : {'Thrust' : [], 'Torque': []}} for kk in depth_est.keys()}
    data['NN'] = NN
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
            ax.plot(nf*[depth_exp,],'k-',label="Nominal")
            # create legend
            ax.legend()
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate N={N}")
            # save figure
            f.savefig(os.path.join(opath,f"{kk}-depth-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}.png"))
            plt.close(f)

            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_torque,'bx')
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate (Torque) N={N}")
            f.savefig(os.path.join(opath,f"{kk}-depth-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-torque-only.png"))
            plt.close(f)

            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_thrust,'rx')
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate (Thrust) N={N}")
            f.savefig(os.path.join(opath,f"{kk}-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-auto-thrust-only.png"))
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
        # fstats.suptitle(f"{kk.capitalize()} Depth Est. Stats using dexp={depth_exp}, win={depth_win}, av=auto, per={per:.2f}")
        fstats.suptitle(f"{kk.capitalize()} Depth Est. Stats using dexp={depth_exp}, win={depth_win}, av=auto, per=")
        fstats.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-av-auto.png"))

        #### plot KDE for different window sizes
        for vv in ["Torque","Thrust"]:
            # form into a pandas dataframe
            df = pd.DataFrame(np.array(data[kk]['depth_est'][vv]).T,columns=[str(N) for N in NN])
            # plot the kernel desnsity estimate
            ax = sns.kdeplot(df)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-{vv}.png"))
            plt.close(ax.figure)
            # plot with bars
            ax = sns.histplot(df,kde=True)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-with-bars-{vv}.png"))
            plt.close(ax.figure)
            # plot with only bars
            ax = sns.histplot(df,kde=False)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-only-bars-{vv}.png"))
            plt.close(ax.figure)

            #### plot evaluation time
            # df = pd.DataFrame(np.array(eval_time[kk]['Torque']).T,columns=[str(N) for N in NN])
            # plot the kernel desnsity estimate
            ax = sns.kdeplot(df)
            ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-kde-{vv}.png"))
            plt.close(ax.figure)
            # plot with bars
            ax = sns.histplot(df,kde=True)
            ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-kde-with-bars-{vv}.png"))
            plt.close(ax.figure)
            # plot with only bars
            ax = sns.histplot(df,kde=False)
            ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-only-bars-{vv}.png"))
            plt.close(ax.figure)
            
            # # distribution plots showing the influence of different correction combinations at a given window size
            # for N in NN:
            #     df = pd.DataFrame(np.array(depth_est['normal'][var][N],
            #         depth_est['normal_c'][var][N],
            #         depth_est['rpca'][var][N],
            #         depth_est['rpca_c'][var][N]).T,columns=['Normal',r'Normal /w PB','RPCA','RPCA /w PB'])

            #     # plot the kernel desnsity estimate
            #     ax = sns.kdeplot(df)
            #     ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv}) N={N}")
            #     ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-{vv}-N-{N}.png"))
            #     plt.close(ax.figure)
            #     # plot with bars
            #     ax = sns.histplot(df,kde=True)
            #     ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv}) N={N}")
            #     ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-with-bars-{vv}-N-{N}.png"))
            #     plt.close(ax.figure)
            #     # plot with only bars
            #     ax = sns.histplot(df,kde=False)
            #     ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv}) N={N}")
            #     ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-only-bars-{vv}-N-{N}.png"))
            #     plt.close(ax.figure)
        
    plt.close('all')
    # save data
    json.dump(data,open(os.path.join(opath,f"depth-estimate-stats-rolling-correct-full-exp-{depth_exp}-window-{depth_win}-av-auto.json"),'w'),default=str)



def depth_est_run_df(df=None,NN=[10,20,30,40,50],add_empty=False,xstart=10.0,end_ref='end',depth_exp=32.0,depth_win=4.0,default=False,pselect='argmax',opath='',plot_steps=False): 
    '''
        Estimate depth of files both with & without the rolling depth correction (pull back) and with & without RPCA in different combinations

        If plot_steps is True, the gradient at different stages of filtering is saved creating many more plots.

        The results are saves as JSONs

        Inputs:
            path         Where the data files are
            NN           List of rolling window sizes. Default [10,20,30,40,50]
            plot_steps   Flag to plot the signals after they've been filtered
            add_empty    Flag to add the empty channel to signals
            xstar        Search period from start of the signal to look for first reference point
            depth_exp    Expected depth. used in searching for second reference point
            depth_win    Search window in mm around 2nd reference point
            default      Flag to default to certain values rather than raise exceptions
            opath        Output path where files are saved

        Example of use
        -------------------------
        depth_est_run(path='8B Life Test/*.xls',NN=[10,20,30,40,50],plot_steps=True,add_empty=True,xstart=20.0,depth_exp=40.0,depth_win=5.0,default=True,opath='8B Life Test/plots/rolling_correct')
    '''
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
    # nf = len(glob(path))
    nf = df.File_Path.unique().shape[0]
    if plot_steps:
        input(f"WARNING: With plot_steps True, this function will produce A LOT OF plots in {opath}! Do you still want to continue? If not, reset the IDLE or press CTRL+C to exit now")
    # minimum length
    min_len = None
    print("Creating data files")
    if plot_steps:
        paths_list = list(glob(path))
    # iterte over each of the files
    fnlist = df.File_Path.unique().tolist()
    for fn in fnlist:
        # load data file
        # data = dp.loadSetitecXls(fn,"auto_data")
        # load the correct dataframe
        mask = df.File_Path==fn
        data = df[mask]
        # get position data data
        xdata = np.abs(data['Position_mm'].values.flatten())
        pos.append(xdata)
        # get min transition index
        sc = np.unique(data['Step_nb'])
        if sc.shape[0]<2:
            warnings.warn(f"Skipping {fn} as it only has one step nc={sc.shape}!")
            sc_mins.append(None)
        else:
            sc_mins.append(data[data['Step_nb'] == sc[1]].index.min())
        # update clipping length
        min_len = xdata.shape[0] if min_len is None else min([min_len,xdata.shape[0]])
        # iterate over the variables
        for var,dt in zip(["Torque","Thrust"],[dt_torque,dt_thrust]):
            # add entry for variable to depth estimate dict
            for kk in depth_est.keys():
                depth_est[kk][var] = {}
                eval_time[kk][var] = {}
            # get base data
            ydata = data[f'I_{var}_A'].values.flatten()
            # add empty
            if add_empty:
                if f'I_{var}_Empty_A' in data:
                    ydata += data[f'I_{var}_Empty_A'].values.flatten()
            # add to array
            dt.append(ydata)

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
                f.savefig(os.path.join(opath,f"{fname}-rpca-and-depth-correct-sc.png"))
                plt.close(f)

                ## for orignl data
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
                f.savefig(os.path.join(opath,f"{fname}-depth-correct-sc.png"))
                plt.close(f)

                ### Plot the gradient of the signal
                # smooth using wiener
                for N in NN:
                    filt_weight = wiener(ydata,N)
                    # perform rolling gradient on smoothed data
                    grad = rolling_gradient(filt_weight,N)
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
                    # set labels
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    cax.set_ylabel("Rolling Gradient")
                    # combine legends together
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = cax.get_legend_handles_labels()
                    cax.legend(lines + lines2, labels + labels2, loc=0)
                    
                    f.suptitle(f"{fname} {var}\nWiener Filtered + Rolling Gradient N={N} {'Empty Added' if add_empty else ''}")
                    f.savefig(os.path.join(opath,f"{fname}-{var}-wiener-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}.png"))
                    plt.close(f)

                    # smooth using wiener
                    filt_weight = wiener(ogydata,N)
                    # perform rolling gradient on smoothed data
                    grad = rolling_gradient(filt_weight,N)
                    # weight to remove edge artifacts
                    win = tukey(len(grad),0.1,True)
                    grad *= win
                    pks,_ = find_peaks(grad,height=grad.max()*0.1)
                    # plot data
                    f,ax = plt.subplots(constrained_layout=True)
                    ax.plot(ogxdata,ogydata,'b-',label='Original')
                    ax.plot(ogxdata,filt_weight,'r-',label="Filtered")
                    cax = ax.twinx()
                    cax.plot(ogxdata,grad,'k-',label="Rolling Gradient")
                    cax.plot(ogxdata[pks],grad[pks],'gx',markersize=10,markeredgewidth=4,label="Peaks")
                    # set labels
                    ax.set(xlabel="Position (mm)",ylabel=f"{var} (A)")
                    cax.set_ylabel("Rolling Gradient")
                    # combine legends together
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = cax.get_legend_handles_labels()
                    cax.legend(lines + lines2, labels + labels2, loc=0)
                    
                    f.suptitle(f"{fname} {var}\nWiener Filtered + RPCA Rolling Gradient N={N} {'Empty Added' if add_empty else ''}")
                    f.savefig(os.path.join(opath,f"{fname}-{var}-wiener-rpca-tukey-rolling-gradient-pks-N-{N}-pselect-{pselect}.png"))
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
    json.dump(depth_est,open(os.path.join(opath,f"depth-estimates-rolling-correct-full-exp-{depth_exp}-window-{depth_win}.json"),'w'),default=str)
    json.dump(eval_time,open(os.path.join(opath,f"depth-estimates-rolling-correct-full-eval-time-exp-{depth_exp}-window-{depth_win}.json"),'w'),default=str)
    # create lists to process
    thrust_mean = []
    thrust_var = []
    thrust_std = []
    torque_mean = []
    torque_var = []
    torque_std = []
    # define JSON to save
    data = {kk : {'depth_est' : {'Thrust' : [], 'Torque': []}, 'mean' : {'Thrust' : [], 'Torque': []}, 'var' : {'Thrust' : [], 'Torque': []}, 'std' : {'Thrust' : [], 'Torque': []}} for kk in depth_est.keys()}
    data['NN'] = NN
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
            ax.plot(nf*[depth_exp,],'k-',label="Nominal")
            # create legend
            ax.legend()
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate N={N}")
            # save figure
            f.savefig(os.path.join(opath,f"{kk}-depth-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}.png"))
            plt.close(f)

            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_torque,'bx')
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate (Torque) N={N}")
            f.savefig(os.path.join(opath,f"{kk}-depth-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-torque-only.png"))
            plt.close(f)

            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_thrust,'rx')
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate (Thrust) N={N}")
            f.savefig(os.path.join(opath,f"{kk}-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}-auto-thrust-only.png"))
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
        fstats.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-av-auto.png"))

        #### plot KDE for different window sizes
        for vv in ["Torque","Thrust"]:
            # form into a pandas dataframe
            df = pd.DataFrame(np.array(data[kk]['depth_est'][vv]).T,columns=[str(N) for N in NN])
            # plot the kernel desnsity estimate
            ax = sns.kdeplot(df)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-{vv}.png"))
            plt.close(ax.figure)
            # plot with bars
            ax = sns.histplot(df,kde=True)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-with-bars-{vv}.png"))
            plt.close(ax.figure)
            # plot with only bars
            ax = sns.histplot(df,kde=False)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-only-bars-{vv}.png"))
            plt.close(ax.figure)

            #### plot evaluation time
            df = pd.DataFrame(np.array(eval_time[kk]['Torque']).T,columns=[str(N) for N in NN])
            # plot the kernel desnsity estimate
            ax = sns.kdeplot(df)
            ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-kde-{vv}.png"))
            plt.close(ax.figure)
            # plot with bars
            ax = sns.histplot(df,kde=True)
            ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-kde-with-bars-{vv}.png"))
            plt.close(ax.figure)
            # plot with only bars
            ax = sns.histplot(df,kde=False)
            ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-only-bars-{vv}.png"))
            plt.close(ax.figure)
            
            # distribution plots showing the influence of different correction combinations at a given window size
            for N in NN:
                df = pd.DataFrame(np.array(depth_est['normal'][var][N],
                    depth_est['normal_c'][var][N],
                    depth_est['rpca'][var][N],
                    depth_est['rpca_c'][var][N]).T,columns=['Normal',r'Normal /w PB','RPCA','RPCA /w PB'])

                # plot the kernel desnsity estimate
                ax = sns.kdeplot(df)
                ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv}) N={N}")
                ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-{vv}-N-{N}.png"))
                plt.close(ax.figure)
                # plot with bars
                ax = sns.histplot(df,kde=True)
                ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv}) N={N}")
                ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-with-bars-{vv}-N-{N}.png"))
                plt.close(ax.figure)
                # plot with only bars
                ax = sns.histplot(df,kde=False)
                ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv}) N={N}")
                ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-only-bars-{vv}-N-{N}.png"))
                plt.close(ax.figure)
        
    plt.close('all')
    # save data
    json.dump(data,open(os.path.join(opath,f"depth-estimate-stats-rolling-correct-full-exp-{depth_exp}-window-{depth_win}-av-auto.json"),'w'),default=str)




def plot_depthest_error(data_json,true_depth,opath='',**kwargs):
    # load data JSON
    depth_est = json.load(open(data_json,'r'))
    # flag to indicate if it's the statistics or original depth estimate JSON
    is_stats=True
    if 'Torque' in depth_est['normal']:
        is_stats = False
        NN = [int(N) for N in list(depth_est['normal'][var].keys())]
    else:
        NN = [int(N) for N in depth_est['NN']]
    # get correction types
    corr_types = list(depth_est.keys())
    # find error between depth estimation and true depth
    # for each window size
    for i,N in enumerate(NN):
        error = []
        # both types of JSONs are organised by the same keys at the top level
        for kk,vv in depth_est.items():
            # stats JSON has depth est results in sub-lists for each window size
            if is_stats:
                error.append(np.abs(true_depth - vv['depth_est']['Torque'][i]))
            else:
                error.append(np.abs(true_depth - vv[kk]['Torque'][N]))
        # convert to a pandas dataframe
        err_df = pd.DataFrame(np.column_stack(error),columns=corr_types)
        # plot the kernel desnsity estimate
        ax = sns.kdeplot(err_df)
        ax.set(xlabel="Depth Estimation Error (mm)",title=f"{kk} Depth Estimate Error Distribution N={N}")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-error-kde-{vv}-N-{N}.png"))
        plt.close(ax.figure)
        # plot with bars
        ax = sns.histplot(err_df,kde=True)
        ax.set(xlabel="Depth Estimation Error (mm)",title=f"{kk} Depth Estimate Error Distribution N={N}")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-error-kde-with-bars-{vv}-N-{N}.png"))
        plt.close(ax.figure)
        # plot with only bars
        ax = sns.histplot(err_df,kde=False)
        ax.set(xlabel="Depth Estimation Error (mm)",title=f"{kk} Depth Estimate Error Distribution N={N}")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-error-only-bars-{vv}-N-{N}.png"))
        plt.close(ax.figure)

def replot_depthest_stats_json(data_json,opath=''):
    # load json file
    data = json.load(open(data_json,'r'))
    for kk in data.keys():
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
        fstats.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-stats-exp-{depth_exp}-window-{depth_win}-av-auto.png"))

        #### plot KDE for different window sizes
        for vv in ["Torque","Thrust"]:
            # form into a pandas dataframe
            df = pd.DataFrame(np.array(data[kk]['depth_est'][vv]).T,columns=[str(N) for N in NN])
            # plot the kernel desnsity estimate
            ax = sns.kdeplot(df)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-{vv}.png"))
            plt.close(ax.figure)
            # plot with bars
            ax = sns.histplot(df,kde=True)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-with-bars-{vv}.png"))
            plt.close(ax.figure)
            # plot with only bars
            ax = sns.histplot(df,kde=False)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-only-bars-{vv}.png"))
            plt.close(ax.figure)

            #### plot evaluation time
            df = pd.DataFrame(np.array(eval_time[kk]['Torque']).T,columns=[str(N) for N in NN])
            # plot the kernel desnsity estimate
            ax = sns.kdeplot(df)
            ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-kde-{vv}.png"))
            plt.close(ax.figure)
            # plot with bars
            ax = sns.histplot(df,kde=True)
            ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-kde-with-bars-{vv}.png"))
            plt.close(ax.figure)
            # plot with only bars
            ax = sns.histplot(df,kde=False)
            ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution ({vv})")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-only-bars-{vv}.png"))
            plt.close(ax.figure)

            # distribution plots showing the influence of different correction combinations
            for N in NN:
                df = pd.DataFrame(np.array(depth_est['normal'][vv][N],
                    depth_est['normal_c'][vv][N],
                    depth_est['rpca'][vv][N],
                    depth_est['rpca_c'][vv][N]).T,columns=['Normal',r'Normal /w PB','RPCA','RPCA /w PB'])

                # plot the kernel desnsity estimate
                ax = sns.kdeplot(df)
                ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv}) N={N}")
                ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-{vv}-N-{N}.png"))
                plt.close(ax.figure)
                # plot with bars
                ax = sns.histplot(df,kde=True)
                ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv}) N={N}")
                ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-with-bars-{vv}-N-{N}.png"))
                plt.close(ax.figure)
                # plot with only bars
                ax = sns.histplot(df,kde=False)
                ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution ({vv}) N={N}")
                ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-only-bars-{vv}-N-{N}.png"))
                plt.close(ax.figure)
        
    plt.close('all')

if __name__ == "__main__":
    # function for estimating the xstart parameter
    # comment out if you want to use your own value
    xstart = estimate_xstart('8B Life Test/*.xls')
    
    depth_est_run(path='8B Life Test/*.xls',    # where the data files are
                  NN=[10,20,30,40,50],          # rolling window sizes
                  plot_steps=True,              # flag to plot the signals after they've been filtered
                  add_empty=False,              # add the empty channel to signals
                  xstart=xstart+1,                  # search period from start of the signal to look for first reference point
                  depth_exp=40.0,               # expected depth. used in searching for second reference point
                  depth_win=5.0,                # search window in mm around 2nd reference point
                  default=True,                 # flag to default to certain values rather than raise exceptions
                  opath='8B Life Test/plots/test') # output path where files are saved
