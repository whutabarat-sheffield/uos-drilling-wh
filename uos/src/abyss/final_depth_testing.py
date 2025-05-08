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
import multiprocessing as mp

'''
    This script is for final testing of the algorithm before handing over the final result.
    This is different from the rolling_gradient sandbox script as it has fewer functions to worry about
'''

def plotDepthCorrectSC(*args,NN=[10,20,30,40,50],win=1.0):
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

        Returns figure
    '''
    # if the user provided a single argument and it's a string
    # treat it as the file path
    if (len(args)==1) and isinstance(args[0],str):
        # load file
        data = dp.loadSetitecXls(args[0],'auto_data')
        # get filename to use as axis titles
        fname = os.path.splitext(os.path.basename(args[0]))[0]
        xdata = np.abs(data['Position (mm)'].values)
        ydata = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
        # check there are multiple unique step codes
        sc = data['Step (nb)'].values
        sc_uq = np.unique(data['Step (nb)'])
    # if the user provided 3 vectors
    elif len(args)==3:
        # split into xdata,ydata and step code vector
        xdata,ydata,sc = args
        # ensure position is absoluted
        xdata = np.abs(xdata)
        sc_uq = np.unique(sc)
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
    # for each window size
    arr = None
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
        # draw arrow showing correction distance
        arr = tax.arrow(x=xdata[grad.argmax()],y=grad.max(),dx=dist,dy=0,color='green',head_width=0.05,shape='full',length_includes_head=False)
        all_dist.append(abs(dist))
    # get first lines from each
    lines = [ax[0].lines[0],tax.lines[0],arr]
    ax[0].legend(lines,['Torque','Gradient','Correction'],handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})
    ax[0].set(xlabel='Position (mm)',ylabel='Torque (A)',title='Data')
    f.suptitle(fname)

    ax[1].plot(NN,all_dist,'x')
    ax[1].set(xlabel="Window Size",ylabel="Distance (mm)",title="Window Distance")
    return f

def plotDepthCorrectedDist(path,NN=[10,20,30,40,50],win=1.0,opath='',no_empty=False):
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
            no_empty : Flag to not use the empty channel

        Returns plotted figure
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
    return f

def gridSearchDest(path,NN,depth_win,use_empty=False,**kwargs):
    '''
        Search a grid of rolling window sizes and search window sizes and calculate the rror

        The current error is relative to depth_exp at the moment.

        The results from each file are stacked to form a 3D matrix. For a given combination of values,
        the proportion of values that is within 0.05mm of the depth_exp is calculated, colour coded and displayed

        Inputs:
            path : Wildcard path to data files
            NN : Iterable of rolling window sizes
            depth_win : Iterable of depth search windows
            use_empty : Load empty channel
            depth_exp : Expected depth in mm. Default 20.0.
            th        : Threshold from depth_exp. Default 0.1mm
    '''
    N_grid = np.array(NN)
    dw_grid = np.array(depth_win)
    depth_est = np.zeros((N_grid.shape[0],dw_grid.shape[0]))
    r,c = depth_est.shape
    depth_est_stack = []
    nf = len(glob(path))
    for fn in glob(path):
        data = dp.loadSetitecXls(fn,"auto_data")
        xdata = np.abs(data['Position (mm)'].values)
        ydata = data['I Torque (A)'].values
        sc = data['Step (nb)']
        if use_empty:
            ydata += data['I Torque Empty (A)'].values

        for i,N in enumerate(NN):
            for j,dw in enumerate(depth_win):
                depth_est[i,j] = depth_est_rolling(ydata,xdata,NA=N,depth_win=dw,depth_exp=kwargs.get("depth_exp",20.0),correct_dist=False,step_vector=sc)
        depth_est_stack.append(depth_est)

    dexp = kwargs.get("depth_exp",20.0)
    th = kwargs.get("th",0.1)
    depth_est_stack = np.dstack(depth_est_stack)
    prob_stack = {}
    # calculate prob of a given combination achieving a certain accuracy
    prob = np.zeros((N_grid.shape[0],dw_grid.shape[0]))
    # <= 0.05
    for i in range(r):
        for j in range(c):
            vector = depth_est_stack[i,j]
            # check by distance from expected depth
            ii = np.where((vector-dexp)<=th)[0]
            prob[i,j] = len(ii)/nf

    f,ax = plt.subplots(constrained_layout=True,figsize=(12,8))
    img = ax.imshow(prob,cmap='hot')
    ax.set_xticks(range(len(depth_win)),labels=[f"{x:.2f}" for x in depth_win])
    ax.set_yticks(range(len(NN)),labels=[f"{x:d}" for x in NN])
    ax.set(xlabel="Depth Window (mm)",ylabel="Rolling Window Size (samples)")
    cbar = plt.colorbar(img)
    cbar.ax.set_ylabel(f'Proportion of values <={th}')
    

    f,ax = plt.subplots()
    sns.histplot(prob.flatten(),binwidth=0.1,ax=ax,kde=True)
    ax.set_title(f"Pribability Distribution th<={th}")
    plt.show()
    
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
                        sc_uq = sc.unique()
                        sc_min = [sc == sc_uq[1]].index().min()
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
    xA -= dist_corr
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
                    return (xmax - xendB) - xA
                elif end_ref == 'start':
                    warnings.warn(f"All empty mask for 2nd window!\nxendA={xendA},xendB={xendB},end_ref={end_ref},xmax={xmax}. Defaulting to {(xmax + xendB) -xA}",category=EmptyMaskWarning)
                    return (xmax + xendB) -xA
            # if specified from expected depth estimate
            # take as upper end of the window period
            elif 'depth_exp' in kwargs:
                warnings.warn(f"All empty mask for 2nd window depth_exp={depth_exp}, win={depth_win}, xA={xA}!\nDefaulting to {xA+depth_exp + (depth_win/2)}",category=EmptyMaskWarning)
                #xB = xA+depth_exp+(win/2)
                return depth_exp+(depth_win/2) - xA
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
                pkB = grad_mask[pks].argmax()
                # find correspondng x value
                return xmask[pks][pkB] - xA
            # get last peak if specified or as part of limit
            elif (pselect == 'limit') or (pselect == 'last'):
                return xmask[pks][-1] - xA
            # use first peak
            elif pselect == 'first':
                return xmask[pks][0] - xA
            # if unsupported default to argmax
            else:
                if default:
                    warnings.warn(f"Unsupported peak selection mode {pselect}. Defaulting to argmax for 2nd window")
                    pkB = grad_mask[pks].argmax()
                    # find correspondng x value
                    return grad_mask[pks][pkB] - xA
                else:
                    raise ValueError(f"Unsupported peak selection mode {pselect}!")

def esimate_xstart(path,proc=np.max):
    '''
        Function for estimating the xstart parameter of a dataset

        xstart is the distance from the start where the depth_est_rolling searches for the first reference point

        The first reference point is typically where the torque first increases as it enters the material.

        When it first enters the material, it also typically changes program step (0 -> 1). Therefore, a way to estimate
        the xstart parameter is to look at the distance from the start to the first program step change.

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
            xstart        Search period from start of the signal to look for first reference point
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
    # list of step indicies
    sc_mins = []
    # list of pos
    pos = []
    # number of files
    nf = len(glob(path))
    if plot_steps:
        input(f"WARNING: With plot_steps True, this function will produce approx. {((3*len(NN))+5)*(((len(NN)*2)+2)*nf)+(2*len(NN)+5)} plots in {opath}! Do you still want to continue? If not, reset the IDLE or press CTRL+C to exit now")
    # minimum length
    min_len = None
    print("Creating data files")
    if plot_steps:
        paths_list = list(glob(path))
    # iterate over each of the files
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
        # get base data
        ydata = data[f'I Torque (A)'].values.flatten()
        # add empty
        if add_empty:
            if f'I Torque Empty (A)' in data:
                ydata += data[f'I Torque Empty (A)'].values
        # add to array
        dt_torque.append(ydata)

    # make copies of arrays
    dt_torque_clip = dt_torque.copy()
    pos_clip = pos.copy()
    # clip arrays
    print(f"clipping arrays to {min_len}")
    for i in range(nf):
        if dt_torque_clip[i].shape[0]>min_len:
            dt_torque_clip[i] = dt_torque_clip[i][:min_len]
            pos_clip[i] = pos_clip[i][:min_len]    
    # stack them to form a 2D matrix
    dt_torque_arr = np.column_stack(dt_torque_clip)
    # process using RPCA
    print("processing using RPCA")
    L,S = R_pca(dt_torque_arr).fit(max_iter=10000)
    np.savez_compressed(os.path.join(opath,"RPCA-torque.npz"),L=L,S=S)
    print(f"finished ",L.shape)
    all_dist_rpca = []
    all_dist = []
    arr = None
    arr_rpca = None
    dep = None

    for i in range(nf):
        # get data vectors
        xdata = pos_clip[i]
        ydata = L[:,i]
        ogxdata = pos[i]
        ogydata = dt_torque[i]
        # get index where program transitions
        sc_min = sc_mins[i]
        # plot the data and intermediate steps of correcting the data
        if plot_steps:
            # get filename for plotting and saving
            fname = os.path.splitext(os.path.basename(paths_list[i]))[0]
            # if there's a valid sc index
            if sc_min:
                frpca,axrpca = plt.subplots(constrained_layout=True,ncols=2)
                # make a twin axis to plot gradient against
                taxrpca = axrpca[0].twinx()
                # get where the program changes
                # index is early in the signal so to same location can be used for both RPCA and non-rpca
                dep = xdata[sc_min]
                # mask to small window around here
                mask = (xdata>= (dep-1.0)) & (xdata<= (dep+1.0))
                xfilt_rpca = xdata[mask]
                # get torque values for target range
                data_filt_rpca = ydata[mask]
                # plot the target window
                axrpca[0].plot(xfilt_rpca,data_filt_rpca,'b-')
                # add vertical line for transition point
                #axrpca[0].vlines(dep,0.9*data_filt_rpca.min(),1.1*data_filt_rpca.max(),colors=['k'])
                axrpca[0].vlines(dep,0.0,10.0,colors=['k'])
                all_dist_rpca.clear()

                # make plots for the non-rpca data
                f,ax = plt.subplots(ncols=2,constrained_layout=True)
                tax = ax[0].twinx()
                # mask to small window
                mask = (ogxdata>= (dep-1.0)) & (ogxdata<= (dep+1.0))
                xfilt = ogxdata[mask]
                # get torque values for target range
                data_filt = ogydata[mask]
                ax[0].plot(xfilt,data_filt,'b-')
                #ax[0].vlines(dep,0.9*data_filt.min(),1.1*data_filt.max(),colors=['k'])
                ax[0].vlines(dep,0.0,10.0,colors=['k'])
                all_dist.clear()

            for N in NN:
                #### RPCA DATA ####
                # filter + smooth
                if sc_min:
                    torque_filt = wiener(data_filt_rpca,N)
                    grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
                    dist = dep-xfilt_rpca[grad.argmax()]
                    # plot the narrow window gradient
                    taxrpca.plot(xfilt_rpca,grad,'r-')
                    # draw an arrow going from the peak in the gradient that would be used to where the new location is
                    arr_rpca = taxrpca.arrow(x=xfilt_rpca[grad.argmax()],y=grad.max(),dx=dist,dy=0,color='green',head_width=0.05,shape='full',length_includes_head=False)
                    all_dist_rpca.append(abs(dist))

                # plot data + gradient
                fdt,axdt = plt.subplots(constrained_layout=True)
                axdt.plot(xdata,ydata,'b-',label='Original')
                # filter entire data vector
                torque_filt = wiener(ydata,N)
                grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
                pks,_ = find_peaks(grad,height=grad.max()*0.1)
                # plot the filtered data
                axdt.plot(xdata,torque_filt,'r-',label="Filtered")
                # create twin axis for gradient
                cax = axdt.twinx()
                # plot gradient
                cax.plot(xdata,grad,'k-',label="Rolling Gradient")
                # mark peaks
                cax.plot(xdata[pks],grad[pks],'gx',markersize=10,markeredgewidth=4,label="Peaks")
                # set labels
                axdt.set(xlabel="Position (mm)",ylabel="Torque (A)")
                cax.set_ylabel("Rolling Gradient")
                # combine legends together
                lines, labels = axdt.get_legend_handles_labels()
                lines2, labels2 = cax.get_legend_handles_labels()
                cax.legend(lines + lines2, labels + labels2, loc=0)
                # set title
                fdt.suptitle(f"{fname} Torque\nRPCA + Wiener Filtered Rolling Gradient N={N}")
                fdt.savefig(os.path.join(opath,f"{fname}-Torque-rpca-wiener-tukey-rolling-gradient-pks-N-{N}.png"))
                plt.close(fdt)

                #### NON-RPCA DATA ####
                # filter + smooth
                if sc_min:
                    torque_filt = wiener(data_filt,N)
                    grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
                    dist = dep-xfilt[grad.argmax()]
                    # plot the narrow window gradient
                    tax.plot(xfilt,grad,'r-')
                    # draw an arrow going from the peak in the gradient that would be used to where the new location is
                    arr = tax.arrow(x=xfilt[grad.argmax()],y=grad.max(),dx=dist,dy=0,color='green',head_width=0.05,shape='full',length_includes_head=False)
                    all_dist.append(abs(dist))

                # plot data + gradient
                fdt,axdt = plt.subplots(constrained_layout=True)
                axdt.plot(ogxdata,ogydata,'b-',label='Original')
                # filter entire data vector
                torque_filt = wiener(ogydata,N)
                grad = rolling_gradient(torque_filt,N) * tukey(len(torque_filt),0.1,True)
                pks,_ = find_peaks(grad,height=grad.max()*0.1)
                # plot the filtered data
                axdt.plot(ogxdata,torque_filt,'r-',label="Filtered")
                # create twin axis for gradient
                cax = axdt.twinx()
                # plot gradient
                cax.plot(ogxdata,grad,'k-',label="Rolling Gradient")
                # mark peaks
                cax.plot(ogxdata[pks],grad[pks],'gx',markersize=10,markeredgewidth=4,label="Peaks")
                # set labels
                axdt.set(xlabel="Position (mm)",ylabel="Torque (A)")
                cax.set_ylabel("Rolling Gradient")
                # combine legends together
                lines, labels = axdt.get_legend_handles_labels()
                lines2, labels2 = cax.get_legend_handles_labels()
                cax.legend(lines + lines2, labels + labels2, loc=0)
                # set title
                fdt.suptitle(f"{fname} Torque\nRPCA + Wiener Filtered Rolling Gradient N={N}")
                fdt.savefig(os.path.join(opath,f"{fname}-Torque-no-rpca-wiener-tukey-rolling-gradient-pks-N-{N}.png"))
                plt.close(fdt)

            # get first lines from each
            if sc_min:
                lines = [axrpca[0].lines[0],taxrpca.lines[0],arr]
                axrpca[0].legend(lines,['Torque','Gradient','Correction'],handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})
                axrpca[0].set(xlabel='Position (mm)',ylabel='Torque (A)')
                
                # plot the distances
                axrpca[1].plot(NN,all_dist,'x',markersize=10,markeredgewidth=5)
                axrpca[1].set_xticks(NN)
                axrpca[1].set(xlabel="Rolling Window Size (Samples)",ylabel="Correction Distance (mm)")
                frpca.suptitle(fname)
                frpca.savefig(os.path.join(opath,f"{fname}-rpca-and-depth-correct-sc.png"))
                plt.close(frpca)

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
            plt.close('all')
        # iterate over different window sizes
        for N in NN:
            # estimate depth using RPCA data
            start = perf_counter()
            dest = depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect)
            end = perf_counter() - start
            # save result in dictionary
            if not (N in depth_est['rpca']):
                depth_est['rpca'][N] = []
                eval_time['rpca'][N] = []
            depth_est['rpca'][N].append(float(dest))
            eval_time['rpca'][N].append(float(end))
            # estimate depth using RPCA data + correction
            # save result in dictionary
            if not (N in depth_est['rpca_c']):
                depth_est['rpca_c'][N] = []
                eval_time['rpca_c'][N] = []
            if sc_min:
                start = perf_counter()
                dest = depth_est_rolling(ydata,xdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect,
                                         correct_dist=True,change_idx=sc_min)
                end = perf_counter() - start
    
                depth_est['rpca_c'][N].append(float(dest))
                eval_time['rpca_c'][N].append(float(end))
            else:
                depth_est['rpca_c'][N].append(None)
                eval_time['rpca_c'][N].append(None)
            #print(f"Processing {var} NORMAL {N}")
            # estimate depth using ORIGINAL data
            start = perf_counter()
            dest = depth_est_rolling(ogydata,ogxdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect)
            end = perf_counter() - start
            # save result in dictionary
            if not (N in depth_est['normal']):
                depth_est['normal'][N] = []
                eval_time['normal'][N] = []
            depth_est['normal'][N].append(float(dest))
            eval_time['normal'][N].append(float(end))
            # estimate depth using ORIGINAL data with correction
            # save result in dictionary
            if not (N in depth_est['normal_c']):
                depth_est['normal_c'][N] = []
                eval_time['normal_c'][N] = []
            if sc_min:
                start = perf_counter()
                dest = depth_est_rolling(ogydata,ogxdata,NA=N,xstart=xstart,depth_exp=depth_exp,depth_win=depth_win,default=default,end_ref=end_ref,pselect=pselect,
                                              correct_dist=True,change_idx=sc_min)
                end = perf_counter() - start
                
                depth_est['normal_c'][N].append(float(dest))
                eval_time['normal_c'][N].append(float(end))
            else:
                depth_est['normal_c'][N].append(None)
                eval_time['normal_c'][N].append(None)
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
    # define JSON to save
    data = {kk : {'depth_est' : [], 'mean' : [], 'var' : [], 'std' : []} for kk in depth_est.keys()}
    data['NN'] = NN
    # iterate over each type of estimate
    for kk,dest in depth_est.items():
        print(f"Plotting {kk} results")
        # iterate over each window size
        for N in NN:
            # get torque and thrust depth estimates for window size
            dest_torque = dest[N]
            # create axes
            f,ax = plt.subplots(constrained_layout=True)
            # plot torque depth estimate with blue X's
            ax.plot(dest_torque,'bx',label="Torque")
            # process and add results to the new JSON
            data[kk]['depth_est'].append([float(x) for x in dest_torque])
            data[kk]['mean'].append(float(np.mean(dest_torque)))
            data[kk]['var'].append(float(np.var(dest_torque)))
            data[kk]['std'].append(float(np.std(dest_torque)))
            # draw a black line for nominal depth
            ax.plot(nf*[depth_exp,],'k-',label="Nominal")
            # create legend
            ax.legend()
            ax.set(xlabel="Hole Number",ylabel="Depth Estimate (mm)",title=f"{kk.capitalize()} Depth Estimate N={N}")
            # save figure
            f.savefig(os.path.join(opath,f"{kk}-depth-estimates-rolling-correct-full-N-{N}-depth-exp-{depth_exp}-window-{depth_win}.png"))
            plt.close(f)
      
        # create axes for statistics    
        fstats,axstats = plt.subplots(ncols=3,sharex=True,constrained_layout=True,figsize=(14,12))
        # plot the statistics of each window size
        axstats[0].plot(NN,data[kk]['mean'],'b-')
        axstats[1].plot(NN,data[kk]['var'],'r-')
        axstats[2].plot(NN,data[kk]['std'],'k-')
        
        axstats[0].set(xlabel="Window Size",ylabel="Mean Depth Estimate",title="Mean Depth Est. (Torque)")
        axstats[1].set(xlabel="Window Size",ylabel="Var Depth Estimate",title="Var Dev Depth Est. (Torque)")
        axstats[2].set(xlabel="Window Size",ylabel="Std Depth Estimate",title="Std Dev Depth Est. (Torque)")
        fstats.suptitle(f"{kk.capitalize()} Depth Est. Stats using dexp={depth_exp}, win={depth_win}")
        fstats.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-stats-exp-{depth_exp}-window-{depth_win}.png"))

        #### plot KDE for different window sizes
        # form into a pandas dataframe
        df = pd.DataFrame(np.column_stack(data[kk]['depth_est']),columns=[str(N) for N in NN])
        # plot the kernel desnsity estimate
        ax = sns.kdeplot(df)
        ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution (Torque)")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-Torque.png"))
        plt.close(ax.figure)
        # plot with bars
        ax = sns.histplot(df,kde=True)
        ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution (Torque)")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-with-bars-Torque.png"))
        plt.close(ax.figure)

        #### plot evaluation time
        df = pd.DataFrame(np.column_stack(list(eval_time[kk].values())),columns=[str(N) for N in NN])
        # plot the kernel desnsity estimate
        ax = sns.kdeplot(df)
        ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution (Torque)")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-kde-Torque.png"))
        plt.close(ax.figure)
        # plot with bars
        ax = sns.histplot(df,kde=True)
        ax.set(xlabel="Evaluation Time (s)",title=f"{kk} Depth Estimate Eval Time Distribution (Torque)")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-eval-time-kde-with-bars-Torque.png"))
        
        # distribution plots showing the influence of different correction combinations at a given window size
        for N in NN:
            df = pd.DataFrame(np.column_stack([depth_est['normal'][N],
                depth_est['normal_c'][N],
                depth_est['rpca'][N],
                depth_est['rpca_c'][N]]),columns=['Normal',r'Normal /w PB','RPCA','RPCA /w PB'])

            # plot the kernel desnsity estimate
            ax = sns.kdeplot(df)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution (Torque) N={N}")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-Torque-N-{N}.png"))
            plt.close(ax.figure)
            # plot with bars
            ax = sns.histplot(df,kde=True)
            ax.set(xlabel="Depth Estimation (mm)",title=f"{kk} Depth Estimate Distribution (Torque) N={N}")
            ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-kde-with-bars-Torque-N-{N}.png"))
            plt.close(ax.figure)
        
    plt.close('all')
    # save data
    json.dump(data,open(os.path.join(opath,f"depth-estimate-stats-rolling-correct-full-exp-{depth_exp}-window-{depth_win}-av-auto.json"),'w'),default=str)

def loadJSONToPanda(path):
    '''
        Load depth estimate JSON file created by depth_est_run into a Pandas dataframe

        Doesn't work with the depth est statistics JSON files.

        Compatible with previous JSON files.

        Pandas DataFrame columns are named by

        type_var_N

        where:
            type : Type of depth estimate e.g. normal, normal_c etc.
            var : Variable (either Torque or Thrust). Used with previous JSON versions
            N : Rolling window size

        Inputs:
            path : Path to depth estimate JSON

        Returns Pandas DataFrame
    '''
    # load json
    data = json.load(open(path,'r'))
    # make an empty dataframe
    df = pd.DataFrame()
    is_stats = False
    vals = data[list(data.keys())[0]]
    if isinstance(vals,dict):
        vals = vals['Torque']
    NN = [int(n) for n in vals.keys()]
    # iterate over items
    for k,v in data.items():
        if k == 'NN':
            continue
        # if it's a sub dictionary
        # then it's organised by variable
        # this is for backwards compatibility with previous versions
        if isinstance(v,dict):
            # iterate over variable and depth
            for var,dd in v.items():
                if isinstance(dd,dict):
                    for N,d in dd.items():
                        df[f"{k}_{var}_{N}"] = [float(i) for i in d]
                else:
                    for N,d in v.items():
                        df[f"{k}_{var}_{N}"] = [float(i) for i in d]
        else:
            for N,d in v.items():
                df[f"{k}_{N}"] = [float(i) for i in d]
    return df

def plot_depthest_error(data_json,true_depth,opath='',**kwargs):
    '''
        Plot the depth estimate RMS Error

        Loads JSON depth estimate file and plot the fitting error against ehe provided vector of depth estimates (true_depth).

        Plots are generated and saved to opath folder showing the distribution of the error at different window sizes, which are
        extracted from the dictionary.

        Inputs:
            data_json : Path to JSON file generated by depth_est_run. Can be either the statistics or depth estimate one.
            true_depth : Vector of true depth.
            opath : Output path to where the plots are generated
    '''
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
                error.append(np.abs(true_depth - vv['depth_est'][i]))
            else:
                error.append(np.abs(true_depth - vv[kk][N]))
        # convert to RMSE
        error = np.array(error)
        error = np.sqrt(np.mean(np.power(error,2)))
        # convert to a pandas dataframe
        err_df = pd.DataFrame(np.column_stack(error),columns=corr_types)
        # plot the kernel desnsity estimate
        ax = sns.kdeplot(err_df)
        ax.set(xlabel="Depth Estimation Error (mm)",title=f"{kk} Depth Estimate Error Distribution N={N}")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-error-kde-{vv}-N-{N}.png"))
        plt.close(ax.figure)
        # plot with bars
        ax = sns.histplot(err_df,kde=True)
        ax.set(xlabel="RMSE (mm)",title=f"{kk} Depth Estimate Error Distribution N={N}")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-error-kde-with-bars-{vv}-N-{N}.png"))
        plt.close(ax.figure)
        # plot with only bars
        ax = sns.histplot(err_df,kde=False)
        ax.set(xlabel="RMSE (mm)",title=f"{kk} Depth Estimate Error Distribution N={N}")
        ax.figure.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-error-only-bars-{vv}-N-{N}.png"))
        plt.close(ax.figure)

def replot_depthest_stats_json(data_json,opath=''):
    '''
        Re-create the depth statistics plots from the statistics JSON file

        Inputs:
            data_json : Path to statistics json file
            opath : Output directory path for the plots
    '''
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
        fstats.suptitle(f"{kk.capitalize()} Depth Est. Stats using dexp={depth_exp}, win={depth_win}")
        fstats.savefig(os.path.join(opath,f"{kk}-rolling-correct-depth-estimate-stats-exp-{depth_exp}-window-{depth_win}.png"))

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
    #for fn in glob('8B Life Test/*.xls'):
    #    plotDepthCorrectSC(fn)
    #    plt.show()

    # function for estimating the xstart parameter
    # comment out if you want to use your own value
    xstart = esimate_xstart('8B Life Test/*.xls')
    df = loadJSONToPanda(r"C:\Users\david\Downloads\depth-estimates-rolling-correct-full-exp-40.0-window-5.0-av-auto-per-0.00.json")
    #gridSearchDest('8B Life Test/*.xls',[10,20,30,40,50],depth_win=np.arange(0.1,2.0,0.1))
    
##    depth_est_run(path='8B Life Test/*.xls',    # where the data files are
##                  NN=[10,20,30,40,50],          # rolling window sizes
##                  plot_steps=True,              # flag to plot the signals after they've been filtered
##                  add_empty=False,              # add the empty channel to signals
##                  xstart=xstart+1,                  # search period from start of the signal to look for first reference point
##                  depth_exp=40.0,               # expected depth. used in searching for second reference point
##                  depth_win=5.0,                # search window in mm around 2nd reference point
##                  default=True,                 # flag to default to certain values rather than raise exceptions
##                  opath='8B Life Test/plots/test', # output path where files are saved
##                  try_logic=True)               # flag to try new logic
    # if try_logic is True
    # then the depth estimate logic is abs((xA - depth_corr) - xB)
    # else the depth estimate logic is abs(xA- xB) + depth_corr
