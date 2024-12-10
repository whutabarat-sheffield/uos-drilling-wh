import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer
from scipy.fft import fft,fftshift,rfft, rfftfreq
from scipy.signal import welch, wiener, get_window, find_peaks, peak_widths
from scipy.signal.windows import tukey
from scipy.signal._arraytools import axis_slice
from scipy.signal._savitzky_golay import _polyder
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import NonlinearConstraint
import pywt
import scaleogram as scg
import os
from itertools import product as iproduct
import warnings
from tslearn.metrics import dtw, dtw_path
import math
# import abyss.uos_depth_est_core as uos
import uos_depth_est_core as uos # if this does not work please try the above

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

class ToolShapelet:
    def __init__(self,dims):
        '''
            Initialise the shapelet from the given tool lengths

            Inputs:
                dims : Lengths of the tool segments starting from the tip in mm
        '''
        # input check
        if any([dd<=0 for dd in dims]):
            raise ValueError("Tool dimensions cannot be less than or equal to 0!")
        self._dims = dims
        self._tl = math.fsum(self._dims)
        self._av = None
        self._scale = 1.0
        # data vectors representing tool
        self._pos = np.empty((0,))
        self._time = np.empty((0,))
        self._signal = np.empty((0,))

    def tlength(self):
        ''' Return the tool length by summing the tool lengths together '''
        return self._tl

    def generate(self,av,scale=1.0,sf=100.0,mirror=False):
        '''
            Create the data for each of the slopes based off the given parameters

            The shapelet is assumed to start from the tip so each of the segments
            alternates between slope, flat, slope etc.

            The y-axis is designed to represent the signal the tool shapelet is compared to.
            The signal data is initially generated on the scale 0-1. The generated signal data
            is multiplied by the parameter scale to adjust it to a desired range.

            The parameter AV refers to feed rate of the tool, essentially the tool velocity.
            This affects the time and position data controlling the duration between tool segments.

            Inputs:
                av : Advance rate of the tool in mm/s
                scale : Max value the signal reaches. Default 1.0
                sf : Sample rate (Hz). Default 100.0
                mirror : Mirror the shapelet as if the tool is exiting a material.

            Return generated time vector, position vector and signal vector
        '''
        if av<=0:
            raise ValueError(f"Advance Rate cannot be less than or equal to 0! Received {av}!")
        if scale<=0:
            raise ValueError(f"Scale cannot be less than or equal to 0! Received {scale}!")
        if sf<=0:
            raise ValueError(f"Sampling Rate cannot be less than or equal to 0! Received {sf}!")
        # store av
        self._av = av
        self._scale = scale
        # create the unique signal points skipping first one as it's always our starting value
        pks = np.linspace(float(mirror),float(not mirror),len(self._dims))[1:].tolist()
        # last max time
        tmax = 0.0
        # vectors to update
        self._time = np.empty((0,))
        self._signal= np.empty((0,))
        # inverse sample rate
        isf = 1./sf
        # iterate over each tool section
        for ii,dd in enumerate(self._dims):
            # create time vector for the section
            td = np.arange(tmax+isf,tmax+(dd/av),isf)
            self._time = np.append(self._time,td,axis=0)
            # if an even number or 0 then we want the signal to slope
            if (ii%2)==0:
                # set end value for section
                sm = pks.pop(0)
                sd = np.linspace(float(mirror) if ii==0 else self._signal[-1],sm,td.shape[0])
            # if an odd number then we want a flat section
            else:
                sd = sm*np.ones(td.shape[0])
            self._signal = np.append(self._signal,sd,axis=0)
            # update max time
            tmax = self._time.max()
        # create position vector
        self._pos = av*self._time
        self._signal *= scale
        return self._time,self._pos,self._signal

    def draw(self,ax=None,**kwargs):
        '''
            Draw the last generated tool shapelet

            On either the given axis or a generated axis, the tool shapelet position and signal
            data is plotted. In addition, lines are drawn with text labels describing the length of each
            section

            Inputs:
                input : Axis to draw on. If None, a new figure and axis are generated. Default None.
                xlabel : Text label for x-axis. Default Position (mm)
                ylabel : Text label for y-axis. Default Signal.
                title : Text title for axis. Default f"Tool Shapelet, AV={self._av}"

            Returns figure object
        '''
        # if the time vector is empty
        if self._time.shape[0] == 0:
            raise ValueError("No tool shapelet has been generated!")
        # if no axis has been given
        # make one
        if ax is None:
            f,ax = plt.subplots()
        else:
            f = ax.figure
        # plot position and signal
        ax.plot(self._pos,self._signal,'b-')
        smin = self._signal.min()
        dmax = 0.0
        # draw lines  with text indicating the distance of each section
        for dd in self._dims:
            smin = self._signal[np.argmin(np.abs(self._pos-(dmax+(dd/2))))]
            ax.plot((dmax,dmax+dd),(smin,smin),'k-')
            ax.text(dmax+(dd/2),smin,f"{dd:.2f}",horizontalalignment='center',verticalalignment='bottom')
            dmax = dmax+dd
        ax.set(xlabel=kwargs.get("xlabel","Position(mm)"),ylabel=kwargs.get("ylabel","Signal"),title=kwargs.get("title",f"Tool Shapelet, AV={self._av}"))
        # return figure object
        return f

# adapted from https://stackoverflow.com/a/64723729
class RollingPosWindow(BaseIndexer):
    def __init__(self,val,width,default=True,key='pos'):
        '''
            Custom indexer for creating variable sized windows based on position.
            This is assumed to be applied on a Series/column which contains position values

            It is recommended you port the data you want to a smaller dataframe and apply this
            as it's faster that way.

            Default is a flag to force certain settings that seem to work well. By forcing certain
            known settings it reduces the syntax and speeds up the function call as there are fewer checks.
            
            The default settings are:
                min_periods=0
                center = True
                closed = 'both'

            e.g. applying max to a window size of 4mm
            # load file
            data = loadSetitecXls(path,"auto_data")
            # extract torque data
            tq = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
            # create sub dataframe
            search = pd.DataFrame.from_dict({'pos' : data['Position (mm)'].values.flatten(),'signal' : tq})
            # use class to create the indicies of 4mm (set using width)
            # center set to True means the window centred
            rrmax = search.signal.rolling(RollingPosWindow(search.pos,width=4.0)).max()
            
            Inputs:
                val : Pandas Series or DataFrame containing position value
                width : Window width in mm
                default : Flag to force certain settings when running. Default True
                key : Key to use if val is a DataFrame to get position values
        '''
        # abs position values to ensure the values are all +ve
        # functions perfer it that way
        if isinstance(val,pd.Series):
            self.val = np.abs(val.values)
        else:
            self.val = np.abs(val[key].values)
        self.width = width
        # pre-calculate half window to save time
        self.__hw = width/2
        self.window_size = 0
        self.default = default

    def get_window_bounds(self, num_values, min_periods, center, closed):
        '''
            Retrieve the indicies for the windows bounds according to class settings

            Required to be used in Pandas rolling window

            Inputs:
                num_values : Total number of data points
                min_periods : Minimum size of periods. If None it is defaulted to 0.
                center : Flag indicating if the window should be centered for the calculation
                closed : Flag controlling how it should be closed
        '''
        if self.default:
            ix0 = np.searchsorted(self.val, self.val + -self.__hw, side='left')
            return ix0,np.maximum(np.searchsorted(self.val, self.val + self.__hw, side='right'), ix0)
        else:
            # handler for when min periods isn't specified
            if min_periods is None: min_periods = 0
            # default closed to left
            if closed is None: closed = 'left'
            # if center is True
            # set window from centre +/- width
            w = (-self.__hw, self.__hw) if center else (0, self.width)
            # set the sides of the window based on how the window size is performed
            side0 = 'left' if closed in ['left', 'both'] else 'right'
            side1 = 'right' if closed in ['right', 'both'] else 'left'
            # get indicies for edges of the windows
            # sort self.val as if it included self.val+w[0] and return indices
            ix0 = np.searchsorted(self.val, self.val + w[0], side=side0)
            ix1 = np.searchsorted(self.val, self.val + w[1], side=side1)
            ix1 = np.maximum(ix1, ix0 + min_periods)
            return ix0, ix1

class R_pca:
    '''
        Class for performing RPCA on a column-wise set of signals

        Processes the signals creating a Low-rank set of signal containing
        the shared behaviour between them and the Scatter set representing
        what was different in each signal that made it stand out.

        Example of use
        ---------------------
        from dataparser import loadSetitecXls
        from glob import glob
        
        dt_torque = []
        min_len = None
        # load signals into a list
        for fn in glob(path):
            data = loadSetitecXls(fn,"auto_data")
            dt_torque.append(data['I Torque (A)'].values.flatten())
            if min_len is None:
                min_len = len(dt_torque[-1])
            else:
                min_len = min(dt_torque[-1],min_len)
        # clip signals to a common length
        dt_torque = [tq[:min_len] for tq in dt_torque]
        # stack into a 2D matrix
        dt_torque_arr = np.column_stack(dt_torque)
        # process using RPCA
        L_torque,S_torque = R_pca(dt_torque_arr).fit(max_iter=10000)
    '''
    def __init__(self, D, mu=None, lmbda=None):
        '''
            Constructor for RPCA class

            Inputs:
                D : Matrix to perform RPCA on.
                mu,lmbda : Used in fitting
        '''
        self.D = D
        self.n, self.d = self.D.shape
        # initialie matrix for fitting
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))
        self.__fitted = False

    # flag to indic
    def is_fitted(self):
        return self.__fitted

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, 'fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=None):
        '''
            Fit RPCA to the data given earlier

            Inputs:
                tol : Tolerance for fitting. If None, set to result of frobenius_norm on D
                max_iter : Max number of iterations to try. Default 1000
                iter_print : At what multiple of iterations a print statement is made. If None,
                            then nothing is printed. Default None.

            Returns low-rank and scatter matricies of signals.
        '''
        # flag for printing statements
        do_print = True
        # if iter_print is None set flag to false
        if iter_print is None:
            do_print = False
        # else check the value given by user
        else:
            if iter_print<=0:
                raise ValueError(f"Printer Iter must be +ve! Received {iter_print}")
        # current iteration
        it = 0
        # error to fitting
        err = np.Inf
        # matrices
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)
        # tolerance for error
        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)
        # reset fitted flag
        self.__fitted = False
        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and it < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)                            #this line implements step 3
            Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)             #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                      #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            it += 1
            # print debug message if allowed
            if do_print:
                if (it % iter_print) == 0 or it == 1 or it > max_iter or err <= _tol:
                    print('itation: {0}, error: {1}'.format(it, err))
        self.__fitted = True
        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True, show=True):
        '''
            Plot the fitted data on a series of subplots and individual figures

            Inputs:
                size : Size of the matrix to plot. If None, calculated from previous matrix.
                tol : Additional tolerance on y-axis limits. Default 0.15
                axis_on: Flag to leave axis visible. Default True.
                show: Flag to show the results as it's being built. This is useful for inspection
                    but if you want each figure Default True.
        '''
        # check if it's been fitted
        if not self._fitted:
            raise ValueError("RPCA not fitted! Run fit first")
        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(self.n))
            nrows = int(sq)
            ncols = int(sq)
        # create the y axis limits
        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        # create the max number of subplots
        #numplots = np.min([self.n, nrows * ncols])
        f,ax = plt.subplots(nrows=nrows,ncols=ncols)
        # iterate over each axes
        for aa in ax:
            # update the y limits
            aa.set_ylim((ymin - tol, ymax + tol))
            # plot L + S as red line
            aa.plot(self.L[self.n, :] + self.S[self.n, :], 'r')
            # plot just L as blue line
            aa.plot(self.L[self.n, :], 'b')

            # create a separate figure for the same data
            ff,bb = plt.subplots()
            bb.plot(self.L[self.n, :] + self.S[self.n, :], 'r')
            bb.plot(self.L[self.n, :], 'b')
            # if show is True when display all generated figures
            if show:
                plt.show()
            if not axis_on:
                plt.axis('off')

# from https://stackoverflow.com/questions/67997826/pandas-rolling-gradient-improving-reducing-computation-time
def get_slope(df):
    '''
        Function to calculate the slope of a DataFrame by fitting a 1d polynomial and returning the gradient parameter

        Inputs:
            df : DataFrame to calculate the gradient for

        Returns the floating point gradient
    '''
    # drop na values
    df = df.dropna()
    # find the minimum index
    min_idx = df.index.min()
    x = df.index - min_idx
    y = df.values.flatten()
    slope,_ = np.polyfit(x,y,1)
    return slope

def get_slope_v2(df):
    '''
        Function to calculate the slope of a DataFrame by using the first and last point of the DataFrame.

        Bit faster than the v1 and the results are similar.

        Inputs:
            df : DataFrame to calculate the gradient across

        Returns the floating point gradient
    '''
    # drop na values
    df = df.dropna()
    min_idx = df.index.min()
    max_idx = df.index.max()
    return (df.values.flatten()[-1] - df.values.flatten()[0])/(max_idx - min_idx)

# calculate the abs normed rolling gradient
def rolling_gradient(y,N,keep_signs=True,use_vers="2"):
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
            keep_signs : Flag to not abs the gradient. Default True.
            use_vers : Use specific version to calculate gradient. Supported 1 or 2 referring to get_slope and
                        get_slope_v2 respectively.

        Returns the processed rolling gradient.
    '''
    if isinstance(y,np.ndarray):
        y = pd.Series(y)
    if not (use_vers in ["1","2"]):
        raise ValueError(f"Unsupported slope version! Received {use_vers}")
    # calculate rolling gradient
    slope = y.rolling(N).apply(get_slope if use_vers == "1" else get_slope_v2,raw=False).values.flatten()
    # replace NaNs with 0.0
    np.nan_to_num(slope,copy=False)
    # keep the sign
    if not keep_signs:
        return np.abs(slope)
    return slope

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
    '''
        Modified version of savgol (Savitzky-Golay) filter that supports a weighting window via window input

        See docs for inputs of scipy.signal.savgol_filter filter in scipy.

        The new input window at the end is for the weighting window applied to polyfit in fit_edge_weight. Can either be a numpy array
        of custom weight or a string supported by scipy to a specific window type. Default is Hann as it deals with common edge artifacts
        present when using the default savgol_filter.

        The function is basically a copy of the code for scipy's savgol_filter with the use of custom _fit_edge_weight to support the use of weights

        
    '''
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

def findStepChangePos(data,step_code=1):
    '''
        Search for the first position where the step changes to the target value

        Some signals transition to a new program too early causing spikes in torque that would trick the depth estimate.
        This program is to identify new boundaries to search the first peak within.

        Inputs:
            data : DataFrame containing at minimum the step codes and position for run data
            step_code : Step code or iterable list of codes to search for

        Returns a list of the first positions each target step code occurs at
    '''
    # find key for step
    # it's normally step (nb) but this is to make sure incase versions change it
    key = list(filter(lambda x : 'step' in x.lower(),data.columns))
    if len(key)==0:
        raise ValueError("Cannot find key for Step column in data")
    key = key[0]
    # format step codes into list
    steps = [float(step_code),] if isinstance(step_code,(float,int)) else step_code
    # search for each step code and find the first position
    return [data[data[key] == steps[0]]['Position (mm)'].values[0] for ss in steps]

def rollingGradientPos(ydata,xdata,N=10.0,return_pos=True):
    '''
        Find the non-overlapping rolling gradient of the given signal according using the position data

        The gradient is calculated over windows of size N mm assuming the x data is Position data

        e.g.(0,10),(10,20),(20,30)...

        This is to reveal general trends in the ydata over position with the intent of being finding the
        general window where the downward trend in the signal starts

        e.g.
        # load file
        data = loadSetitecXls(path,"auto_data")
        # calculating non-overlapping gradient
        pgrad,grad = rollingGradientPos(data['I Torque (A)'].values.flatten()+data['I Torque Empty (A)'].values.flatten(),data['Position (mm)'].values.flatten(),10.0,return_pos=True)
        # find the mid point in the window where gradient is smallest
        wstart = pgrad[np.argmin(grad)]+N/2
    '''
    xdata = np.abs(xdata)
    # get limits of the x-axis
    pmin = xdata.min()
    pmax = xdata.max()
    # arrange into limits on the windows with N space between them
    pgrad = np.arange(pmin,pmax,N)
    # iterate over windows in pairs and calculate the gradient across the values within that window
    # save the results in a list
    grad = [get_slope_v2(pd.Series(ydata[(xdata >= pA) & (xdata <= pB)])) for pA,pB in zip(pgrad,pgrad[1:])]
    # if flag is true, return the window positions used and gradient values for quick plotting
    if return_pos:
        return pgrad,grad
    else:
        return grad

def findSampleDensity(xdata,N=1.0,norm=True):
    '''
        Find the number of samples within each non-overlapping window

        The idea is to find the density of points within particular windows of N.

        As the advance rate of the data varies, the number of points within certain position windows isn't
        consistent

        Inputs:
            xdata : Data to analyse
            N : Window size to search within
            norm : Normalize the values relative to max number of samples

        Returns vector of density values
    '''
    # abs values
    # useful if position data
    pos = np.abs(xdata)
    # get min max values
    pmin = pos.min()
    pmax = pos.max()
    # create the position vector of window limits
    pgrad = np.arange(pmin,pmax,N)
    # set denominator controlling whether the values are normalized or not
    nf=1
    if norm:
        nf = pos.shape[0]
    # calculate the proportion of values within each window
    return [pos[(pos >= pA) & (pos <= pB)].shape[0]/nf for pA,pB in zip(pgrad,pgrad[1:])]

def dtwAlign(path,ref=0,key='torque',plot_res=False):
    '''
        Align signals using DTW and return the results

        Due to variances in the lengths of the signals, material thicknesses and manufacturing tolerances
        the key features of the recorded signals are slightly out of sync with each other. So to help with
        comparing the signals, it's best to align them.

        This function uses dtw_path to find the indicies that would best align each point in the target signal
        with the reference signal. The reference signal is specified either by index or by a specific path.

        The function returns a list of tuples containing the aligned position data and target signal data
        specified by key. The signal data is the target added to the empty data (eg. if key is torque,
        then the signal data is I Torque (A) + I Torque Empty (A).

        Inputs:
            path : Wildcard path to set of paths
            ref : Int index or specific path to act at the reference signal. Default 0
            key : String target signal to align. Either torque or thrust. Default torque.
            plot_res : Plot the aligned signals with the reference signal highlighted.

        Returns a list of tuples containing the aligned position and target signals and the figure.
        If plot_res is False, the figure will be None
    '''
    from dataparser import loadSetitecXls
    if isinstance(path,str):
        paths = glob(path)
    else:
        paths = path
    if isinstance(ref,int):
        ref_path = paths.pop(ref)
    elif isinstance(ref,str):
        ref_path = ref
        paths.remove(ref)
    # load reference data
    ref_data = loadSetitecXls(ref_path,"auto_data")
    # get torque signals
    ref_tq = ref_data[f'I {key.capitalize()} (A)'].values.flatten()
    if f'I {key.capitalize()} Empty (A)' in ref_data:
        ref_tq += ref_data[f'I {key.capitalize()} Empty (A)'].values.flatten()
    ref_pos = ref_data['Position (mm)'].values.flatten()
    f = None
    if plot_res:
        f,ax = plt.subplots()
        ax.plot(ref_pos,ref_tq,label='ref')
    stack = []
    # iterate over the others
    for fn in paths:
        # load reference data
        data = loadSetitecXls(fn,"auto_data")
        # get torque signals
        tq = data['I Torque (A)'].values.flatten() + data['I Torque Empty (A)'].values.flatten()
        pos = data['Position (mm)'].values.flatten()
        dtp,_ = dtw_path(ref_tq,tq)
        # extract ref
        ref_idx = [x[1] for x in dtp]
        if plot_res:
            ax.plot(pos[ref_idx],tq[ref_idx])
        stack.append((pos[ref_idx],tq[ref_idx]))
    print(ref)
    stack.insert(ref,(ref_pos,ref_tq))
    if plot_res:
        ax.legend()
    return stack,f

def _score(x,shapelet):
    ''' Score given Pandas window using DTW '''
    x.dropna()
    if isinstance(x,pd.DataFrame):
        x = x['signal']
    return dtw(shapelet,x)

def compareShapeletPosFile(fn,tool_dims,av='auto',key="torque",per='auto',mirror=True,plot_res=False,**kwargs):
    '''
        Compare contents of file against the tool shapelet specified by tool_dims and period.

        The target signal specified by key is loaded from the file fn.
        If key is torque, then I Torque (A) + I Torque Empty (A) are added together.
        If key is thrust, then I Thrust (A) + I Thrust Empty (A) are added together. If I Thrust Empty (A)
        is not in the file, then it isn't added.

        The tool shapelet is specified by tool_dims and av. tool_dims is a list of lengths of each tool segment in mm.
        The av is the AV or Feed Rate of the tool used to convert tool_dims to a time series.

        If av is auto, then the AV/Feed rate values are extracted from the program metadata. The last is used as that's
        likely the value used when actually drilling the material.

        The Position (mm) data from the file is grouped into batches of 'per' mm. The tool shapelet is then compared
        against each group using DTW to produce a score. The function returns the centre values of the windows
        and the scores for easy plotting and inspecting. 'per' can also be a list of periods to try which is useful
        for debugging and trialling things.

        If per is None, then it is set to 60% of the sum of tool_dims as that performed well in testing.

        Inputs:
            fn : File path to target file.
            tool_dims : List of dimensions for each stage of the tool.
            av : Advance/Feed Rate in mm/s or auto. Default auto.
            key : Simplified key to target signal. torque or thrust. Default torque.
            mirror : Mirror the tool shapelet so the signal decreases. Default True.
            per : Single or list of period sizes to group the data into. Default auto.
            plot_res : Plot the results on a matplotlib figure. Default False.
            scale : Scale applied to shapelet. See ToolShapelet docs.
            stitle : Axis title used if plot_res is True. Default Tool Shapelet.
            title : Figure title used if plot_res is True.
                    Default DTW Pos Score av={av},tl={tool.tlength():.3f},per={per_str},{'mirrored' if mirror else ''}
            shapelet : String to describe the shapelet. Used in filename. Default unknown.
            opath : Output file path. If given, the figure is saved to that folder. The filename is auto generated.

        Returns figure, list of list of window center points and list of list of score values for each window for each period size.
        If plot_res is False, the figure will be None.
     '''
    # load file
    if av == 'auto':
        data_all = loadSetitecXls(fn,"auto")
        av = getAV(data_all)[-1]
        data = data_all.pop(-1)
        del data_all
    else:
        data = loadSetitecXls(fn,"auto_data")
    # get signal to search
    # flip the signals as as if the position was absoluted 
    if key in ("torque","thrust"):
        search = data[f"I {key.capitalize()} (A)"].values.flatten()
        if f"I {key.capitalize()} Empty (A)" in data:
            search += data[f"I {key.capitalize()} Empty (A)"].values.flatten()
        else:
            warnings.warn(f"Could not find {key} empty channel in {fn}!")
    else:
        raise ValueError(f"Unsupported target {key}! Supported ('torque','thrust')")
    # create shapelet data
    tool = ToolShapelet(tool_dims)
    scale = kwargs.get('scale',None)
    if (scale == 'max') or ((scale is None) or mirror):
        scale = search.max() - search[0]
    elif scale == 'min':
        scale = search.min()
    elif (scale == 'mid') or ((not mirror) or (scale is None)):
        scale = (search.max()-search.min())/2
    # generate tool time series data
    _,spos,shapelet = tool.generate(av,scale,mirror=mirror)
    # if period is None, set to 60%
    if per is None:
        per = 0.6*tool.tlength()
    # offset it so it lines up with the bottom of the signal
    shapelet += search[0]
    # convert to a dataframe
    search = pd.DataFrame.from_dict({'pos':np.abs(data['Position (mm)'].values.flatten()),'signal':search})
    f = None
    if plot_res:
        f,ax = plt.subplots(ncols=2,constrained_layout=True)
        # plot tool
        ax[0].plot(spos,shapelet)
        ax[0].set(xlabel="Position (mm)",ylabel=key,title=kwargs.get("stitle","Tool Shapelet"))
        # plot original signal
        ax[1].plot(np.abs(search['pos'].values.flatten()),search['signal'],'b-')
        # create twin axis for score
        tax = ax[1].twinx()
    # if a single period value convert to a list
    if isinstance(per,float):
        per = [per,]
    scores = []
    posits = []
    # iterate over each target period size
    for pp in per:
        # group values into groups of size per mm
        gps = search.groupby(search['pos'].apply(lambda x : pp*round(x/pp)))
        pos = np.abs(list(gps.indices.keys()))+pp/2
        score = gps.apply(lambda x : _score(x,shapelet)).values.flatten()
        scores.append(score)
        posits.append(pos)
        if plot_res:
            tax.plot(pos,score,'r-',label=f"p={pp:.2f}mm")
    if plot_res:
        tax.legend()
        # set labels
        ax[1].set(xlabel="Position (mm)",ylabel=key,title="Score")
        tax.set_ylabel("DTW Score")
        per_str = '-'.join([f"{pp:.2f}" for pp in per])    
        f.suptitle(kwargs.get("title",f"DTW Pos Score av={av},tl={tool.tlength():.3f},per={per_str},{'mirrored' if mirror else ''}"))
        # save the result
        if 'opath' in kwargs:
            f.savefig(os.path.join(kwargs['opath'],f"{os.path.splitext(os.path.basename(fn))[0]}-av-{av}-tl-{per_str}{'-mirror' if mirror else ''}-sp-{kwargs.get('shapelet','unknown')}.png"))
            plt.close(f)
    # if only one period was given then get the results rather than returning a list of lissts
    if len(per)==1:
        return f,posits[0],scores[0]
    return f,posits,scores

def compareShapeletPosData(x,y,tool_dims,av,per=1.0,mirror=False,plot_res=False,**kwargs):
    '''
        Compare x,y data against the tool shapelet specified by tool_dims and period.

        The target signal is defined by the x and y vectors

        The tool shapelet is specified by tool_dims and av. tool_dims is a list of lengths of each tool segment in mm.
        The av is the AV or Feed Rate of the tool used to convert tool_dims to a time series.

        If av is auto, then the AV/Feed rate values are extracted from the program metadata. The last is used as that's
        likely the value used when actually drilling the material.

        The Position (mm) data from the file is grouped into batches of 'per' mm. The tool shapelet is then compared
        against each group using DTW to produce a score. The function returns the centre values of the windows
        and the scores for easy plotting and inspecting. 'per' can also be a list of periods to try which is useful
        for debugging and trialling things.

        If per is None, then it is set to 60% of the sum of tool_dims as that performed well in testing.

        Inputs:
            fn : File path to target file.
            tool_dims : List of dimensions for each stage of the tool.
            av : Advance/Feed Rate in mm/s or auto. Default auto.
            mirror : Mirror the tool shapelet so the signal decreases. Default True.
            per : Single or list of period sizes to group the data into. Default auto.
            plot_res : Plot the results on a matplotlib figure. Default False.
            scale : Scale applied to shapelet. See ToolShapelet docs.
            stitle : Axis title used if plot_res is True. Default Tool Shapelet.
            title : Figure title used if plot_res is True.
                    Default DTW Pos Score av={av},tl={tool.tlength():.3f},per={per_str},{'mirrored' if mirror else ''}
            shapelet : String to describe the shapelet. Used in filename. Default unknown.
            opath : Output file path. If given, the figure is saved to that folder. The filename is auto generated.

        Returns figure, list of list of window center points and list of list of score values for each window for each period size.
        If plot_res is False, the figure will be None.
     '''
    # create shapelet data
    tool = ToolShapelet(tool_dims)
    scale = kwargs.get('scale',None)
    if (scale == 'max') or ((scale is None) and mirror):
        scale = search.max() - search[0]
    elif scale == 'min':
        scale = search.min()
    elif (scale == 'mid') or ((not mirror) and (scale is None)):
        scale = (search.max()-search.min())/2

    _,spos,shapelet = tool.generate(av,scale,mirror=mirror)
    shapelet += y[0]

    search = pd.DataFrame.from_dict({'pos':np.abs(x),'signal':y})

    if plot_res:
        f,ax = plt.subplots(ncols=2,constrained_layout=True)
        ax[0].plot(spos,shapelet)
        ax[0].set(xlabel="Position (mm)",ylabel=key,title=kwargs.get("stitle","Tool Shapelet"))
        ax[1].plot(np.abs(search['pos'].values.flatten()),search['signal'],'b-')
        tax = ax[1].twinx()

    if isinstance(per,float):
        per = [per,]
    scores = []
    posits = []
    for pp in per:
        # group values into groups of size per mm
        gps = search.groupby(search['pos'].apply(lambda x : pp*round(x/pp)))
        pos = np.abs(list(gps.indices.keys()))+pp/2
        score = gps.apply(_score).values.flatten()
        scores.append(score)
        posits.append(pos)
        if plot_res:
            tax.plot(pos,score,'r-',label=f"p={pp:.2f}mm")
    if plot_res:
        tax.legend()
        #tax.set_ylim(0,score.max())
        ax[1].set(xlabel="Position (mm)",ylabel=key,title="Score")
        tax.set_ylabel("DTW Score")
        per_str = '-'.join([f"{pp:.2f}" for pp in per])    
        f.suptitle(kwargs.get("title",f"DTW Pos Score av={av},tl={math.fsum(tool_dims):.3f},per={per_str},{'mirrored' if mirror else ''}"))
        if 'opath' in kwargs:
            f.savefig(os.path.join(kwargs['opath'],f"{os.path.splitext(os.path.basename(fn))[0]}-av-{av}-tl-{per_str}{'-mirror' if mirror else ''}-sp-{kwargs.get('shapelet','unknown')}.png"))
            plt.close(f)
    if len(per)==1:
        return posits[0],scores[0]
    return posits,scores

def depth_est_rolling(ydata,xdata=None,method='wiener',NA=20,NB=None,xstart=10.0,hh=0.1,pselect='argmax',filt_grad=True,default=True,end_ref='end',window="hann",**kwargs):
    '''
        Depth Estimate using rolling gradient of smoothed signal

        Smoothes the signal using the method specified by method parameter and performs rolling gradient on the result.

        There is also an option to apply a weighting window to the rolling gradient to help remove edge artifacts caused by ripples at the start/end of the
        signal not removed by the smoothing. The window is specified by the filt_grad keyword. If True (default) then a Tukey window from scipy is applied with an
        alpha of 0.1 which found to perform well in testing. A numpy array the same length as the gradient can be given too.

        Peaks are then detected in two specified periods in the start and end of the signal.

        The first period can specified as either a single value as follows

        xdata <= xstart

        Or a two element list of where to start searching from and how far forward to search. This is to allow skipping problematic areas at the start such as when a new program step
        is engaged early

        (xdata >= xstart[0]) & (xdata <= sum(xstart))

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
        filt_weight = wiener(ydata if not isinstance(ydata,pd.DataFrame) else ydata.signal,NA)
    elif method == 'savgol':
        filt_weight = weighted_savgol_filter(ydata if not isinstance(ydata,pd.DataFrame) else ydata.signal,NA,1,deriv=0, window=window)
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
    # if a single value
    if isinstance(xstart,float):
        mask = xdata <= xstart
    # if a two element mask
    # mask to between first value and first value + window
    else:
        mask = (xdata >= xstart[0]) & (xdata <= sum(xstart))
    # if mask doesn't select and values
    # set first reference as first value
    if np.where(mask)[0].shape[0] ==0:
        if default:
            warnings.warn(f"Empty mask for 1st window!\nxstart={xstart},xmax={xmax}, NA={NA}, NB={NB}\nDefaulting to first value {xdata[0]}",category=EmptyMaskWarning)
            xA = xdata[0]
        # else raise exception if default is False
        else:
            raise EmptyMaskException(f"Empty mask for 1st window!\nxstart={xstart},xmax={xdata[-1]}, NA={NA}, NB={NB}")
    else:
        # mask gradient to first period
        grad_mask = grad[mask]
        # mask xdata to first period
        xmask = xdata[mask]
        # get max gradient value
        hlim = grad_mask.max()
        # find peaks ignoring those below the target threshold
        pks,_ = find_peaks(grad_mask, height=hh*hlim)                    
        # if no peaks were found
        if len(pks)==0:
            # if defaulting set to where max peak occurs
            if default:
                warnings.warn(f"No peaks found for 1st window {xdata.min()} to {xstart}, NA={NA}, NB={NB}! Defaulting to where max gradient occurs",category=MissingPeaksWarning)
                xA = xmask[grad_mask.argmax()]
            else:
                raise MissingPeaksException(f"No peaks found for 1st window {xdata.min()} to {xdata.max()} vs {xstart}, NA={NA}, NB={NB}!")
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
        if depth_win < 0:
            raise ValueError(f"Target window period cannot be negative! Received {depth_win}")
        # create mask starting from the first reference point+expected depth estimate
        # with a window period around it
        half_win = depth_win/2
        mask= (xdata >= (xA+depth_exp-half_win))&(xdata <= (xA+depth_exp+half_win))
    # if using minimum gradient as 2nd point
    # calculate depth estimate as distance between xA and where the minimum gradient occurs
    elif kwargs.get('try_gmin',True):
        return xdata[grad.argmin()] - xA
    # if finding the second ref using DTW
    elif kwargs.get('try_dtw',True):
        # check for depth window
        # target window size around mid point
        depth_win = kwargs.get('depth_win',None)
        # if the user didn't supply an accompanying window size
        if depth_win is None:
            raise ValueError("Missing depth window depth_win to pair with expected depth!")
        # if the user provided a negative window
        # a neg window is likely to create an empty window and raise an error
        if depth_win < 0:
            raise ValueError(f"Target window period cannot be negative! Received {depth_win}")
        # check for tool shapelet
        tool_dims = kwargs.get('tool_dims',[])
        if not tool_dims:
            raise ValueError("Tool Shapelet dimensions not specified!")
        # check for zero or negative tool dimensions
        if any([t<=0 for t in tool_dims]):
            raise ValueError(f"Tool Shapelet dimensions cannot be 0 or negative! Received {tool_dims}")
        # check feed rate
        av = kwargs.get('feed_rate',None)
        if av is None:
            raise ValueError("Missing feed rate for tool shapelet!")
        if len(av)==0:
            raise ValueError("Tool Feed Rate cannot be empty!")
        # check period
        per = kwargs.get('per',None)
        if per is None:
            per = 1.0*math.fsum(tool_dims)
        # need to check for 0 or negative
        if per <=0:
            raise ValueError(f"Grouping period cannot be 0 or negative! Received {per}!")
        fsc = ydata.signal[0]
        lsc = ydata.signal[n-1]
        offset = min([fsc,lsc])
        half_win = per/2
        # number of data points
        n = len(ydata)
        sbck_score = []
        sbck_pos = []
        # get torque signal
        scales = [ydata.signal[:n//2].max(),ydata.signal[n//2:].max()]
        for sc,av in zip(scs,avs):
            # clip to target step code
            search = ydata[ydata.sc == sc]
            # select scale
            scale = [scales[0] if search.index.max()<=n//2 else scales[1]]
            # group by period
            gps = search.groupby(search.pos.apply(lambda x : pp*round(x/pp)))
            # get position values
            pos = np.abs(list(gps.indices.keys()))+pp/2
            # generate tool shapelet
            shapelet = tool.generate(av,scale,mirror=True)[-1]+offset
            # calculate score
            score = gps.apply(lambda x : _score(x,shapelet)).values.flatten()
            # append to list
            sbck_score.append(score)
            sbck_pos.append(pos)
        # combine into arrays        
        sbck_score = np.hstack(sbck_score)
        sbck_pos = np.hstack(sbck_pos)
        np.nan_to_num(sbck_score,copy=False)
        # clip arrays to ensure it's after xA
        sbck_score = sbck_score[sbck_pos>=xA]
        sbck_pos = sbck_pos[sbck_pos>=xA]
        score_win = sbck_pos[np.where(sbck_score == sbck_score.min())[0].min()]
        # search window is 
        mask = (xdata >= (score_win-half_win)) & (xdata <= (score_win+half_win))
    # if the user hasn't specified a way to define the 2nd period to search
    # raise error
    else:
        raise ValueError("2nd window not specified! Need to specify either xendA & xendB, depth_exp & depth_win, try_gmin or try_dtw.")
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
                # find the replacement value as either the end of the signal or the upper end of the window
                xendB = min(xmax,depth_exp+(depth_win/2)) - xA
                warnings.warn(f"All empty mask for 2nd window depth_exp={depth_exp}, win={depth_win}, xA={xA}!\nDefaulting to {xendB}",category=EmptyMaskWarning)
                return xendB       
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

def plotCWT(xdata,ydata,wavelet,sr=100.0,scales=None):
    '''
        Plot the CWT of the given data

        Inputs:
            xdata: Time data
            ydata : Amplitude data
            wavelet : Target wavelet to use
            scales : Scales to try. Default None
    '''
    if isinstance(wavelet,str):
        wname = wavelet
    elif isinstance(wavelet,pywt.Wavelet):
        wname = wavelet.name

    time = xdata/sr
    # using default cone of influence settings causes an error about alpha not being a number
    # hard defining it fixes the problem
    coi = {
        'alpha':0.5,
        'hatch':'/',
    }
    return scg.cws(time, ydata, scales, wavelet=wname,figsize=(12,6), ylabel="Signal Strength", xlabel='Time (secs)', yscale='log', coikw=coi).figure

def plotCWTSeveral(xdata,ydata,sf,scales=np.arange(1,512)):
    '''
        Plot the CWT of the given data trying several wavelets

        Currently trialled wavelets are:
            gaus1, gaus2, mexh and morl

        Each result is placed on a separate set of axes

        Inputs:
            xdata: X-axis data used when plotting
            ydata : Y-axis data
            sf : Sampling rate (hz)
            scales : Scales to try with the wavelets
    '''
    wvl = pywt.wavelist(kind='continuous')
    for ww in ['cmor','fbsp','shan']:
        wvl.remove(ww)
    f,axes = plt.subplots(len(wvl)+1,1,sharex=True,tight_layout=True,figsize=(12,40))
    axes[0].plot(xdata,ydata)
    axes[0].set_xlabel("Position (mm)")
    axes[0].set_ylabel("Variable (A)")
    for ax,wv in zip(axes[1:],wvl):
        cwt,freq = pywt.cwt(ydata,scales,wv,1/sf)
        im = ax.pcolormesh(xdata,freq,np.abs(cwt),cmap="hsv")
        ax.set_ylabel("Scale")
        ax.set_xlabel("Position (mm)")
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im,cax=cax)
        ax.set_title(f"Scalegram using wavelet {wv.upper()}")
    f.tight_layout()
    return f

def plotCWTSeveralPower(xdata,ydata,sf,scales=np.arange(1,512)):
    '''
        Plot the Power CWT of the given data trying several wavelets

        Currently trialled wavelets are:
            gaus1, gaus2, mexh and morl

        Each result is placed on a separate set of axes

        Inputs:
            xdata: X-axis data used when plotting
            ydata : Y-axis data
            sf : Sampling rate (hz)
            scales : Scales to try with the wavelets
    '''
    wvl = pywt.wavelist(kind='continuous')
    for ww in ['cmor','fbsp','shan']:
        wvl.remove(ww)
    f,axes = plt.subplots(len(wvl),1,sharex=True,tight_layout=True,figsize=(12,40))
    for ax,wv in zip(axes,wvl):
        cwt,freq = pywt.cwt(ydata,scales,wv,1/sf)
        ax.pcolormesh(xdata,freq,np.abs(cwt),cmap="hsv")
        ax.set_ylabel("Freq (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Scalegram using wavelet {wv.upper()}")
        f.tight_layout()
    return f

def plotMultiDec(xdata,ydata,sf,scales=np.arange(1,512),levels=2):
    wvl = pywt.wavelist(kind='continuous')
    for ww in ['cmor','fbsp','shan']:
        wvl.remove(ww)
    f,axes = plt.subplots(len(wvl),2,sharex=True,tight_layout=True,figsize=(12,40))
    for ax,wv in zip(axes,wvl):
        cA2, cD2, cD1 = pywt.wavedec(ydata,wv,level=2)
        
def applyFft(data,**kwargs):
    '''
        Apply FFT to the target data and plot it

        Inputs:
            data : 1D array to apply FFT to
            kwargs: Keyword arguments
                sf : Sampling frequency. Used to build time vector
                time : Time vector

        Returns figure with two axes
    '''
    # get size of dataset
    N = max(data.shape)
    # apply fft
    
    #yf = fftshift(yf)
    if 'sf' in kwargs:
        sf = kwargs['sf']
        T = 1.0/sf
        xf = rfftfreq(N,T)
        yf = rfft(data)
    elif 'time' in kwargs:
        time = kwargs['time']
        xf = fftshift(time[1:N//2])
        yf = fft(data[1:N//2])
    else:
        raise ValueError("Missing time vector or sample period!")
    # create 2 axes
    f,(ax1,ax2) = plt.subplots(ncols=2,tight_layout=True,figsize=(12,6))
    # plot amplitude
    #ax1.semilogy(xf[1:N//2],2.0/N * np.abs(yf[1:N//2]))
    #ax1.plot(xf,np.abs(yf))
    ax1.semilogy(xf,np.abs(yf))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Amplitude")
    ax1.set_xlim(0,xf.max())
    plt.grid()
    # plot phase
    p = np.angle(yf)
    ax2.plot(xf,p)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Phase (Radians)")
    ax2.set_title("Phase")
    ax2.set_xlim(0,xf.max())
    plt.grid()
    return f

def plotWelch(data,fname,sf,window="hann", nperseg=256, noverlap=128, scaling='spectrum',corr=1.0):
    '''
        Apply Welch's method to the FFT of each hole, coupon and column

        The FFT of the entire signal is very dense and difficult to interpret. This due to non-stationary
        features caused by the changes in the tool surface.

        WARNING: This generates a lot of plots and takes some time, so leave adequate space and time before running

        The generated files follow the format

            fft-welch-{fname}-hole-{int(hi)}-coupon-{int(ci)}-col-{cc}-window-{window}.png

        The corr parameter is used to scale the results after computing

            sample_freq=fftshift(sample_freq)
            power=fftshift(power)/corr

        where:
            fname : Source filename without extension
            hi : Hole ID
            coupon : Coupon ID
            cc : Column
            window : Window name
            nperseg : Number of points per window/segment
            noverlap : Number of points of overlap
            scaling : Computeing either PSD or power spectrum
            corr : Scaling factor

        Inputs:
            data : Mancheste DataFrame loaded using loadSetitecNPZ
            fname : Filename used in labels and saved filename
            sf : Sampling frequency.
            window : Tapering window used. Default Hann.
            nperseg : Number of points per window/segment. Default 256
            noverlap : Number of points of overlap. Default 128.
            scaling : Computing either PSD or power spectrum
            corr : Scaling factor
    '''
    # get columns
    cols = data.loc[:,~data.columns.isin(['Time_Seconds','IndexTool','IndexHole','IndexCoupon'])].columns
    # iterate over tools
    for tt in np.unique(data['IndexTool'].values):
        # filter to tool data
        data_tool = data.loc[data['IndexTool'] == tt]
        # iterate over holes
        for hi in np.unique(data_tool['IndexHole'].values):
            # get data for hole
            data_hole = data_tool.loc[(data_tool['IndexHole'] == hi)]
            # iterate over coupons as pairs
            for ci in np.unique(data_hole['IndexCoupon']):
                print(tt,hi,ci)
                # get coupon data
                data_c0 = data_hole.loc[(data_hole['IndexCoupon'] == ci)]
                # iterate over columns
                for cc in cols:
                    # welch to get a clearer picture
                    sample_freq, power = welch(data_c0[cc].values, fs=sf, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling)
                    #fftshift the output 
                    #sample_freq=fftshift(sample_freq)
                    #power=fftshift(power)/corr
                    # create plots
                    print(f"building plots for {cc}")
                    f,ax1 = plt.subplots(tight_layout=True,figsize=(12,6))
                    # plot mag diff
                    ax1.semilogy(sample_freq,power)
                    ax1.set_xlabel("Frequency (Hz)")
                    ax1.set_ylabel("Power Spectrum Density (V**2/Hz)")
                    # set title
                    f.suptitle(f"Welch {fname},tool {tt}, hole {hi},coupon {ci}, column {cc}\nwindow {window}, nperseg {nperseg},noverlap {noverlap}, scaling {scaling}")
                    f.savefig(f"fft-welch-{fname}-tool-{tt}-hole-{int(hi)}-coupon-{int(ci)}-col-{cc}-window-{window}.png")
                plt.close('all')

def plotWelchOverlap(data,fname,sf,window="hann", nperseg=256, noverlap=128, scaling='spectrum',corr=1.0):
    '''
        Apply Welch's method to the FFT of each hole, coupon and column.

        ALL THE COUPON PLOTS ARE ON THE SAME AXES

        The FFT of the entire signal is very dense and difficult to interpret. This due to non-stationary
        features caused by the changes in the tool surface.

        WARNING: This generates a lot of plots and takes some time, so leave adequate space and time before running

        The generated files follow the format

            fft-welch-overlap-{fname}-hole-{int(hi)}-coupon-{int(ci)}-col-{cc}-window-{window}.png

        The corr parameter is used to scale the results after computing

            sample_freq=fftshift(sample_freq)
            power=fftshift(power)/corr

        where:
            fname : Source filename without extension
            hi : Hole ID
            coupon : Coupon ID
            cc : Column
            window : Window name
            nperseg : Number of points per window/segment
            noverlap : Number of points of overlap
            scaling : Computeing either PSD or power spectrum
            corr : Scaling factor

        Inputs:
            data : Manchester DataFrame loaded using loadSetitecNPZ
            fname : Filename used in saved filenames and plot labels.
            sf : Sampling rate.
            window : Tapering window used. Default Hann.
            nperseg : Number of points per window/segment. Default 256
            noverlap : Number of points of overlap. Default 128.
            scaling : Computing either PSD or power spectrum
            corr : Scaling factor
    '''
    # get columns
    cols = data.loc[:,~data.columns.isin(['Time_Seconds','IndexTool','IndexHole','IndexCoupon'])].columns
    for tt in np.unique(data['IndexTool'].values):
        data_tool = data.loc[data['IndexTool'] == tt]
        for hi in np.unique(data_tool['IndexHole'].values):
            # get data for hole
            data_hole = data_tool.loc[(data_tool['IndexHole'] == hi)]
            for cc in cols:
                f,ax1 = plt.subplots(tight_layout=True,figsize=(12,6))
                # iterate over coupons
                for ci in np.unique(data_hole['IndexCoupon']):
                    # get coupon data
                    data_c0 = data_hole.loc[(data_hole['IndexCoupon'] == ci)]
                    # welch to get a clearer picture
                    sample_freq, power = welch(data_c0[cc].values, fs=sf, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling)
                    #fftshift the output 
                    #sample_freq=fftshift(sample_freq)
                    #power=fftshift(power)/corr
                    # create plots
                    # plot mag diff
                    ax1.semilogy(sample_freq,power,label="C"+str(ci))
                plt.legend()
                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Power Spectrum (V**2/Hz)")
                # set title
                f.suptitle(f"Welch {fname},tool {tt}, coupon {ci}, hole {hi}, column {cc}\nwindow {window}, nperseg {nperseg},noverlap {noverlap}, scaling {scaling}")
                f.savefig(f"fft-welch-overlap-{fname}-tool-{tt}-hole-{int(hi)}-col-{cc}-window-{window}.png")
                plt.close('all')

def findCouponFFTDiff(data,fname,sf,**kwargs):
    '''
        Find the difference between FFT magnitudes and phase between coupons

        Iterates over holes, calculates FFT of each non-index column and then plots the difference.
        The figure is saved under the following format
            fft-coup-diff-{fname}-hole-{int(hi)}-coupons-{int(c0)}-{int(c1)}-col-{cc}.png
        where:
            fname : Filename of source without extension
            hi : Hole number
            c0: First coupon
            c1 : Second coupon
            cc : Column

        Difference is calculated as fft(c0) - fft(c1)

        Inputs:
            data : Manchester DataFrame loaded using loadSetitecNPZ
            fname : Filename used in saved filenames and plot labels.
            sf : Sampling rate.
            tool : Tool for data
    '''
    # time sample period
    T = 1.0/sf
    # get columns
    cols = data.loc[:,~data.columns.isin(['Time_Seconds','IndexTool','IndexHole','IndexCoupon'])].columns
    for hi in np.unique(data['IndexHole'].values):
        # get data for hole
        data_hole = data.loc[(data['IndexHole'] == hi)]
        # check if empty
        if min(data_hole.shape)==0:
            continue
        # find smallest data size
        sh = set(data_hole.loc[(data_hole['IndexCoupon'] == ci) & (data_hole['IndexHole'] == hi)]['Time_Seconds'].shape[0] for ci in np.unique(data_hole['IndexCoupon']) for hi in np.unique(data_hole['IndexHole']))
        N = min(sh)
        # create time data
        xf = rfftfreq(N,T)
        # setup target coupons
        if 'tcoups' in kwargs:
            tcps = kwargs['tcoups']
            if isinstance(tcps,(int,float)):
                tcps = [1.0,float(tcps)]
            if len(tcps)<2:
                raise ValueError(f"Need to supply a min of 2 target coupons {len(tcps)}<2")
            # ensure they're floating values
            tcps = [float(tp) for tp in tcps]
        else:
            tcps = np.unique(data_hole['IndexCoupon'])
        # iterate over coupons as pairs
        for c0,c1 in zip(tcps,tcps[1:]):
            print(hi,c0,c1)
            # get coupon data
            data_c0 = data_hole.loc[(data_hole['IndexCoupon'] == c0)][:N]
            data_c1 = data_hole.loc[(data_hole['IndexCoupon'] == c1)][:N]
            # if the filtered datasets are empty
            # move on. user likely gave a bad hole number
            if (min(data_c0.shape)==0) or (min(data_c1.shape)==0):
                continue
            # iterate over columns
            for cc in cols:
                # perform fft
                yf_c0 = rfft(data_c0[cc].values.flatten())
                yf_c1 = rfft(data_c1[cc].values.flatten())
                # find difference between the magnitudes
                mag_diff = np.abs(yf_c0) - np.abs(yf_c1)
                # find difference in phases
                phase_diff = np.angle(yf_c0) - np.angle(yf_c1)
                # create plots
                print(f"building plots for {cc}")
                f,(ax1,ax2) = plt.subplots(ncols=2,tight_layout=True,figsize=(12,6))
                # plot mag diff
                ax1.semilogy(xf,mag_diff)
                ax1.set_xlabel("Freq (hz)")
                ax1.set_ylabel("Mag Diff")
                ax1.set_title("Magnitude Difference")
                # plot phase diff
                ax2.plot(xf,phase_diff)
                ax2.set_xlabel("Freq (hz)")
                ax2.set_ylabel("Phase Diff (rads)")
                ax2.set_title("Phase Difference")
                # set title
                f.suptitle(f"{fname},coupons ({c0},{c1}), hole {hi}, tool {kwargs.get('tool','Unknown')}")
                f.savefig(f"fft-coup-diff-{fname}-hole-{int(hi)}-coupons-{int(c0)}-{int(c1)}-col-{cc}.png")
            plt.close('all')

def findCouponFFTDiffWelch(data,fname,sf,**kwargs):
    '''
        Find the difference between FFT magnitudes and phase between coupons

        Iterates over holes, calculates FFT of each non-index column and then plots the difference.
        The figure is saved under the following format
            fft-coup-diff-{fname}-hole-{int(hi)}-coupons-{int(c0)}-{int(c1)}-col-{cc}.png
        where:
            fname : Filename of source without extension
            hi : Hole number
            c0: First coupon
            c1 : Second coupon
            cc : Column

        Difference is calculated as fft(c0) - fft(c1)

        Inputs:
            data : Manchester DataFrame loaded using loadSetitecNPZ
            fname : Filename used in saved filenames and plot labels.
            sf : Sampling rate.
        
    '''
    # time sample period
    #T = 1.0/sf
    # get columns
    cols = data.loc[:,~data.columns.isin(['Time_Seconds','IndexTool','IndexHole','IndexCoupon'])].columns
    for tt in np.unique(data['IndexTool'].values):
        data_tool = data.loc[data['IndexTool']==tt]
        for hi in np.unique(data_tool['IndexHole'].values):
            # get data for hole
            data_hole = data_tool.loc[(data_tool['IndexHole'] == hi)]
            # check if empty
            if min(data_hole.shape)==0:
                continue
            # find smallest data size
            sh = set(data_hole.loc[(data_hole['IndexCoupon'] == ci) & (data_hole['IndexHole'] == hi)]['Time_Seconds'].shape[0] for ci in np.unique(data_hole['IndexCoupon']) for hi in np.unique(data_hole['IndexHole']))
            N = min(sh)
            # create time data
            #xf = rfftfreq(N,T)
            # setup target coupons
            if 'tcoups' in kwargs:
                tcps = kwargs['tcoups']
                if isinstance(tcps,(int,float)):
                    tcps = [1.0,float(tcps)]
                if len(tcps)<2:
                    raise ValueError(f"Need to supply a min of 2 target coupons {len(tcps)}<2")
                # ensure they're floating values
                tcps = [float(tp) for tp in tcps]
            else:
                tcps = np.unique(data_hole['IndexCoupon'])
            # iterate over coupons as pairs
            for c0,c1 in zip(tcps,tcps[1:]):
                print(hi,c0,c1)
                # get coupon data
                data_c0 = data_hole.loc[(data_hole['IndexCoupon'] == c0)][:N]
                data_c1 = data_hole.loc[(data_hole['IndexCoupon'] == c1)][:N]
                # if the filtered datasets are empty
                # iterate over columns
                for cc in cols:
                     # perform fft
                    yf_c0 = rfft(data_c0[cc].values.flatten())
                    yf_c1 = rfft(data_c1[cc].values.flatten())
                    # find difference between the magnitudes
                    mag_diff = np.abs(yf_c0) - np.abs(yf_c1)
                    sample_freq, power = welch(mag_diff, fs=sf, window=kwargs.get('window','hann'), nperseg=kwargs.get('nperseg',256), noverlap=kwargs.get('noverlap',128), scaling=kwargs.get('scaling','spectrum'))
                    # find difference in phases
                    phase_diff = np.angle(yf_c0) - np.angle(yf_c1)
                    _, power_phase = welch(phase_diff, fs=sf, window=kwargs.get('window','hann'), nperseg=kwargs.get('nperseg',256), noverlap=kwargs.get('noverlap',128), scaling=kwargs.get('scaling','spectrum'))
                    # create plots
                    print(f"building plots for {cc}")
                    f,(ax1,ax2) = plt.subplots(ncols=2,tight_layout=True,figsize=(12,6))
                    # plot mag diff
                    ax1.semilogy(sample_freq,power)
                    ax1.set_xlabel("Freq (hz)")
                    ax1.set_ylabel("Mag Diff")
                    ax1.set_title("Welch'd Magnitude Difference")
                    # plot phase diff
                    ax2.plot(sample_freq,power_phase)
                    ax2.set_xlabel("Freq (hz)")
                    ax2.set_ylabel("Phase Diff (rads)")
                    ax2.set_title("Welch'd Phase Difference")
                    # set title
                    f.suptitle(f"{fname}, tool {tt} coupons ({c0},{c1}), hole {hi}")
                    f.savefig(f"fft-welch-coup-diff-{fname}-tool-{tt}-hole-{int(hi)}-coupons-{int(c0)}-{int(c1)}-col-{cc}.png")
                plt.close('all')

def apply_fft_filt(data,sf=100):
    import scipy.signal as signal
    window = signal.general_gaussian(51, p=0.5, sig=1.5)
    filtered = signal.fftconvolve(window, data)
    filtered = (np.average(data) / np.average(filtered)) * filtered

    f,ax = plt.subplots()
    ax.plot(data,label='Original')
    ax.plot(filtered,label='Filtered')

    return f

def pcaAllHoles(data,fname,sf,ncomp='auto',T=True):
    '''
        Apply Principal Component Analysis the data for each Hole

        Iterates over each hole and non-index column. The data from each coupon at the target column
        is packed into a matrix of shape (minimum data shape, number of coupons). The minimum shape is the minimum
        shape of a column. This is so a regular matrix can be contstructed.

        Creates a plot of the PCA matrix and saves it

        Inputs:
            data : Manchester DataFrame loaded using loadSetitecNPZ
            fname : Filename used in saved filenames and plot labels.
            sf : Sampling rate.
            ncomp : Number of components. If auto, the number of components is set to number of coupons
            
    '''
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    # iterate over number of holes
    for hi in np.unique(data['IndexHole'].values):
        # filter to target hole
        data_hole = data.loc[data['IndexHole'] == hi]
        # find smallest sample sizes
        sh = set(data_hole.loc[(data_hole['IndexCoupon'] == ci) & (data_hole['IndexHole'] == hi)]['Time_Seconds'].shape[0] for ci in np.unique(data_hole['IndexCoupon']) for hi in np.unique(data_hole['IndexHole']))
        min_sh = min(sh)
        # get the number of coupons
        nc = np.unique(data_hole['IndexCoupon']).shape[0]
        # iterate over data columns
        for cc in data_hole.loc[:,~data_hole.columns.isin(['Time_Seconds','IndexTool','IndexHole','IndexCoupon'])].columns:
            # create an array to store the values
            # number of samples vs number of columns
            data_pack = np.zeros((min_sh,nc),dtype=np.float64)
            # iterate over the coupons and populate matrix
            for ci in range(1,nc):
                # filter to target coupon + column
                data_pack[:,ci] = data_hole.loc[data_hole['IndexCoupon'] == float(ci)][cc].values[:min_sh]
            data_pack = data_pack.T
            sc = StandardScaler().fit(data_pack)
            # create + fit + apply PCA
            data_trans = PCA(n_components=(nc if ncomp == 'auto' else ncomp),svd_solver='auto').fit_transform(sc.transform(data_pack))
            # plot + save
            f,ax2 = plt.subplots()
            ax2.imshow(data_trans)
            ax2.set_title('PCA')
            ax2.set_ylabel('PCA Index')
            ax2.set_xlabel('Column Index')
            f.suptitle(f"PCA {fname}, hole {hi}") 
            f.savefig(f"pca-{fname}-hole-{int(hi)}-column-{cc}-nc-{ncomp}.png")
            plt.close('all')

def pcaAllCols(data,fname,ncomp='auto'):
    '''
        Input:
            data : Manchester DataFrame loaded using loadSetitecNPZ
            fname : Filename used in saved filenames and plot labels.
            sf : Sampling rate.
    '''
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    # iterate over number of holes
    for hi in np.unique(data['IndexHole'].values):
        # filter to target hole
        data_hole = data.loc[data['IndexHole'] == hi]
        # find smallest sample sizes
        sh = set(data_hole.loc[(data_hole['IndexCoupon'] == ci) & (data_hole['IndexHole'] == hi)]['Time_Seconds'].shape[0] for ci in np.unique(data_hole['IndexCoupon']) for hi in np.unique(data_hole['IndexHole']))
        min_sh = min(sh)
        # iterate over coupons
        for ci in np.unique(data_hole['IndexCoupon']):
            # get data
            data_coup = data_hole.loc[(data_hole['IndexCoupon'] == float(ci))]
            data_pack = data_coup.loc[:,~data_hole.columns.isin(['Time_Seconds','IndexTool','IndexHole','IndexCoupon'])][:min_sh].values
            data_pack = data_pack.T
            print(hi,ci,data_pack.shape)
            # normalize
            data_pack = StandardScaler().fit_transform(data_pack)
            # create + fit + apply PCA
            data_trans = PCA(n_components=(data_pack.shape[1] if ncomp == 'auto' else ncomp),svd_solver='auto').fit_transform(data_pack)
            # plot + save
            f,ax2 = plt.subplots()
            ax2.imshow(data_trans)
            ax2.set_title('PCA')
            ax2.set_ylabel('Column')
            ax2.set_xlabel('Index')
            f.suptitle(f"PCA {fname}, hole {hi}") 
            f.savefig(f"pca-columns-{fname}-hole-{int(hi)}-coupon-{ci}-nc-{ncomp}.png")
            plt.close('all')

def plotDeriv(xdata,ydata):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(tight_layout=True)
    gs = GridSpec(2,2,figure=fig)
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.plot(xdata,ydata)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal")
    ax1.set_title("Original")
    
    yd = np.diff(ydata)/np.diff(xdata)
    xd = (xdata[1:] + xdata[1:])/2
    ax2.plot(xd,yd)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Derivative of Signal")
    ax2.set_title("1st Derivative")
    
    yd = np.diff(yd)/np.diff(xd)
    xd = (xd[1:] + xd[1:])/2
    ax3.plot(xd,yd)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Derivative of Signal")
    ax3.set_title("2nd Derivative")

    return fig
    
class BreakpointFit:
    '''
        Class to perform breakpoint fitting on the target data

        Small wrapper around a call to pwlf.PiecewiseLinFit and constructing a scipy.optimize.NonlinearConstraint

        It's designed to provide a quick and simple interface for fitting and processing the data.

        The user construct a class and then passes the data via the fit method.

        e.g.

        import dataparser as dp
        data = dp.loadSetitecXLS(path)[-1]
        fit_model = BreakpointFit()
        bps = fit_model.fit(data["Position (mm)"],data["I Thrust (A)"])

        Once fitted, the last breakpoints can be accessed via the bps attribute.
    '''
    def __init__(self,**kwargs):
        '''
            Constructor for BreakpointFit class for estimate the breakpoints from the target data

            The user specifies the fitting parameters for the fitting here and then passes the data via the fit class method

            Inputs:
                min_dist : Minimum distance between breakpoints. Default 0.55
                max_dist : Maximum distance between breakpoints. Default 5.5
                seed : Seed used to initialize starting positions for differential evolution. Added for reproducability. Default None.
                nsegs : Number of segments to fit the dat to. Default 13.
                log_fit : Flag or string to indicate that the fitting history should be logged. If True, the a log called pwlf-custom-training-log-ord.txt is created.
                            If a string, then it is taken as as the path to use. If an empty string or False, then no log is produced.        
        '''
        warnings.warn("This method of estimating depth is deprecated and should NOT be used! Use depth_est_rolling instead",category=DeprecationWarning)
        # minimum distance between breakpoints
        self.min_dist = kwargs.get("min_dist",0.55)
        if self.min_dist <= 0.0:
            raise ValueError(f"Minimum distance between breakpoints has to be positive and non-zero, Received {self.min_dist}")
        # maximum distance between breakpoints
        self.max_dist = kwargs.get("max_dist",5.5)
        if self.max_dist <= 0.0:
            raise ValueError(f"Maximum distance between breakpoints has to be positive and non-zero, Received {self.max_dist}")
        # get seed for randomizing the starting points of the differential evolution fitting
        self.seed = kwargs.get("seed",None)
        # number of segments
        self.nsegs = int(kwargs.get("nsegs",13))
        if self.nsegs <=0:
            raise ValueError(f"Target number of segments cannot be less than or equal to 0. Received {self.nsegs}")
        # construct non-linear constraint
        self._const = [NonlinearConstraint(self._min_diff_de,self.min_dist,self.max_dist,hess=lambda *x: np.zeros((self.nsegs-1,self.nsegs-1)))]
        # set logging
        self.log_fit = kwargs.get("log_fit",False)
        # if the user gave a path to log to
        if isinstance(self.log_fit,str):
            self.log_path = self.log_fit
            self.log_fit = True
        # else create a path to use
        elif self.log_fit:
            self.log_path = "pwlf-custom-training-log-ord.txt"
        # number of workers to user for fitting
        self.workers = int(kwargs.get("workers",6))
        if self.workers<0:
            raise ValueError(f"Number of workers cannot be zero! Received {self.workers}")
        self.bps = None

    def fit(self,x,y,**kwargs):
        '''
            Fit piecewise linear model to the given data using the set parameters

            Inputs:
                x : xdata
                y : ydata

            Returns the breakpoints
        '''
        import pwlf
        # construct model class
        self._fit = pwlf.PiecewiseLinFit(x,y)
        # fit model to data passing constraints
        if self.log_fit:
            ubs = [ct.ub for ct in self._const]
            lbs = [ct.lb for ct in self._const]
            open(self.log_path,'w').write(f"ub={ubs}\nlb={lbs}\nnsegs={self.nsegs}")
            self.bps= self._fit.fit(self.nsegs,seed=self.seed,constraints=self._const,callback=self.log_torque_est,workers=self.workers,**kwargs)
        else:
            self.bps =self._fit.fit(self.nsegs,seed=self.seed,constraints=self._const,workers=self.workers,**kwargs)
        # return break points
        return self.bps

    @staticmethod
    def _min_diff_de(x):
        ''' Check minimum distance between breakpoints'''
        # if the minimum distance between break points is less than the min
        # return arbitary large number to indicate a high cost
        return np.diff(x.flatten()).min()

    def log_torque_est(self,xk,convergence):
        '''Log callback for recording training history'''
        with open(self.log_path,'a') as log:
            log.write(','.join([str(kk) for kk in xk]))
            log.write(',')
            log.write(str(convergence))
            log.write("\n")

    def depthMatrix(self,targets=None):
        '''
            Calculate depth estimate as abs difference between target breakpoints

            If the user specifies None or all, then the pairwise difference between each breakpoint is found and returned as a
            nsegs x nsegs matrix. The intention being it can be rapidly plotted for inspection.

            For a more targeted list, the user must give an even length, iterable list of indicies to use. It is iterated over as
            pairs and is used to create a vector of distances.

            If the breakpoints haven't been found yet, then it returns None.

            Inputs:
                target : Even-length list of target breakpoint indicies to use to estimate depth or None or all. If None or all,
                        then the distance between each pair of breakpoints is found and returned

            Returns an array of depth estimates based on found breakpoints from the last fitting run.
        '''
        # if the breakpoints haven't been found yet
        if self.bps is None:
            return
        if (targets is None) or (targets=='all'):
            # iterate over possible combinations of break points
            depth_est =[np.abs(self.bps[b0]-self.bps[b1]) for b0,b1 in iproduct(np.arange(len(self.bps)),repeat=2)]
            # rearrange into matrix
            return np.asarray(depth_est).reshape((len(self.bps),len(self.bps)))
        else:
            # check if something iterable was given
            try:
                for ii in targets:
                    break
            except TypeError:
                raise TypeError(f"Targets must be all, None or an even length list of breakpoint indicies. Received {type(targets)}")
            # check if length of targets is even
            if (len(targets)%2)!=0:
                raise ValueError(f"The number of target breakpoints used to estimate depth must be even. Given {len(targets)} targets")
            # iterate over the targets in pairs
            return np.asarray([np.abs(self.bps[A]-self.bps[B]) for A,B in zip(targets[::2],targets[1::2])])

def wavedecShow(path,var='I Thrust (A)'):
    '''
        Decompose the target signal using DWT in a Matlab GUI

        Use the two sliders to change the target wavelet and padding method

        Uses pywt.wavedec

        The axes are dynamically destroyed and rebuilt to display the detail levels. The number of levels is the max for the target wavelet

        Inputs:
            path : Input file path to a XLS Setitec file
            var : Target variable to decompose
    '''
    from matplotlib.gridspec import GridSpec
    import dataparser as dp
    # make only figure object
    f = plt.figure(constrained_layout=False)
    # plot original data
    data = dp.loadSetitecXls(path)[-1]
    xdata = data['Position (mm)'].values
    # absolute the position data
    xdata = np.abs(xdata)
    ydata = data[var].values
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(bottom=0.21)
    # Make a horizontal slider to control the frequency.
    wvl = pywt.wavelist(kind='discrete')
    axwvl = plt.axes([0.25, 0.1, 0.65, 0.03])
    wvl_slider = Slider(
        ax=axwvl,
        label='Wavelet',
        valmin=0,
        valmax=len(wvl)-1,
        valinit=0,
        valfmt = "%d",
        valstep=1)

    f.suptitle(os.path.splitext(os.path.basename(path))[0])
    # signal extension mode used by pywt
    # exends the signal to make its size a power of 2 ?
    # pads in order to be multiple of the wavelet length?
    modes = pywt.Modes.modes
    axmode = plt.axes([0.25, 0.15, 0.65, 0.03])
    mode_slider = Slider(
        ax=axmode,
        label='Mode',
        valmin=0,
        valmax=len(pywt.Modes.modes)-1,
        valinit=4, # periodic
        valfmt = "%d",
        valstep=1.0)

    def plot_coeffs_line(ax,cwt):
        # check if there are existing lines
        if ax.collections:
            ax.collections.pop()
        ax.vlines(np.arange(0,cwt.shape[0]),[0],cwt)
        ax.set_ylim(cwt.min(),cwt.max())
        return ax

    # function to update the plots each time the wavelets is changes
    def update_plots(val):
        widx = int(wvl_slider.val)
        # get target wavelet
        wv = pywt.Wavelet(wvl[widx])
        print("processin for ",wv)
        # get the max decomp level for the current wavelet
        max_lvl = pywt.dwt_max_level(xdata.shape[0],wv.dec_len)
        # rebuild Gridspec
        gs = GridSpec(max_lvl,2,figure=f)
        # delete axes in figure
        for aa in f.axes:
            if aa in [axwvl,axmode]:
                continue
            aa.remove()
        # plot original data
        ax = f.add_subplot(gs[:,0])
        ax.plot(xdata,ydata)
        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Thrust (A)")
        ax.set_title("Original")
        # decompose the signal
        coeffs = pywt.wavedec(ydata,wvl[widx],level=max_lvl,mode=modes[int(mode_slider.val)])
        # iterate over coefficients and plots
        axprev = None
        for ii,cc in zip(range(max_lvl),coeffs):
            # share axes with previous plot
            if not (axprev is None):
                ax = f.add_subplot(gs[ii,1],sharex=axprev)
            else:
                ax = f.add_subplot(gs[ii,1])
            ax = plot_coeffs_line(ax,cc)
            ax.set_title(f"{'Approx.' if ii==0 else 'Detail'} coefficients for {wvl[widx].upper()}")
            axprev = ax
        gs.tight_layout(f)
        #gs.update(top=0.95)
        # update result
        f.canvas.draw_idle()
    update_plots(0)
    # assign update
    wvl_slider.on_changed(update_plots)
    mode_slider.on_changed(update_plots)
    plt.show()

def waveletStats(path,var='I Thrust (A)',stats="all",scales=None):
    '''
        Matplotlib GUI to show some statistics of the CWT response.

        Currently wavelet is set to morl.

        The target statistics are set as a list of supported phrases. Each statistic is plotted on separate
        axes.

        Inputs:
            path : Path to target file
            var : Target variable to apply the data to.
            stats : List of statistics to apply to the response. Supported phrases
                all : Applies max, min, mean and variance.
                max,maximum : Maximum wavelet response at each x-value across scales.
                min, minimum : Minimum wavelet response at each x-value across scales.
                mean, average : Average response across scales.
                var, variance : Variance of wavelet response across scales
                scales : Wavelet scales to apply.
    '''
    from matplotlib.gridspec import GridSpec
    import dataparser as dp
    from scaleogram.wfun import fastcwt
    # make only figure object
    f = plt.figure(constrained_layout=False)
    f.suptitle(os.path.splitext(os.path.basename(path))[0])
    # plot original data
    data = dp.loadSetitecXls(path)[-1]
    xdata = data['Position (mm)'].values
    print("xdata ",xdata.shape)
    # absolute the position data
    xdata = np.abs(xdata)
    T = 1/100.0
    time = T*np.arange(xdata.shape[0],dtype="float64")
    print("time ",time.min(),time.max())
    ydata = data[var].values
    ydata = ydata - ydata.mean()
    # adjust the main plot to make room for the sliders
    #plt.subplots_adjust(bottom=0.21)

    coi = {
        'alpha':0.5,
        'hatch':'/',
    }
    # if user wants all stats
    # override list
    if stats == "all":
        stats = ["max","min","mean","var"]

    # gridspec for plotting
    gs = GridSpec(len(stats),3,figure=f)
    # clear existing axes
    for aa in f.axes:
        aa.remove()
    # plot the original data
    ax = f.add_subplot(gs[:,0])
    # plot the original data
    ax.plot(xdata,ydata)
    # set the labels
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel(var)
    ax.set_title("Original")
    # add an axes for cWT
    ax = f.add_subplot(gs[:,1])
    # generate scales if not specified
    if scales is None:
        scales = np.linspace(10,min(len(time)/5,100),100)
    # plot the wavelet response
    scg.cws(time, ydata, scales, wavelet='morl',figsize=(12,6),ylabel="Scale", yaxis="Scale",xlabel='Time (secs)',title=var, yscale='log', coikw=coi, ax=ax)
    coeffs, _ = fastcwt(ydata, scales, 'morl', T)
    # iterate over each stats and plot axes
    for si,stat in enumerate(stats):
        # add subplot
        ax = f.add_subplot(gs[si,2])
        # if the user wants max coefficient response
        if stat.lower() in ["max","maximum"]:
            ax.plot(time,coeffs.max(axis=0),'b')
            # twin axis
            ax2 = ax.twinx()
            ax2.set_ylabel("Max Index")
            ax2.plot(time,coeffs.argmax(axis=0),'r')
            print(coeffs.argmax(axis=0).shape,coeffs.argmax(axis=0).min(),coeffs.argmax(axis=0).max())
            ax.set_ylabel("Max Response")
        # if the user wants min
        elif stat.lower() in ["min","minimum"]:
            ax.plot(time,coeffs.min(axis=0))
            ax.set_ylabel("Min Response")
        # if the user wants mean/average
        elif stat.lower() in ["mean","average"]:
            ax.plot(time,coeffs.mean(axis=0))
            ax.set_ylabel("Mean Response")
        # if the user wants variance
        elif stat.lower() in ["var","variance"]:
            ax.plot(time,coeffs.var(axis=0))
            ax.set_ylabel("Variance Response")
        # x axis is the same for all
        ax.set_xlabel("Time (s)")
        ax.set_title(stat.upper())
        
    return f

def waveletAnimation(path,opath=None,wavelet='morl',tool="UC",var='I Thrust (A)',scales=None,clim=None,dt=1.0/100.0,cmap='jet'):
    '''
        Construct an animation of the CWT wavelet response using each of the files in the target path

        The animation is constructed using matplotlib FuncAnimation and imagemagick writer.

        The target column in each file is processed by applying CWT using the target wavelet.

        Inputs:
            path : Wildcard path to input files. They need to have the hole and coupon number in a way that can be parsed by dataparser.get_hole_coupon
            opath : Output path to save the animation as. If None, it is set to wavelet-history.gif
            wavelet : Which pywt wavelet to use. Default morl.
            tool : The target tool for the dataset. Passed to get_hole_coupon
            var : Target column in data loaded using dataparser.loadSetitecXLS. Default I Thrust (A).
            scales : Wavelet scales to apply. If None, scales are set to np.arange(1, min(len(time)/10, 100)). Default None.
            clim : Colormap limits to apply to all animation frames. Default None.
            dt : Difference between time samples. Default 1/100
    '''
    import dataparser as dp
    from glob import glob
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from scaleogram.wfun import fastcwt
    from scaleogram.cws import COI_DEFAULTS, CBAR_DEFAULTS
    from matplotlib.colors import LogNorm

    # create figure
    f,ax = plt.subplots()
    # load files
    files = list(glob(path))
    # sort values according to hole and coupon
    files.sort(key=lambda x : dp.get_hole_coupon(x,tool=tool))
    max_coeff = []
    min_coeff = []
    # this code is a direct copy of the CWS code from scaleogram
    # with a couple of tweaks
    # Some if statements have been removed based on how we know the data is going to be formed
    # the important bit is the pcolormesh object is returned so it can be returned to FuncAnimation
    def cws(time, signal, scales=None, wavelet=None,
         periods=None,
         spectrum='amp', coi=True, coikw=None,
         yaxis='period',
         cscale='linear', cmap='gray', clim=None,
         cbar='vertical', cbarlabel=None,
         cbarkw=None,
         xlim=None, ylim=None, yscale=None,
         xlabel=None, ylabel=None, title=None,
         figsize=None, ax=None):
        # build a default scales array
        if scales is None:
            scales = np.arange(1, min(len(time)/10, 100))
        if scales[0] <= 0:
            raise ValueError("scales[0] must be > 0, found:"+str(scales[0]) )
        # wavelet transform
        coefs, scales_freq = fastcwt(signal, scales, wavelet, dt)
        max_coeff.append(coefs.max(axis=(0,1)))
        min_coeff.append(coefs.min(axis=(0,1)))
        # adjust y axis ticks
        scales_period = 1./scales_freq  # needed also for COI mask
        xmesh = np.concatenate([time, [time[-1]+dt]])
        if yaxis == 'period':
            ymesh = np.concatenate([scales_period, [scales_period[-1]+dt]])
            ylim  = ymesh[[-1,0]] if ylim is None else ylim
            ax.set_ylabel("Period" if ylabel is None else ylabel)
        elif yaxis == 'frequency':
            df    = scales_freq[-1]/scales_freq[-2]
            ymesh = np.concatenate([scales_freq, [scales_freq[-1]*df]])
            # set a useful yscale default: the scale freqs appears evenly in logscale
            yscale = 'log' if yscale is None else yscale
            ylim   = ymesh[[-1, 0]] if ylim is None else ylim
            ax.set_ylabel("Frequency" if ylabel is None else ylabel)
            #ax.invert_yaxis()
        elif yaxis == 'scale':
            ds = scales[-1]-scales[-2]
            ymesh = np.concatenate([scales, [scales[-1] + ds]])
            ylim  = ymesh[[-1,0]] if ylim is None else ylim
            ax.set_ylabel("Scale" if ylabel is None else ylabel)
        else:
            raise ValueError("yaxis must be one of 'scale', 'frequency' or 'period', found "
                              + str(yaxis)+" instead")

        # limit of visual range
        xr = [time.min(), time.max()]
        if xlim is None:
            xlim = xr
        else:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # adjust logarithmic scales on request (set automatically in Frequency mode)
        if yscale is not None:
            ax.set_yscale(yscale)

        # choose the correct spectrum display function and name
        if spectrum == 'amp':
            values = np.abs(coefs)
            sp_title = "Amplitude"
            cbarlabel= "abs(CWT)" if cbarlabel is None else cbarlabel
        elif spectrum == 'real':
            values = np.real(coefs)
            sp_title = "Real"
            cbarlabel= "real(CWT)" if cbarlabel is None else cbarlabel
        elif spectrum == 'imag':
            values = np.imag(coefs)
            sp_title = "Imaginary"
            cbarlabel= "imaginary(CWT)" if cbarlabel is None else cbarlabel
        elif spectrum == 'power':
            sp_title = "Power"
            cbarlabel= "abs(CWT)$^2$" if cbarlabel is None else cbarlabel
            values = np.power(np.abs(coefs),2)
        elif hasattr(spectrum, '__call__'):
            sp_title = "Custom"
            values = spectrum(coefs)
        else:
            raise ValueError("The spectrum parameter must be one of 'amp', 'real', 'imag',"+
                             "'power' or a lambda() expression")

        # labels and titles
        ax.set_title("Continuous Wavelet Transform "+sp_title+" Spectrum"
                     if title is None else title)
        ax.set_xlabel("Time/spatial domain" if xlabel is None else xlabel )

        # colorbar scale
        if cscale == 'log':
            isvalid = (values > 0)
            cnorm = LogNorm(values[isvalid].min(), values[isvalid].max())
        elif cscale == 'linear':
            cnorm = None
        else:
            raise ValueError("Color bar cscale should be 'linear' or 'log', got:"+
                             str(cscale))

        # plot the 2D spectrum using a pcolormesh to specify the correct Y axis
        # location at each scale
        qmesh = ax.pcolormesh(xmesh, ymesh, values, cmap=cmap, norm=cnorm)
        # colorbar limits
        if clim:
            qmesh.set_clim(*clim)

        # fill visually the Cone Of Influence
        # (locations subject to invalid coefficients near the borders of data)
        if coi:
            # convert the wavelet scales frequency into time domain periodicity
            scales_coi = scales_period
            max_coi  = scales_coi[-1]

            # produce the line and the curve delimiting the COI masked area
            mid = int(len(xmesh)/2)
            time0 = np.abs(xmesh[0:mid+1]-xmesh[0])
            ymask = np.zeros(len(xmesh), dtype=np.float16)
            ymhalf= ymask[0:mid+1]  # compute the left part of the mask
            ws    = np.argsort(scales_period) # ensure np.interp() works
            minscale, maxscale = sorted(ax.get_ylim())
            if yaxis == 'period':
                ymhalf[:] = np.interp(time0,
                      scales_period[ws], scales_coi[ws])
                yborder = np.zeros(len(xmesh)) + maxscale
                ymhalf[time0 > max_coi]   = maxscale
            elif yaxis == 'frequency':
                ymhalf[:] = np.interp(time0,
                      scales_period[ws], 1./scales_coi[ws])
                yborder = np.zeros(len(xmesh)) + minscale
                ymhalf[time0 > max_coi]   = minscale
            elif yaxis == 'scale':
                ymhalf[:] = np.interp(time0, scales_coi, scales)
                yborder = np.zeros(len(xmesh)) + maxscale
                ymhalf[time0 > max_coi]   = maxscale
            else:
                raise ValueError("yaxis="+str(yaxis))

            # complete the right part of the mask by symmetry
            ymask[-mid:] = ymhalf[0:mid][::-1]

            # plot the mask and forward user parameters
            plt.plot(xmesh, ymask)
            coikw = COI_DEFAULTS if coikw is None else coikw
            ax.fill_between(xmesh, yborder, ymask, **coikw )

        # color bar stuff
        if cbar:
            cbarkw   = CBAR_DEFAULTS[cbar] if cbarkw is None else cbarkw
            colorbar = plt.colorbar(qmesh, orientation=cbar, ax=ax, **cbarkw)
            if cbarlabel:
                colorbar.set_label(cbarlabel)

        return ax,qmesh
    # function to build animation
    def animate(i):
        # load data
        data = dp.loadSetitecXls(files[i])[-1]
        xdata = data['Position (mm)'].values
        #print("xdata ",xdata.shape)
        # absolute the position data
        xdata = np.abs(xdata)
        # get y data
        ydata = data[var].values
        # update title
        f.suptitle(os.path.splitext(os.path.basename(files[i]))[0])
        # subtract mean to improve result
        ydata = ydata - ydata.mean()
        # create time vector
        time = dt*np.arange(xdata.shape[0],dtype="float64")
        # create scales from the time vector
        #scales = np.linspace(1,min(len(time)/5,100),100)
        # clear existing wavelet plot
        for cc in plt.gca().collections:
            cc.remove()
        # plot CWT
        ax,qmesh = cws(time, ydata, scales=scales, wavelet='morl',figsize=(12,6),yaxis="scale",ylabel="Scale", xlabel='Time (secs)',title=var, yscale='log', clim=clim,coikw=COI_DEFAULTS, ax=plt.gca(),cbar=False,cmap=cmap)
        ax.set_xlim(time.min(),time.max())
        # get image
        return qmesh,
        
    # create animation object
    print("creating animation object")
    anim = FuncAnimation(f,animate,frames=len(files),interval=500,blit=True)
    if opath is None:
        opath = f"wavelet-history-{cmap}.gif"
    # render animation
    anim.save(opath,writer=PillowWriter(fps=20) if os.path.splitext(opath)[1] == '.gif' else FFMpegWriter(fps=20))
    print("finished ",opath)

def waveletMinMaxResponse(files,var='I Thrust (A)',tool="UC",wavelet='morl',include_loc=False,T = 1/100.0):
    '''
        Perform CWT on each of the files in the given wildcard path and plot the min and max response.

        The files are each read in using loadSetitecXls from dataparser.

        The target variable must be a valid column name in the XLS file

        The target wavelet must be supported in pywt.

        Uses fastcwt from scaleogram package

        Inputs:
            files : Wildcard path to where several XLS files are stored
            var : Target variable to analyse
            wavelet : Supported PYWT wavelet to use
            T : Time period between samples

        Returns matplotlib object
    '''
    from scaleogram.wfun import fastcwt
    import dataparser as dp
    from glob import glob
    # list of max reponse values
    max_resp = []
    # list of min response values
    min_resp = []
    # list of locations
    if include_loc:
        max_resp_idx = []
        min_resp_idx = []
    files = list(glob(files))
    files.sort(key=lambda x : dp.get_hole_coupon(x,tool))
    # iterate over each file
    for path in files:
        # load data
        data = dp.loadSetitecXls(path)[-1]
        xdata = data['Position (mm)'].values
        # absolute the position data
        xdata = np.abs(xdata)
        # get y data
        ydata = data[var].values
        # subtract mean to improve result
        ydata = ydata - ydata.mean()
        # create scales from the time vector
        scales = np.linspace(1,min(xdata.shape[0]/5,100),100)
        # perform CWT
        coefs, _ = fastcwt(ydata, scales, wavelet, T)
        # process and save results
        max_resp.append(coefs.max(axis=(0,1)))
        min_resp.append(coefs.min(axis=(0,1)))
        # if the user wants the locations of the min and max
        if include_loc:
            cflat = coefs.flatten()
            max_resp_idx.append(scales[np.unravel_index(cflat.argmax(),coefs.shape)[0]])
            min_resp_idx.append(scales[np.unravel_index(cflat.argmin(),coefs.shape)[0]])
    # plot scales
    if not include_loc:
        # create figure and index
        f,ax = plt.subplots(constrained_layout=True)
    else:
        f,[ax,ax_idx] = plt.subplots(nrows=2,constrained_layout=True)
        # plot locations
        ax_idx.plot(max_resp_idx,'m',label="min idx")
        ax_idx.plot(min_resp_idx,'g',label="max idx")
        # update y label
        ax_idx.set_ylabel("Scale")
        ax_idx.set_xlabel("Hole Number")
    # plot min and max locations
    ax.plot(max_resp,'b',label="max")
    ax.plot(min_resp,'r',label="min")
    # update the labels
    ax.set_xlabel("Hole Number")
    ax.set_ylabel("Wavelet Response")
    ax.set_title(f"Min and Max Wavelet Response for tool {tool}")
    # add legend
    f.legend()
    # return figure object
    return f

def waveletMinMaxResponseAnim(files,opath=None,var='I Thrust (A)',tool="UC",wavelet='morl',include_loc=False,T = 1/100.0):
    '''
        Construct an animation of the min and max wavelet response over the entire signal.

        The animation has 5 axes organised from top to bottom in the following order:
            - original data
            - max response at each sample across the scales
            - minimum response at each sample across the scales
            - at which scale the max response occurs
            - at which scale the min response occurs

        If the target animation is a GIF, then PillowWriter is used. For everything else FFMpegWriter is used.

        The target files are sorted by coupon and then hole.
        
        Inputs:
            files : Wildcard file path to the target files to use.
            opath : Output path for the animation. If None, it is set to f"{tool}-{var}-wavelet-response-min-max.gif"
            var : Target variable to use from the dataset. Default I Thrust (A)
            tool : Target
    '''
    from scaleogram.wfun import fastcwt
    import dataparser as dp
    from glob import glob
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    
    f,ax = plt.subplots(nrows=5,sharex=True,figsize=(14,13),constrained_layout=True)
    # get first file
    files = list(glob(files))
    
    #files.sort(key=lambda x : dp.get_hole_coupon(x,tool))
    files.sort(key=lambda x : dp.get_hole_coupon(x,tool))
    f.suptitle(f"Tool {tool}, Min Max Wavelet Response over Recording")

    ax[0].set_xlabel("Position (mm)")
    ax[0].set_ylabel(var)
    ax[0].set_title("Data")

    ax[1].set_xlabel("Postion (mm)")
    ax[1].set_ylabel("Max Response")

    ax[2].set_xlabel("Postion (mm)")
    ax[2].set_ylabel("Min Response")

    ax[3].set_xlabel("Position (mm)")
    ax[3].set_ylabel("Scale Index")

    ax[4].set_xlabel("Position (mm)")
    ax[4].set_ylabel("Scale Index")
    
    def animate(i):
        data = dp.loadSetitecXls(files[i])[-1]
        xdata = data['Position (mm)'].values
        # absolute the position data
        xdata = np.abs(xdata)
        # get y data
        ydata = data[var].values
        # subtract mean to improve result for wavelet
        ydata = ydata - ydata.mean()
        # create artificial time vectir
        time = T*np.arange(xdata.shape[0],dtype="float64")
        
        # create scales from the time vector
        scales = np.linspace(1,min(len(time)/5,100),100)
        # perform CWT
        coefs, _ = fastcwt(ydata, scales, wavelet, T)
        values = np.abs(coefs)
        # remove all lines
        for aa in f.axes:
            for line in aa.lines:
                line.remove()
                #f.canvas.draw_idle()
        # plot data
        lines = []
        lines.append(ax[0].plot(xdata,ydata,'b')[0])
        ax[0].set_title(f"Data {dp.get_hole_coupon(files[i],tool)}")
        lines.append(ax[1].plot(xdata,values.max(axis=0),'b',label="Max")[0])
        lines.append(ax[2].plot(xdata,values.min(axis=0),'r',label="Min")[0])
        lines.append(ax[3].plot(xdata,values.argmax(axis=0),'m',label="Arg Max")[0])
        lines.append(ax[4].plot(xdata,values.argmin(axis=0),'k',label="Arg Min")[0])
        f.legend()
        #f.canvas.draw_idle()
        # return lien artists
        return lines
    
    # create animation object
    anim = FuncAnimation(f,animate,frames=len(files),interval=100,blit=True,repeat=False)
    if opath is None:
        opath = f"{tool}-{var}-wavelet-response-min-max.gif"
    # render animation
    anim.save(opath,writer=PillowWriter(fps=20) if os.path.splitext(opath)[1] == '.gif' else FFMpegWriter(fps=20))
    print("finished ",opath)
    
def waveletStatsAnimation(path,opath=None,tool="UC",var='I Thrust (A)',stats="all",scales=None,cmap='jet',T = 1/100.0):
    '''
        Construct an animation of the CWT wavelet response and plots of statistics. using each of the files in the target path

        The animation is constructed using matplotlib FuncAnimation and imagemagick writer.

        The target column in each file is processed by applying CWT using the target wavelet.

        Inputs:
            path : Wildcard path to input files. They need to have the hole and coupon number in a way that can be parsed by dataparser.get_hole_coupon
            opath : Output path to save the animation as. If None, it is set to wavelet-history.gif
            wavelet : Which pywt wavelet to use. Default morl.
            tool : The target tool for the dataset. Passed to get_hole_coupon
            var : Target column in data loaded using dataparser.loadSetitecXLS. Default I Thrust (A).
            scales : Wavelet scales to apply. If None, scales are set to np.arange(1, min(len(time)/10, 100)). Default None.
            clim : Colormap limits to apply to all animation frames. Default None.
            dt : Difference between time samples. Default 1/100.
            stats : List of statistics to apply to the response. Supported phrases
                all : Applies max, min, mean and variance.
                max,maximum : Maximum wavelet response at each x-value across scales.
                min, minimum : Minimum wavelet response at each x-value across scales.
                mean, average : Average response across scales.
                var, variance : Variance of wavelet response across scales
    '''
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from scaleogram.wfun import fastcwt
    from matplotlib.gridspec import GridSpec
    import dataparser as dp
    from glob import glob
    # create figure
    f = plt.figure()
    # if user wants all stats
    # override list
    if stats == "all":
        stats = ["max","min","mean","var"]
    # create grid spec
    gs = GridSpec(len(stats),len(stats),figure=f)
    # CSW hatch spacing
    coi = {
        'alpha':0.5,
        'hatch':'/',
    }
    # load files
    files = list(glob(path))
    # sort values according to hole and coupon
    files.sort(key=lambda x,tool=tool : dp.get_hole_coupon(x,tool))
    ## create axes
    ax_data = f.add_subplot(gs[:,0])
    ax_data.set_xlabel("Position (mm)")
    ax_data.set_ylabel(var)
    ax_data.set_title("Original")
    # create line object to update
    ax_data.set_xlabel("Position (mm)")
    ax_data.set_ylabel(var)
    ax_data.set_title("Original")
    # add axes for CWT
    ax_cws = f.add_subplot(gs[:,1]) #cwt plot
    # add axes for stats
    ax_stats = []
##    line_stats = {}
##    ax_twinmax = None
##    twin_line = None
    # iterate over wanted statistics
    for si,stat in enumerate(stats):
        # create and append to list
        ax_stats.append(f.add_subplot(gs[si,2]))
        # if the maximum stat
        # create a twin axis for the max index
        if stat.lower() in ["max","maximum"]:
            ax_stats[-1].set_ylabel("Max Response")
        # if the user wants min stat
        elif stat.lower() in ["min","minimum"]:
            ax_stats[-1].set_ylabel("Min Response")
        # if the user wants mean
        elif stat.lower() in ["mean","average"]:
            ax_stats[-1].set_ylabel("Mean Response")
        # if the user wants variance
        elif stat.lower() in ["var","variance"]:
            ax_stats[-1].set_ylabel("Variance Response")
        # set x axis and title
        ax_stats[-1].set_xlabel("Time (s)")
        ax_stats[-1].set_title(stat.upper())

    #orig_axes = list(f.axes)
    # animation update function
    def animate(i):
        # if we're on the final frame, inform the user
        if i==len(files)-1:
            print("final frame!")
        # remove all lines from all axes
        for aa in f.axes:
            for line in aa.lines:
                line.remove()
        # create list
        lines = []
        #print("frame ",i,files[i])
        # load data
        data = dp.loadSetitecXls(files[i])[-1]
        xdata = data['Position (mm)'].values
        #print("xdata ",xdata.shape)
        # absolute the position data
        xdata = np.abs(xdata)
        # create time vector
        time = T*np.arange(xdata.shape[0],dtype="float64")
        # create scales from the time vector
        scales = np.linspace(1,min(len(time)/5,100),100)
        #print("time ",time.min(),time.max())
        # get y data
        ydata = data[var].values
        # subtract mean to improve result
        ydata = ydata - ydata.mean()
        # plot the data
        # plot the original data
        lines.append(ax_data.plot(xdata,ydata,'b')[0])
        f.suptitle(f"{os.path.splitext(os.path.basename(files[i]))[0]}")
        ax_data.set_title(f"{dp.get_hole_coupon(files[i],tool)}")
        # clear CWT axes
        # plot CWT
        scg.cws(time, ydata, scales, wavelet='morl',figsize=(12,6),ylabel="Scale", xlabel='Time (secs)',title=var, yscale='log', coikw=coi, ax=ax_cws,cbar=False,cmap=cmap)
        coeffs, _ = fastcwt(ydata, scales, 'morl', T)
        #print("coeffs ",coeffs.shape,coeffs.min(),coeffs.max())
        # iterate over each stats and plot axes
        for stat,ax in zip(stats,ax_stats):
            # if the user wants max coefficient response
            if stat.lower() in ["max","maximum"]:
                max_coeffs = coeffs.max(axis=0)
                # plot max coefficient values
                lines.append(ax.plot(time[:max_coeffs.shape[0]],max_coeffs,'b')[0])
                max_coeffs = coeffs.argmax(axis=0)
            # if the user wants min coefficient response
            elif stat.lower() in ["min","minimum"]:
                min_coeffs = coeffs.min(axis=0)
                lines.append(ax.plot(time[:min_coeffs.shape[0]],min_coeffs,'r')[0])
            # if the user wants avg coeficient responses
            elif stat.lower() in ["mean","average"]:
                mean_coeffs = coeffs.mean(axis=0)
                lines.append(ax.plot(time[:mean_coeffs.shape[0]],mean_coeffs,'m')[0])
            # if the user wants variance of coefficient response
            elif stat.lower() in ["var","variance"]:
                var_coeffs = coeffs.var(axis=0)
                lines.append(ax.plot(time[:var_coeffs.shape[0]],var_coeffs,'k')[0])
        #print("num lines ",len(lines))
        return lines
    # create animation object
    anim = FuncAnimation(f,animate,frames=len(files),interval=20,blit=True,repeat=False)
    if opath is None:
        opath = "wavelet-statistics.gif"
    # render animation
    anim.save(opath,writer=PillowWriter(fps=20) if os.path.splitext(opath)[1] == '.gif' else FFMpegWriter(fps=20))
    print("finished making ",opath)

def rollingAvgWindowSize(path,stats="all"):
    '''
        Matplotlib GUI to show the effects of different rolling window sizes.

        Plots the Thrust data in the file specified and the rolling window average

        For the rolling data, the rolling median of the Position (mm) data is used

        The slider controls the rolling window size
    '''
    import dataparser as dp
    # create two axes
    f,ax = plt.subplots(ncols=2)
    # load data
    data = dp.loadSetitecXls(path)[-1]
    xdata = data['Position (mm)']
    # absolute position data
    xdata = np.abs(xdata)
    ydata = data['I Thrust (A)']
    # plot the original data
    ax[0].plot(xdata,ydata,'b')
    ax[0].set_xlabel("Position (mm)")
    ax[0].set_ylabel("I Thrust (A)")
    ax[0].set_title(os.path.splitext(os.path.basename(path))[0])
    ax[0].set_xlim(xdata.min(),xdata.max())
    # setup labels for windowed data
    ax[1].set_xlabel("Position (mm)")
    ax[1].set_ylabel("Average Var")
    ax[1].set_xlim(xdata.min(),xdata.max())
    # set title to file path
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}")
    # create slider for target window size
    axwin = plt.axes([0.25, 0.1, 0.65, 0.03])
    win_slider = Slider(
        ax=axwin,
        label='Window Size',
        valmin=1,
        valmax=len(ydata), # set max size to length of dataset
        valinit=1,
        valfmt = "%d",
        valstep=1)
    # updating function for slider
    def update(val):
        # remove lines
        # creating + updating the line didn't work for some reason
        for lines in ax[1].lines:
            lines.remove()
        # get slider value
        winsz = int(win_slider.val)
        # perform rolling average of the y data
        ydata_win = ydata.rolling(winsz).mean()
        # plot rolling data
        # use rolling median of x data for plotting
        ax[1].plot(xdata.rolling(winsz).median(),ydata_win,'b')
        # update title
        ax[1].set_title(f"Window Size {winsz}")
    # assign function to slider
    win_slider.on_changed(update)
    plt.show()

def findXA(ydata,xdata=None,method='wiener',NA=20,NB=None,xstart=10.0,hh=0.1,pselect='argmax',filt_grad=True,default=True,end_ref='end',window="hann",**kwargs):
    '''
        Find first transition point using rolling gradient method

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

        Returns position of first reference point.
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
            return xdata[0]
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
                return xmask[grad_mask.argmax()]
            else:
                raise MissingPeaksException(f"No peaks found for 1st window {xdata.min()} to {xstart}, NA={NA}, NB={NB}! Defaulting to max value in window")
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
                if default:
                    warnings.warn(f"Unsupported peak selection mode {pselect}. Defaulting to argmax for first period")
                    # find where the highest peak occurs
                    pkA = grad_mask[pks].argmax()
                    # find correspondng x value
                    return grad_mask[pks][pkA]
                else:
                    raise ValueError(f"Unsupported peak selection mode {pselect}! Should be either argmax,max,limit,first or last")


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
    from dataparser import loadSetitecXls
    from glob import glob
    from matplotlib import cm   

    dtwAlign("8B Life Test/*.xls",plot_res=True)
