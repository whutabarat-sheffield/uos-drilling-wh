import pywt
from scipy.signal import iirfilter, sosfiltfilt
import numpy as np
from scaleogram.wfun import fastcwt
from abyss.plotting import icwt
import noisereduce as nr
from pandas import DataFrame

def denoise_cwt(signal,scales,T,wv,**kwargs):
    '''
        Denoise the given signal using global features from the CWT response

        The CWT using the target wavelet and given scales is applied to generate a coefficient matrix.
        
        The threshold is then calculated using features from each scale response
            sigma = (1.0/0.6745) * np.mean(np.abs(coefs[ci,:]-np.mean(coefs[ci,:])))

            uthresh = sigma * np.sqrt(2 * np.log(len(coefs[ci,:])))
        where madev is the mean absolute deviation of the signal

        The threshold is applied globally

        Inputs:
            signal : Input signal to denoise
            scales : Scales to use in wavelet transform
            T : Sampling period of the signal
            wv : Target wavelet to use
            level : Target scale level to base threshold off.
                'each' : Threshold each scale level independently
                'global' : Threshold from global features
                int : Index of coefficient row to use as reference

        Returns denoised signal as numpy array
    '''
    # get target scale
    level = kwargs.get('level','global')
    ## check target scale
    # if the user instead it as something other than each
    # raise error
    if level not in ['each','global']:
        raise ValueError(f"Target scale {level} is not supported!")
    # perform CWT
    coefs,_ = fastcwt(signal,scales,wv,T)
    # if scale is an integer
    if isinstance(level,int):
        # check if it's out of bounds
        if (level>coefs.shape[0]):
            raise ValueError(f"Target scale index {level} out of bounds!")
    # if the level is an integer
    if isinstance(level,int):
        # computer upper threshold using features from target scale index
        sigma = (1.0/0.6745) * np.mean(np.abs(coefs[level,:]-np.mean(coefs[level,:])))
        uthresh = sigma * np.sqrt(2 * np.log(len(coefs[level,:])))
        return icwt(pywt.threshold(coefs,value=uthresh,mode='hard'))
    # if level is each then threshold each level separately
    if level == 'each':
        # iterate over each scale level
        for ci in range(coefs.shape[0]):
            # computer upper threshold using features at scale index
            sigma = (1.0/0.6745) * np.mean(np.abs(coefs[ci,:]-np.mean(coefs[ci,:])))
            uthresh = sigma * np.sqrt(2 * np.log(len(coefs[ci,:])))
            # threshold scale level
            coefs[ci,:] = pywt.threshold(coefs[ci,:], value=uthresh, mode='hard')
        return icwt(coefs)
    # if level is global calculate threshold from global features
    if level == 'global':
        # computer upper threshold
        sigma = (1.0/0.6745) * np.mean(np.abs(coefs-np.mean(coefs)))
        uthresh = sigma * np.sqrt(2 * np.log(len(coefs)))
        # threshold coefficients
        coefs = pywt.threshold(coefs, value=uthresh, mode='hard')
        return icwt(coefs)

def denoise_specgate(data,T,Tnoise,**kwargs):
    '''
        Denoise the given signal using Spectral Gating.

        Features are calculated from the STFT of the noise period specified by the user and removed from the rest of the signal.

        Stationary flag is set to True to apply stationary denoising.

        Tnoise can be specified a number of ways:
            float : First Tnoise seconds
            int : First Tnoise values
            (float,float) : Signal between two timestamps
            int array : Array of indices masking values

        Inputs:
            signal : Signal to denoise
            T : Sample period of the signal. Used to mask the data to specify the noise period
            Tnoise : Period of noise to use as reference
            **kwargs : See docs for noisereduce.reduce_noise

        Returns denoised signal as numpy array
    '''
    # if the user specified noise period by single float value
    # set reference period as first Tnoise seconds
    if isinstance(Tnoise,float):
        # calculate the index to mask to
        ni = int(Tnoise/T)
        return nr.reduce_noise(y = data, sr=1/T, stationary=True,y_noise=data[:ni],**kwargs)
    # if uder specified noise period by single int
    # set reference as up to Tnoise index
    elif isinstance(Tnoise,int):
        return nr.reduce_noise(y = data, sr=1/T, stationary=True,y_noise=data[:Tnoise],**kwargs)
    # if the user gave a two element iterable, treat as time period
    elif len(Tnoise)==2:
        return nr.reduce_noise(y = data, sr=1/T, stationary=True,y_noise=data[int(Tnoise[0]/T):int(Tnoise[1]/T)],**kwargs)
    # else assumed to be index array of values to provide reference
    else:
        return nr.reduce_noise(y = data.values.flatten(), sr=1/T, stationary=True,y_noise=data.values.flatten()[Tnoise],**kwargs)

def denoise_rms(data,N=10000):
    '''
        Denoise data by applying RMS

        The given data is converted to a Pandas DataFrame and normed by absoluting and squaring the data. A rolling window
        is applied at the given size. For each window, the data is replaced with the square root of the mean.

        Inputs:
            data : 1D ndarray input data
            N : Size of the rolling window.

        Returns a denoised ndarray
    '''
    return ((DataFrame(np.abs(data)**2).rolling(N).mean())**0.5).values.flatten()

class WaveRecDenoise:
    '''
        Class for filtering data that thresholds wavedec levels according to a target threshold.

        The threshold is based on the mean abs deviation of the coefficient components at a target level.

        Means abs deviation is calculated as
            np.mean(np.absolute(d - np.mean(d, axis)))

        The threshold is then calculated as
            sigma = (1.0/self.factor) * WaveRecDenoise.madev(coeff[self.level])

            uthresh = sigma * np.sqrt(2 * np.log(len(x)))

        Coefficients that are less than uthresh are replaced according to set thresholding mode

        where:
            level is the target coefficient level
            factor is the denominator of the scaling factor applied
            x is the target dataset

        Data can be passed to the constructor for single use filtering or can be given to the run method if the user
        wants to adjust the paramters.

        e.g.
            import dataparser as dp
            data = dp.loadSetitecXLS(path)[-1]
            thrust_filt = WaveRecDenoise(data['I Thrust (A)'])

        e.g.2
            import dataparser as dp
            data = dp.loadSetitecXLS(path)[-1]
            wvr_filt = WaveRecDenoise()
            ## change parameters acording to some other inputs ##
            thrust_filt = wvr_filt.run(data['I Thrust (A)'])
    '''
    def __init__(self,**kwargs):
        '''
            Constructor for WaveRecDenoise

            Set the parameters for the denoising filter.

            If the user passes data, it is denoised and returned instead of the class

            If any of the parameters are invalid or out of bounds, a ValueError is raised

            Inputs:
                **kwargs: Keyword arguments for setting filter parameters
                    factor : Denominator for factor used to scale mean abs dev of coefficients. Default 0.6745
                    wavelet : Target wavelet. Default bior1.1
                    mode : Signal extension mode used to pad signal. See https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
                    level : Coefficient level to base upper threshold off. Default 6.
                    tmode : Thresholding mode passed to pywt.threshold. Default hard
        '''
        # get factor for scaling
        self.factor = kwargs.get("factor",0.6745)
        # get wavelet
        wvt = kwargs.get("wavelet","bior1.1")
        if not (wvt in pywt.wavelist(kind='discrete')):
            raise ValueError(f"{wvt} is not discrete! This filter requires a discrete wavelet")
        self.wvt = pywt.Wavelet(wvt)
        # get signal extension mode
        mode = kwargs.get("mode","constant")
        if not (mode in pywt.Modes.modes):
            raise ValueError(f"Mode {mode} not supported!")
        self.mode = mode
        # get level used to set threshold
        level = kwargs.get("level",6)
        self.level = level
        # threshold mode
        tmode = kwargs.get("tmode","hard")
        if not (tmode in ["soft","hard","garrote","greater","less"]):
            raise ValueError(f"Thresholding mode {tmode} not supported!")
        self.tmode = tmode
        
    # from https://www.kaggle.com/code/theoviel/denoising-with-direct-wavelet-transform/notebook
    @staticmethod
    def madev(d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def denoise(self,x):
        '''
            Decompose signal using target discrete wavelet and return the denoised data

            Signal is denoised by calculating the mean abs deviation of the DWT coefficients at the target level,
            scaling it and then calculating an upper threshold from this value

            Filter parameters are retrieved from set class parameters

            Inputs:
                x : Input data

            Returns denoised data
        '''
        # decompose the wavelet in periodic mode
        coeff = pywt.wavedec(x, self.wvt, mode=self.mode)
        # scaled mean abs deviation of signal
        sigma = (1.0/self.factor) * WaveRecDenoise.madev(coeff[self.level])
        # computer upper threshold
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        # hard threshold values
        # only keeps coefficients above the threshold
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode=self.tmode) for i in coeff[1:])
        # reconstruct
        return pywt.waverec(coeff, self.wvt, mode=self.mode)

    def check_params(self,x):
        '''
            Function to check that the set parameters are valid and can be applied to the given data

            If any of the parameters are valid, the function returns False, the problem parameters
            and an error message that can be displayed


            Inputs:
                x : Input data to be analysed

            Returns a flag indicating if the current set parameters are valid. If it's False, then it
            also returns the name of the problem parameter and a message for displaying
        '''
        # check wavelet
        if not (self.wvt in pywt.wavelist(kind='discrete')):
            return False, 'wvt', f"{self.wvt} is not discrete! This filter requires a discrete wavelet"
        # check padding mode
        if not (self.mode in pywt.Modes.modes):
            return False, 'mode', f"Mode {self.mode} not supported!"
        # check target denoising level
        max_lvl = pywt.dwt_max_level(x.shape[0],self.wvt.dec_len)
        if self.level > max_lvl:
            return False, 'level', f"Target denoising level {self.level} is greater than {max_lvl}!"
        # check thresholding method
        if not (self.tmode in ["soft","hard"]):
            return False, 'tmode', f"Thresholding mode {self.tmode} not supported!"
        return True, None, None

    def run(self,x):
        '''
            Apply the filter with the current parameters to the given data

            Calls check_params to see if the set parameters are valid. If there are any issues.
            a ValueError is raised

            Inputs:
                x : Input data

            Returns the denoised data
        '''
        # check parameters
        flag,_,msg = self.check_params(x)
        if not flag:
            raise ValueError(msg)
        # if params are fine denoise data
        return self.denoise(x)

class EnergyFilter:
    '''
        Break the data into windows and filter the data using an iirfilter if the amount of energy in the window exceeds a set limit

        The idea is to leave the low noise regions alone and target the areas with the highest amount of noise removing the HF noise causing the issues.

        The amount of energy in a given window is calculated as

        Power = sum(y_win, y_win) / y_win.size
        Energy = Power * y_win.size

        And the energy threshold to target window is a percentage of the maximum energy across all windows. If the target window energy is higher than
        the threshold, then it is filtered using a specified lowpass iirfilter applied using filtfilt.
    '''
    def __init__(self,**kwargs):
        '''
            Constructor for EnergyFilter

            Specify the energy threshold and parameters of the filter using the keyword arguments

            If data is given, smoothing is performed straight away.

            Inputs:
                window : Type of iirfilter to apply. Supported windows are ["butter","cheby1","cheby2","ellip","bessel"]. Anything else will raise a ValueError. Default butter.
                winsz : Window size. Default 87.
                force_equal : Flag to force the windows to be equal size. If True, if the target window size does not results in an integer number of windows, then it is decreased until it does. Default True.
                elimit : Percentage of maximum energy to set as the threshold. Value between 0.0 and 1.0. Default 0.77
                order : Model order. Default 1
                critfreq : Critical frequencies passed to filter. Can be a single value or an iterable list. All values must be between 0.0 and 1.0. Default and 0.95.

            If given data, the smoothed data is returned
        '''
        # window used to create filter
        window = kwargs.get("window","butter")
        # if the user gave an unsupported window raise error
        if not (window in ["butter","cheby1","cheby2","ellip","bessel"]):
            raise ValueError(f"Invalid Window choice {window}!")
        self.window = window
        # flag to force equal window size
        self.force_equal = kwargs.get("force_equal",True)
        # target window size
        winsz = kwargs.get("winsz",90)
        if winsz <=0:
            raise ValueError(f"Target window size has to be greater than 0. Given window size {winsz}")
        # save the window size
        self.winsz = winsz
        # target sampling rate
        fs = kwargs.get("fs",100.0)
        if fs <= 0.0:
            raise ValueError(f"Sampling frequency has to be greater than zero. Given sampling frequency {fs}")
        self.fs = fs
        # energy limit
        elimit = kwargs.get("elimit",0.77)
        # The energy limit is between 0.0-1.0
        # based on the max energy
        if (elimit < 0.0) or (elimit > 1.0):
            raise ValueError(f"Energy limit has to be within range [0.0, 1.0]. Given energy limit {elimit}")
        self.elimit = elimit
        # get model order
        order = kwargs.get("order",1)
        if order <=0:
            raise ValueError(f"Filter Model order has to be greater then 0! Given model order {order}")
        self.order = int(order)
        # get critical frequency
        freq = kwargs.get("critfreq",0.95)
        if isinstance(freq,float):
            if (freq < 0.0) or (freq > 1.0):
                raise ValueError(f"Filter model critical frequencies has to be in range [0.0, 1.0]! Given crit freq {freq}")
        else:
            # attempt to iterate through structure
            try:
                for ii in freq:
                    pass
            except TypeError as exp:
                raise exp
            # check values to see if they're within range
            if any([(ff < 0.0) or (ff > 1.0) for ff in freq]):
                raise ValueError(f"Filter model critical frequencies has to be in range [0.0, 1.0]! Given crit freq {freq}")
        self.freq = freq

    def findMidPoint(self,x):
        '''
            Utility function for finding the middle value of the x-data for each window so energy results can be plotted:

            Inputs:
                x : X-axis data

            Returns np.array of mid points for each window
        '''
        if self.force_equal:
            x_win = x.reshape((-1,self.winsz))
            x_win_mid = x_win[:,self.winsz//2].flatten()
        else:
            x_win = np.array_split(x,self.winsz)
            x_win_mid = np.asarray([win[self.winsz//2] for win in x_win])
        return x_win_mid

    def check_params(self,x):
        '''
            Function to check that the set parameters are valid and can be applied.

            Doesn't take any inputs

            If any of the parameters are valid, the function returns False, the problem parameters
            and an error message that can be displayed

            Returns a flag indicating if the current set parameters are valid. If it's False, then it
            also returns the name of the problem parameter and a message for displaying
        '''
        # check filter window
        if not (self.window in ["butter","cheby1","cheby2","ellip","bessel"]):
            return False, 'window', f"Invalid Window choice {self.window}!"
        # check window size
        if self.winsz <=0:
            return False, 'winsz', f"Target Window Size has to be greater than 0. Given window size {self.winsz}"
        # get number of data points
        data_sh = len(x)
        # get the size of the window needed
        # while the remainder for number of windows is non-zero
        # keep substracting by 1 until we get a valid size
        if self.force_equal:
            inc = -1
            while (data_sh%int(self.winsz))!=0:
                self.winsz += inc
                # if the window size gets less than 10
                if self.winsz <= 7:
                    # reset to default
                    #self.winsz = 7
                    # set to search upwards
                    inc = 1
        # check sampling frequency of source data
        if self.fs <= 0.0:
            return False, 'fs', f"Sampling frequency has to be greater than zero. Given sampling frequency {self.fs}"
        # check energy limit
        if (self.elimit < 0.0) or (self.elimit > 1.0):
            return False, 'elimit', f"Energy limit has to be within range [0.0, 1.0]. Given energy limit {self.elimit}"
        # check model order
        if self.order <=0:
            return False, 'order', f"Filter Model order has to be greater then 0! Given model order {self.order}"
        # check filter critical frequencies
        if isinstance(self.freq,float):
            if (self.freq < 0.0) or (self.freq > 1.0):
                return False, 'freq', f"Filter model critical frequencies has to be in range [0.0, 1.0]! Given crit freq {self.freq}"
        else:
            # attempt to iterate through structure
            try:
                for ii in self.freq:
                    pass
            except TypeError:
                return False, 'freq', f"Filter model critical frequencies has to be either a float or an iterable list of floats! Given crit freq {self.freq}"
            # check values to see if they're within range
            if any([(ff < 0.0) or (ff > 1.0) for ff in self.freq]):
                return False, 'freq', f"Filter model critical frequencies has to be in range [0.0, 1.0]! Given crit freq {self.freq}"
        return True, None, None
        
    def smooth(self,x):
        '''
            Smooth the target data using the parameters set in the constructor

            The function check_params is called in case the user has edited the parameters and it needs to be checked to see if it still complies
            with the constraints.

            The data is split into windows of self.winsz or if force_equal is False, whatever number of windows comes out. The energy for each window is
            calculated and compared against the target threshold. If it exceeds the threshold, the constructed irrfilter is applied using filtfilt.

            Inputs:
                x : Data to smooth

            Returns smoothed data.
        '''
        # check parameters
        flag,_,msg = self.check_params(x)
        if not flag:
            raise ValueError(msg)
        # if the windows are equal size
        # then we can use reshape
        if self.force_equal:
            y_win = x.reshape((-1,self.winsz))
            # calculate power
            self.p = np.sum(y_win*y_win,1)/y_win.size
            # calculate energy
            self.e = self.p * y_win.size
            # calculate threshold for max energy
            eth = self.elimit*np.max(self.e)
            # find where the energy is greather than the threshold
            smooth_idx = np.where(self.e>=eth)[0]
            # create filter
            sos = iirfilter(self.order,self.freq,fs=self.fs,btype="lowpass",ftype=self.window,output='sos')
            # iterate over windows to filter
            for smi in smooth_idx:
                y_win[smi] = sosfiltfilt(sos,y_win[smi,:])         
            return y_win.reshape(-1)
        # if unequal then array_split is used
        else:
            y_win = np.array_split(x,int(np.ceil(x.shape[0]/self.winsz)))
            # calculate power for each window
            self.p = [(np.sum(yw*yw)/yw.size,yw.size) for yw in y_win]
            # calculate energy
            self.e = np.array([pp*ysz for pp,ysz in self.p])
            # calculate threshold for max energy
            eth = self.elimit*np.max(self.e)
            # find where the energy is greather than the threshold
            smooth_idx = np.where(self.e>=eth)[0]
            # create filter
            sos = iirfilter(self.order,self.freq,fs=self.fs,btype="lowpass",ftype=self.window,output='sos')
            # iterate over windows to filter
            for smi in smooth_idx:
                y_win[smi] = sosfiltfilt(sos,y_win[smi])
            return np.hstack(y_win)

if __name__ == "__main__":
    pass
