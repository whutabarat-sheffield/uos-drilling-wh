import pandas as pd
import numpy as np
from glob import glob
import os
from scipy import signal
from scipy.optimize import minimize, LinearConstraint
from scipy.interpolate import UnivariateSpline
import stumpy
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import entropy
import pywt
from matplotlib.colors import Normalize, LogNorm, NoNorm
import scipy.special as special
from abyss.dataparser import loadSetitecXls

class ToolShapelet:
    def __init__(self,*args,**kwargs):
        '''
            Class for definining tool geometry

            The distances represent dimensions of each tool section in order going from one end of the tool
            to the other. The exact direction is up to the user.

            The keyword mode states what the measurements refer to.
            Supported modes are:
                - (default) width_height
                - angle_diameter

            IF mode is width_height

                All distances are in millimetres

                User can define it two ways:
                    - Single list of width and height pairs
                    - Two lists of width and height values respectively

            IF mode is angle_diameter
                Angles are assumed to be in radians by default but can be treated as degrees if is_rads is set to False.
                Diameters are in millimetres.

                User can define it two ways:
                    - Single list of angle and diameter pairs
                    - Two lists of angle and diameter values respectively

                Values are converted to width and height internally

            Example:
                tool_widths = [1.2,2.7,3.3]
                tool_heights = [1.2,2.7,3.3]
                tls = ToolShapelet(tool_widths,tool_heights,mode="width_height")

            Inputs:
                *args : Collection of measurements
                mode : Data mode describing the context of the data. Default width_height.
                is_rads : Flag indicating that the angle data is in radians. Default True.
        '''
        mode = kwargs.get("mode","width_height")
        if not (mode in ["width_height","angle_diameter"]):
            raise ValueError(f"Unsupported mode {mode}! Only supported modes are width_height and angle_diameter")
        # check the inputs
        if ToolShapelet.sanity_check(*args):
            # if width ahd height lists
            if len(args)==2:
                if mode == "angle_diameter":
                    self.__heights = [d/2 for d in args[1]]
                    self.__widths = [r/np.tan(theta) if kwargs.get("is_rads",False) else r/special.tandg(np.mod(theta,360.0)) for r,theta in zip(self.__heights,args[0])]
                elif mode == "width_height":
                    self.__widths = args[0]
                    self.__heights = args[1]
            # if two value pairs
            elif len(args)==1:
                self.__widths = []
                self.__heights = []
                for w,h in zip(args[0],args[1]):
                    if mode == "angle_diameter":
                        self.__heights.append(h/2)
                        self.__widths.append(h/np.tan(w) if kwargs.get("is_rads",False) else h/special.tandg(np.mod(w,360.0)))
                    elif mode == "width_height":
                        self.__widths.append(w)
                        self.__heights.append(h)
            # number of segments
            self.__ns = len(self.__widths) # number of sections

    def widths(self)->list:
        ''' Return current list of tool segment widths '''
        return self.__widths
    def heights(self)->list:
        ''' Return current list of tool segment heights '''
        return self.__heights

    # convert the tool segments into samples instead of mm
    def convertSegToSamples(self,feed_rate:float,rpm:float,sample_rate:(int,float))->tuple[int,int]:
        '''
            Convert the currently set widths and heights to number of samples

            Inputs:
                feed_rate: Tool feed rate in mm/rev.
                rpm : Angular velocity.
                sample_rate : Signal sampling rate

            Returns widths in terms of samples and heights in terms of samples
        '''
        # convert feed rate from mm/rev -> mm/s
        fr_mms = feed_rate * (rpm/60.0)
        # calculate spatial resolution
        spatial_res = fr_mms/sample_rate
         # convert to number of samples
        widths_samples = [int(round(a/spatial_res,0)) for a in self.__widths]
        height_samples = [int(round(a/spatial_res,0)) for a in self.__heights]
        return widths_samples,height_samples        

    # get total tool length
    def calculateToolLength(self)->float:
        '''
            Calculate the tool length by summing the tool widths

            Returns summation of currently set tool widths
        '''
        return sum(self.__widths)

    def maxToolHeight(self)->float:
        '''
            Calculate the height length by summing the tool height

            Returns summation of currently set tool heights
        '''
        return sum(self.__heights)

    def numSegments(self)->int:
        ''' Return number of tool segments '''
        return self.__ns

    @staticmethod
    def sanity_check(*args)->bool:
        '''
            Checks the given inputs to see if it matches the supported formats

            The class supports the following input formats:
                - Two lists representing the widths and heights of the tool segments in that order
                - Single list of pairs of numbers representing the width and height, in that order,
                  of each tool segment

            If the inputs don't satisfy these conditions, an Exception is raised.

            Inputs:
                *args : Input lists to check

            Returns True if it passes else an Exception is raised
        '''
        # check if the methods match the supported formats
        # if two lists
        if (len(args)==2) and all([type(a)==list for a in args]):
            if len(args[0])==0 or len(args[1])==0:
                raise Exception("Lists cannot be zero-length! Height and width lists must contain elements!")
            # check that they're both the same length
            if len(args[0]) != len(args[1]):
                raise Exception("Lengths do not match! Height and width lists must be the same length!")
        # if a single list
        elif len(args)==1:
            if len(args)==0:
                raise Exception("Lists cannot be zero-length! Height-width list must contain elements!")
            # check that each element is two elements in length
            if not all([len(a)==2 for a in args]):
                raise Exception("Incorrect length! For single lists, it must contain two element containers")
        return True

    def drawTool(self,**kwargs)->matplotlib.figure.Figure:
        '''
            Draw the +ve side of the tool

            Creates a new matplotlib figure and draws a series of line segments

            Inputs:
                use_fig : Draw tool segment on given axis instead of generating a
                          new one. Default None.

            Returns figure
        '''
        # make figure
        if not kwargs.get("use_fig",None):
            f,ax = plt.subplots()
        else:
            f = kwargs.get("use_fig",None)
            ax = f.axes[0]
        x=0
        y=0
        #ym=0
        # iterate over widths and heights
        for w,h in zip(self.__widths,self.__heights):
            ax.plot((x,x+w),(y,y+h),'-')
            #ax.plot((x,x+w),(ym,ym-h),'b-')
            x+=w
            y+=h
        ax.set(xlabel="Width (mm)",ylabel="Height (mm)",title="Tool Drawing")
        return f

def loadDrillingFile(torque_path:str,**kwargs):
    '''
        Wrapper function for loading drilling files

        Attempts to load Setitec file via loadSetitecXls. If the columns don't have Position,
        then it failed to load the header data so must be Lueberring.

        The loaded data is returned along with a flag indicating if it's Setitec (True) or
        Lueberring (False).

        If Setitec is loaded then the result of that function is returned. If Lueberring,
        then the loaded Panda dataframe is returned

        Inputs:
            torque_path : Path to single torque file
            **kwargs : See loadSetitecXls

        Returns the result of loadSetitecXls if a Setitec file or a pandas dataframe if Lueberring AND
        a flag indicating if it's a Setitec file or not
    '''
    is_setitec = True
    df = loadSetitecXls(torque_path,**kwargs)[-1]
    # if none of the columns contain the phrase Position then it's a Lueberring file
    # Lueberring has no header
    if not any(['Position' in c for c in df.columns]):
        is_setitec = False
        df = pd.read_excel(torque_path)
    return df,is_setitec

def getTorquePosData(df:pd.DataFrame,get_filtered:bool=False):
    '''
        Extract the torque and position data from the given dataframe

        Searches for the appropriate column name and returns the target values

        The get_filtered flag is to specify whether the user wants the filtered or
        unfiltered torque from Lueberring files

        Inputs:
            df : DataFrame loaded from either a Setitec or Lueberring data file
            get_filtered : Flag to retrieve the filtered torque column from Lueberring data. Default False.

        Returns torque and position data as specified.
    '''
    # search for torque position data
    for c in df.columns:
        if c == 'I Torque (A)':
            return df['I Torque (A)'].values,df['Position (mm)'].values
        elif c == 'distance':
            return df.filtered if get_filtered else df.unfiltered,df.distance

#https://stackoverflow.com/a/8260297
def window_rms(a, window_size):
    '''
        RMS smooth signal using numpy

        Inputs:
            window_size : Window to find value over

        Returns smoothed signal
    '''
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))

def window_rms_pd(a, window_size):
    '''
        RMS smooth signal using pandas

        Inputs:
            window_size : Window to find value over

        Returns smoothed signal
    '''
    return pd.Series(a).pow(2).rolling(window_size).mean().apply(np.sqrt,raw=True)

def compare_signals(win_size,torque,power):
    '''
        Compare two signals smoothing the first and find the RMSE between them

        torque signal is smooted using specified win_size

        Inputs:
            win_size : RMS window size
            torque : Unfiltered torque array
            power : Power signal

        Returns RMSE between the two
    '''
    if isinstance(win_size,list):
        win_size = win_size.pop()
    #print("filtering signal using ",win_size)
    filt_signal = window_rms_pd(torque,int(win_size))
    #print(np.sqrt(np.mean((filt_signal - power) ** 2)))
    #print('finding rmse')
    return np.sqrt(np.mean((filt_signal-power)**2))

def align_signals(signal1, signal2):
    '''
        Align two signals and return the results

        The signals are aligned using the stumpy.mass function setting
        the template to be the from 1/3 into signal1 to 2/3 into signal1.

        Inputs:
            signal1 : Numpy array
            signal2 : Numpy array

        Returns signal1 and signal2 in order aligned
    '''
    n = len(signal1)
    ini_t = n // 3
    end_t = n - (n // 3)

    # Extract template from signal1
    template = signal1[ini_t:end_t]

    # Calculate distance profile
    distance_profile = stumpy.mass(template, signal2)

    # Find index of best match
    idx = np.argmin(distance_profile)

    # Extract aligned signals
    sub_ini = idx*(idx<ini_t) + ini_t*(ini_t<idx)
    d1 = len(signal1)-end_t
    d2 = len(signal2)-(idx+end_t-ini_t)
    sup_end = d1*(d1<d2)+d2*(d2<d1)
    aligned_signal1 = signal1[ini_t-sub_ini:end_t+sup_end]
    aligned_signal2 = signal2[idx-sub_ini:idx+end_t-ini_t+sup_end]

    return aligned_signal1, aligned_signal2

def raise_sr(signal,old_sr,new_sr=100,smoothing=0.5):
    '''
        Raise the sampling rate of the signal by interpolation

        Creates artificial time vector to act as input and then creates
        a new time vector to interpolate up to

        Inputs:
            signal : np.ndarray to interpolate
            old_sr : Original sampling rate of signal in Hz.
            new_sr : New sampling rate to interpolate to in Hz. Default 100
            smoothing_factor: Smoothing factor of spline.

        Returns interpolated signal
    '''
    # create time vector
    old_time = np.arange(0,signal.shape[0])*(1.0/old_sr)
    new_time = np.linspace(old_time.min(),old_time.max(),int(signal.shape[0]*(new_sr/old_sr)))
    # define splint
    spl = UnivariateSpline(old_time,signal)
    spl.set_smoothing_factor(smoothing)
    # new distance values
    return spl(new_time)

def read_csv_file(filepath, skip = True):
    if skip:
    # Read the CSV file, specifying the separator as ";"
        df = pd.read_csv(filepath, sep=';', skiprows=13)
    else:
        df = pd.read_csv(filepath, sep=';')
    # Return the DataFrame
    return df

def find_ideal_win(unfiltered : (str,np.ndarray),power_signal : np.ndarray,**kwargs):
    '''
        Find the ideal window size comparing unfiltered torque against power signal

        Searches RMSE search windows to minimize the RMSE between the power and unfiltered torque signal

        The purpose of upsample is to interpolate between values to create an artificially higher sampling rate.
        The new sampling rate is applied to both the unfiltered torque and target power

        Example usage:
            torque_path = "luberring_torque_file.xlsx"
            power_path = "power_file.xlsx"
            #### load power signal and extract target period ####
            power_data = extract_power(power_path)

            res = find_ideal_win(torque_path,power_data)

        Inputs:
            unfiltered (str,np.ndarray) : Path or np array of of Lueberring unfiltred torque signal
            target (np.ndarray) : Power signal numpy array
            upsample (int) : Interpolate each signal to the target sampling rate. Default 100 Hz
            win_max : Max window size. Used to set search limits for minimize. Default half length of unfiltered torque
    '''
    # if unfiltered is specified as a path
    if isinstance(unfiltered,str):
        unfiltered = pd.read_excel(unfiltered).torque/1000*48
    # upsample the torque
    if kwargs.get("upsample",100)>0:
        unfiltered = raise_sr(unfiltered,20,new_sr = kwargs.get("upsample",100))
    # upsample the power
    if kwargs.get("upsample",100)>0:
        power_signal = raise_sr(power_signal,10,new_sr = kwargs.get("upsample",100)) 
    # setup the class for comparing
    # fn = compareTorque(power_signal.copy())
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    aligned_signal1, aligned_signal2 = align_signals(unfiltered,power_signal)
    res = minimize(compare_signals,
                   x0=[kwargs.get("x0",10),], # set starting window size
                   args=(aligned_signal1,aligned_signal2), # give signals
                   bounds=[(1,kwargs.get("win_max",len(aligned_signal1)//2))],
                   callback = print, options={"maxiter":1000,"eps":1e-4,"disp":True}, method="L-BFGS-B"#,
                   #tol = 1e-6
            )# set bounds on the window size
    return res

def plot_rmse_wsize(unfiltered : (str,np.ndarray),power_signal : np.ndarray,**kwargs):
    '''
        Find, apply and plot the ideal RMSE window size to the given torque and power signal

        The power signal is assumed to already be a subset of a larger file. The torque signal
        can either be the already loaded torque of a path to the torque file from which it is converted.

        The signals are upsampled using function raise_sr and aligned before searching for the best window size.

        Two plots are created:
            - Plot of the aligned torque and power signals
            - Plot of the Window sizes vs RMSE error with the best size marked with a red X

        Inputs:
            unfiltered : Path to or already loaded torque signal
            power_signal : Numpy array of power signal
            upsample : New sampling rate to upsample the signals to. Default 100Hz
            markersize : Size of the marker for each window size. Default 2.

        Returns created figures and best window size
    '''
    # if unfiltered is specified as a path
    if isinstance(unfiltered,str):
        unfiltered = pd.read_excel(unfiltered).torque/1000*48
    # upsample the torque
    if kwargs.get("upsample",100)>0:
        unfiltered = raise_sr(unfiltered,20,new_sr = kwargs.get("upsample",100))
    # upsample the power
    if kwargs.get("upsample",100)>0:
        power_signal = raise_sr(power_signal,10,new_sr = kwargs.get("upsample",100))
    # setup the class for comparing
    # fn = compareTorque(power_signal.copy())
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    torque, power = align_signals(unfiltered,power_signal)

    # plot the torque and power signal
    f,ax = plt.subplots()
    ax.plot(torque,'b-')
    ax.set(xlabel="Index",ylabel="Torque (mA)",title="Torque vs Power Analyser")
    tax = ax.twinx()
    tax.set_ylabel("Power (mA)")
    tax.plot(power,'r-')

    # find the RMSE for each window size
    wsizes = list(range(1,kwargs.get("win_max",len(torque)//2),1))
    rmse = [compare_signals(w,torque,power) for w in wsizes]

    # plot the RMSE for each window size
    ff,ax = plt.subplots(constrained_layout=True)
    ms = kwargs.get("markersize",2)
    ax.plot(wsizes,rmse,'-',markersize=ms,markeredgewidth=ms//2)

    # find the window where the RMSE is smallest
    ii = np.argmin(rmse)
    mms = ms*8
    ax.plot(wsizes[ii],rmse[ii],'rx',markersize=mms,markeredgewidth=mms//2)
    # create ticks for x-axis
    wticks = np.array(wsizes)
##    wticks = np.array(wsizes)[::10]
##    ax.set_xticks(wticks,wticks)
    ax.set_xlim(wticks.min(),wticks.max())
    ax.set(xlabel="Window Size (index)",ylabel="RMSE",title=f"Effect of Window Size on RMSE min at wsize={wsizes[ii]}")
    return f, ff, wsizes[ii]

def read_new_power(path_L_new : str,power_only=True):
    '''
        Read the power signals of the new tool into a pandas dataframe

        If power_only is True, the following dataframe is made.
        The pandas dataframe has the columns Signal Index and P[W].
        The column Signal Index is the order in which the power signals occur in the file
        The column P[W] is the power signal itself.

        When flag power_only is False, the min power, max power and total energy used
        in the signal is calculated and stored in a Pandas DataFrame with the columns Min, Max and Energy

        Inputs:
            path : Path to power signals for new tool
            power_only : Flag to only load the power signal or calculate the energy.

        Returns a Pandas DataFrame
    '''
    df = read_csv_file(path_L_new)
    # Indices for extract the corresponding Power Analyser signal
    ini = [270, 784, 1243, 1635, 2027, 2443, 2993, 3534, 3967, 4365]
    #[275, 789, 1248, 1640, 2032, 2448, 2998, 3539, 3972, 4370]
    end = [398, 911, 1372, 1764, 2156, 2571, 3122, 3663, 4095, 4493]
    #[393, 906, 1367, 1759, 2151, 2566, 3117, 3658, 4090, 4488]
    if power_only:
        ii = []
        power = []
        for i,(s,e) in enumerate(zip(ini,end)):
            p = df['P[W]'][s:e]
            power += p.values.tolist()
            ii += len(p)*[i,]
        return pd.DataFrame(np.concatenate([[ii,],[power,]]).T,columns=['Signal Index','P[W]'])
    else:
        j=0
        min_v = []
        max_v = []
        ener_v = []
        for i in range(len(ini)):
            # Power analyser data
            pa_power = df['P[W]'][ini[i]:end[i]]
            mini = pa_power.min()
            min_v.append(mini)
            maxi = pa_power.max()
            max_v.append(maxi)
            energy = get_energy_estimation(pa_power)
            ener_v.append(energy)

        data = list(zip(min_v,max_v,ener_v))
        df_fea = pd.DataFrame(data, columns=['Min', 'Max', 'Energy'])
        return df_fea

def read_old_power(path_L_worn,power_only=True):
    '''
        Read the power signals of the worn tool into a pandas dataframe

        If power_only is True, the following dataframe is made.
        The pandas dataframe has the columns Signal Index and P[W].
        The column Signal Index is the order in which the power signals occur in the file
        The column P[W] is the power signal itself.

        When flag power_only is False, the min power, max power and total energy used
        in the signal is calculated and stored in a Pandas DataFrame with the columns Min, Max and Energy

        Inputs:
            path : Path to power signals for new tool
            power_only : Flag to only load the power signal or calculate the energy.

        Returns a Pandas dataframe
    '''
    df = read_csv_file(path_L_worn)
    # Indices for extract the corresponding Power Analyser signal
    ini = [120,635,1210,1681,2044,2461,2846,3237,3661,4154]
    end = [247,763,1338,1808,2172,2588,2975,3364,3788,4282]

    if power_only:
        ii = []
        power = []
        for i,(s,e) in enumerate(zip(ini,end)):
            p = df['P[W]'][s:e]
            power += p.values.tolist()
            ii += len(p)*[i,]
        return pd.DataFrame(np.concatenate([[ii,],[power,]]).T,columns=['Signal Index','P[W]'])
    else:
        j=0
        min_v = []
        max_v = []
        ener_v = []
        for i in range(len(ini)):
            # Power analyser data
            pa_power = df['P[W]'][ini[i]:end[i]]
            mini = pa_power.min()
            min_v.append(mini)
            maxi = pa_power.max()
            max_v.append(maxi)
            energy = get_energy_estimation(pa_power)
            ener_v.append(energy)

        data = list(zip(min_v,max_v,ener_v))
        df_fea = pd.DataFrame(data, columns=['Min', 'Max', 'Energy'])
        return df_fea

def optimal_window_tool(torque_path,power_path,is_new,**kwargs):
    # read in power csv
    if is_new:
        power_full = read_new_power(power_path,power_only=True)
    else:
        power_full = read_old_power(power_path,power_only=True)

    new_sr = kwargs.get("upsample",100)
    # list to hold best window sizes
    best_wsize = []
    # iterate over each torque file
    for i,fn in enumerate(glob(torque_path)):
        # load torque signal
        # convert to mA
        unfiltered = pd.read_excel(fn).torque/1000*48
        # upsample to 100 Hz
        if new_sr>0:
            unfiltered = raise_sr(unfiltered,20,new_sr = new_sr)
        # get power signal
        power_fn = power_full[power_full['Signal Index']==i]['P[W]'].values
        # upsample to 100 Hz
        if new_sr>0:
            power_fn = raise_sr(power_fn,10,new_sr = new_sr)
        # align signals
        torque, power = align_signals(unfiltered,power_fn)
        # create window sizes
        wsizes = list(range(1,len(torque)//2,1))
        # find rmse for each window size
        rmse = [compare_signals(w,torque,power) for w in wsizes]
    
        # plot the RMSE for each window size
        f,ax = plt.subplots(constrained_layout=True)
        ms = kwargs.get("markersize",2)
        ax.plot(wsizes,rmse,'-',markersize=ms,markeredgewidth=ms//2)
        #ax.set(xlabel="RMS Window Size",ylabel="RMSE Error")

        # find the window where the RMSE is smallest
        ii = np.argmin(rmse)
        mms = ms*8
        ax.plot(wsizes[ii],rmse[ii],'rx',markersize=mms,markeredgewidth=mms//2)
        best_wsize.append(wsizes[ii])
        # create ticks for x-axis
        wticks = np.array(wsizes)
        ax.set_xlim(wticks.min(),wticks.max())
        ax.set(xlabel="Window Size (index)",ylabel="RMSE",title=os.path.splitext(os.path.basename(fn))[0])
        f.suptitle(f"Effect of Window Size on RMSE min at wsize={best_wsize[-1]}")
        f.savefig(f"{os.path.splitext(os.path.basename(fn))[0]}-wsizes-rmse.png")
        plt.close(f)

        # plot torque signal using best window size
        f,ax = plt.subplots()
        ax.plot(torque,'b-',label="Original")
        ax.plot(window_rms_pd(torque,int(best_wsize[-1])),'r-',label="Filtered")
        ax.set(xlabel="Index",ylabel="Torque (mA)",title=f"RMS Smoothing using best wsize={best_wsize[-1]}")
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]}")
        f.savefig(f"{os.path.splitext(os.path.basename(fn))[0]}-best-wsize.png")
        plt.close(f)

    # plot the best size found for each file
    ff,aa = plt.subplots(constrained_layout=True)
    aa.plot(best_wsize,'x',markersize=10,markeredgewidth=10//2)
    aa.set_xticks(range(len(glob(torque_path))),[os.path.splitext(os.path.basename(fn))[0] for fn in glob(torque_path)],rotation=90)
    ff.suptitle(f"Best RMS Window Size for\n{os.path.dirname(torque_path)}")
    aa.set_ylabel("RMS Window Size")
    return ff

def calcEntropy(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

def plotWinSizeEnergy(torque_path,power_path,is_new,**kwargs):
    # read in power csv
    if is_new:
        power_full = read_new_power(power_path,power_only=True)
    else:
        power_full = read_old_power(power_path,power_only=True)

    new_sr = kwargs.get("upsample",100)
    # list to hold best window sizes
    best_wsize = []
    signal_energy = []
    signal_entropy = []
    power_energy = []
    power_entropy = []
    # iterate over each torque file
    for i,fn in enumerate(glob(torque_path)):
        # load torque signal
        # convert to mA
        unfiltered = pd.read_excel(fn).torque/1000*48
        # upsample to 100 Hz
        if new_sr>0:
            unfiltered = raise_sr(unfiltered,20,new_sr = new_sr)
        # get power signal
        power_fn = power_full[power_full['Signal Index']==i]['P[W]'].values
        # upsample to 100 Hz
        if new_sr>0:
            power_fn = raise_sr(power_fn,10,new_sr = new_sr)
        # align signals
        torque, power = align_signals(unfiltered,power_fn)
        # create window sizes
        wsizes = list(range(1,len(torque)//2,1))
        # find rmse for each window size
        rmse = [compare_signals(w,torque,power) for w in wsizes]

        # find the window where the RMSE is smallest
        ii = np.argmin(rmse)
        best_wsize.append(wsizes[ii])

        signal_energy.append(np.sum(torque**2))
        signal_entropy.append(calcEntropy(torque))
        
        power_energy.append(np.sum(power**2))
        power_entropy.append(calcEntropy(power))
        
    
    # plot the best size found for each file
    ff,aa = plt.subplots(nrows=2,ncols=2,constrained_layout=True)
    aa[0,0].plot(best_wsize,signal_energy,'bx',markersize=10,markeredgewidth=10//2)
    aa[0,0].set_title(f"Best RMS Window Size vs Signal Energy")
    aa[0,0].set(xlabel="Best RMS Window Size",ylabel="Torque Signal Energy")

    aa[0,1].plot(best_wsize,signal_entropy,'bx',markersize=10,markeredgewidth=10//2)
    aa[0,1].set_title(f"Best RMS Window Size vs Signal Entropy")
    aa[0,1].set(xlabel="Best RMS Window Size",ylabel="Torque Signal Entropy")

    aa[1,0].plot(best_wsize,power_energy,'r^',markersize=10,markeredgewidth=10//2)
    aa[1,0].set_title(f"Best RMS Window Size vs Power Energy")
    aa[1,0].set(xlabel="Best RMS Window Size",ylabel="Power Signal Energy")

    aa[1,1].plot(best_wsize,power_entropy,'r^',markersize=10,markeredgewidth=10//2)
    aa[1,1].set_title(f"Best RMS Window Size vs Power Entropy")
    aa[1,1].set(xlabel="Best RMS Window Size",ylabel="Power Entropy")

    ff.suptitle(f"{os.path.dirname(torque_path)}")

    return ff


def plotWinSizeEnergyBoth(torque_new_path,power_new_path,torque_worn_path,power_worn_path,**kwargs):
    from scipy import signal

    new_sr = kwargs.get("upsample",100)
    # list to hold best window sizes
    best_wsize = []
    signal_energy = []
    signal_entropy = []
    power_energy = []
    power_entropy = []

    power_full = read_old_power(power_worn_path,power_only=True)

    # iterate over each torque file
    for i,fn in enumerate(glob(torque_worn_path)):
        # load torque signal
        # convert to mA
        unfiltered = pd.read_excel(fn).torque/1000*48
        # upsample to 100 Hz
        if new_sr>0:
            unfiltered = raise_sr(unfiltered,20,new_sr = new_sr)
        # get power signal
        power_fn = power_full[power_full['Signal Index']==i]['P[W]'].values
        # upsample to 100 Hz
        if new_sr>0:
            power_fn = raise_sr(power_fn,10,new_sr = new_sr)
        # align signals
        torque, power = align_signals(unfiltered,power_fn)
        # check if tukey window should be applied
        if kwargs.get("apply_tukey",False):
            if kwargs.get("tukey_width",0.9)>0:
                win = signal.windows.tukey(len(torque),alpha=kwargs.get("tukey_width",0.99))
                if kwargs.get("plot_win",True):
                    fw,axw = plt.subplots(ncols=2,constrained_layout=True)
                    axw[0].plot(torque,'b-',label="Original")
                    axw[0].plot(torque*win,'r-',label="Tukey Filtered")
                    axw[0].set(xlabel="Index",ylabel="Torque (W)",title="Torque")
                    axw[0].legend()
                    axw[1].plot(power,'b-',label="Original")
                    axw[1].plot(power*win,'r-',label="Tukey Filtered")
                    axw[1].set(xlabel="Index",ylabel="Power (W)",title="Power")
                    axw[1].legend()
                    fw.suptitle("Tukey Filtered Signals (Worn)")
                torque *= win
                power *= win
        # create window sizes
        wsizes = list(range(1,len(torque)//2,1))
        # find rmse for each window size
        rmse = [compare_signals(w,torque,power) for w in wsizes]

        # find the window where the RMSE is smallest
        ii = np.argmin(rmse)
        best_wsize.append(wsizes[ii])

        signal_energy.append(np.sum(torque**2))
        signal_entropy.append(calcEntropy(torque))
        
        power_energy.append(np.sum(power**2))
        power_entropy.append(calcEntropy(power))

    power_full = read_new_power(power_new_path,power_only=True)

    # iterate over each torque file
    for i,fn in enumerate(glob(torque_new_path)):
        # load torque signal
        # convert to mA
        unfiltered = pd.read_excel(fn).torque/1000*48
        # upsample to 100 Hz
        if new_sr>0:
            unfiltered = raise_sr(unfiltered,20,new_sr = new_sr)
        # get power signal
        power_fn = power_full[power_full['Signal Index']==i]['P[W]'].values
        # upsample to 100 Hz
        if new_sr>0:
            power_fn = raise_sr(power_fn,10,new_sr = new_sr)
        # align signals
        torque, power = align_signals(unfiltered,power_fn)
        if kwargs.get("apply_tukey",False):
            if kwargs.get("tukey_width",0.9)>0:
                win = signal.windows.tukey(len(torque),alpha=kwargs.get("tukey_width",0.99))
                if kwargs.get("plot_win",True):
                    fw,axw = plt.subplots(ncols=2,constrained_layout=True)
                    axw[0].plot(torque,'b-',label="Original")
                    axw[0].plot(torque*win,'r-',label="Tukey Filtered")
                    axw[0].set(xlabel="Index",ylabel="Torque (W)",title="Torque")
                    axw[0].legend()
                    axw[1].plot(power,'b-',label="Original")
                    axw[1].plot(power*win,'r-',label="Tukey Filtered")
                    axw[1].set(xlabel="Index",ylabel="Power (W)",title="Power")
                    axw[1].legend()
                    fw.suptitle("Tukey Filtered Signals (New)")
                torque *= win
                power *= win
        # create window sizes
        wsizes = list(range(1,len(torque)//2,1))
        # find rmse for each window size
        rmse = [compare_signals(w,torque,power) for w in wsizes]

        # find the window where the RMSE is smallest
        ii = np.argmin(rmse)
        best_wsize.append(wsizes[ii])

        signal_energy.append(np.sum(torque**2))
        signal_entropy.append(calcEntropy(torque))
        
        power_energy.append(np.sum(power**2))
        power_entropy.append(calcEntropy(power))
         
    # plot the best size found for each file
    ff,aa = plt.subplots(nrows=2,ncols=2,constrained_layout=True)
    aa[0,0].plot(best_wsize,signal_energy,'bx',markersize=10,markeredgewidth=10//2)
    aa[0,0].set_title(f"Best RMS Window Size vs Signal Energy")
    aa[0,0].set(xlabel="Best RMS Window Size",ylabel="Torque Signal Energy")

    aa[0,1].plot(best_wsize,signal_entropy,'bx',markersize=10,markeredgewidth=10//2)
    aa[0,1].set_title(f"Best RMS Window Size vs Signal Entropy")
    aa[0,1].set(xlabel="Best RMS Window Size",ylabel="Torque Signal Entropy")

    aa[1,0].plot(best_wsize,power_energy,'r^',markersize=10,markeredgewidth=10//2)
    aa[1,0].set_title(f"Best RMS Window Size vs Power Energy")
    aa[1,0].set(xlabel="Best RMS Window Size",ylabel="Power Signal Energy")

    aa[1,1].plot(best_wsize,power_entropy,'r^',markersize=10,markeredgewidth=10//2)
    aa[1,1].set_title(f"Best RMS Window Size vs Power Entropy")
    aa[1,1].set(xlabel="Best RMS Window Size",ylabel="Power Entropy")

    ff.suptitle(f"4B Both New and Worn Tool")

    return ff

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def waveletDenoiseSignal(signal,wavelet='db4',level=4):
    coeff = pywt.wavedec(signal, wavelet, mode="symmetric")
    sigma = (1/0.6745) * madev(coeff[-level])
    # determine upper threshold based on sigma
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    # caps the value of coefficients to uthresh
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='symmetric')

def plot_fft(path,norm_freq=True):
    from scipy.fft import rfft, rfftfreq
    import os
    df = pd.read_excel(path)
    yf = rfft(df.torque)
    xf = rfftfreq(df.shape[0],1/20.0)
    if norm_freq:
        xf /= 20.0

    f,ax = plt.subplots(ncols=2,constrained_layout=True)
    ax[0].plot(xf,np.abs(yf),'b-')
    ax[0].set_yscale('log')
    ax[0].set(xlabel=("Normalized " if norm_freq else "")+"Frequency (Hz)",ylabel="Magnitude [dB]",title="Amplitude")

    ax[1].plot(xf,np.angle(yf),'r-')
    ax[1].set(xlabel="Frequency (Hz)",ylabel="Phase (rads)",title="Phase")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]} RFFT")
    return f

def plot_fft_diff(path):
    from scipy.fft import rfft, rfftfreq
    import os
    df = pd.read_excel(path)
    # perform rfft on the original signal
    yf = rfft(df.torque)
    xf = rfftfreq(df.shape[0],1/20.0)
    # perform rfft on the overly smoothed signal
    yf_fil = rfft(df.fil_torque)
    xf_fil = rfftfreq(df.shape[0],1/20.0)

    # generate plots
    f,ax = plt.subplots(nrows=3,ncols=2,constrained_layout=True)
    ax[0,0].plot(xf,np.abs(yf),'b-')
    ax[0,0].set_yscale('log')
    ax[0,0].set(xlabel="Frequency (Hz)",ylabel="Magnitude [dB]",title="Amplitude (Unfiltered)")

    ax[0,1].plot(xf,np.angle(yf),'r-')
    ax[0,1].set(xlabel="Frequency (Hz)",ylabel="Phase (rads)",title="Phase (Unfiltered)")

    ax[1,0].plot(xf,np.abs(yf_fil),c='orange')
    ax[1,0].set_yscale('log')
    ax[1,0].set(xlabel="Frequency (Hz)",ylabel="Magnitude [dB]",title="Amplitude (Oversmoothed)")

    ax[1,1].plot(xf,np.angle(yf_fil),'g-')
    ax[1,1].set(xlabel="Frequency (Hz)",ylabel="Phase (rads)",title="Phase (Oversmoothed)")

    # find difference
    ax[2,0].plot(xf,np.abs(yf)-np.abs(yf_fil),'m-')
    ax[2,0].set_yscale('log')
    ax[2,0].set(xlabel="Frequency (Hz)",ylabel="Magnitude [dB]",title="Amplitude (Sub Difference)")

    ax[2,1].plot(xf,np.angle(yf)-np.angle(yf_fil),'k-')
    ax[2,1].set(xlabel="Frequency (Hz)",ylabel="Phase (rads)",title="Phase (Sub Difference)")
    
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]} RFFT")

    ff,axf = plt.subplots()
    axf.plot(np.abs(yf_fil)/np.abs(yf))
    axf.set_yscale('log')
    return f

def plot_fft_raise_sr(path,new_sr=100,norm_freq=False):
    from scipy.fft import rfft, rfftfreq
    # get data
    df = pd.read_excel(path)
    yf = rfft(df.torque)
    # sample rate from the time vector
    xf = rfftfreq(df.shape[0],1/20.0)
    if norm_freq:
        xf = xf/10.0
    # raise sampling rate
    torqe_sr = raise_sr(df.torque,20.0,new_sr)
    yf_sr = rfft(torqe_sr)
    xf_ssr = rfftfreq(torqe_sr.shape[0],1/new_sr)
    if norm_freq:
        xf_ssr = xf_ssr/(new_sr/2)

    f,ax = plt.subplots(ncols=2)
    ax[0].plot(xf,np.abs(yf),'b-',label="Original")
    ax[0].plot(xf_ssr,np.abs(yf_sr),'r-',label=f"SR={new_sr}")
    ax[0].set_yscale('log')
    ax[0].set(xlabel="Frequency (Hz)",ylabel="Magnitude",title="Amplitude")
    ax[0].legend()

    ax[1].plot(xf,np.angle(yf),'b-',label="Original")
    ax[1].plot(xf_ssr,np.angle(yf_sr),'r-',label=f"SR={new_sr}")
    ax[1].set(xlabel="Frequency (Hz)",ylabel="Phase (rads)",title="Phase")
    ax[1].legend()

    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]} FFT")
    return f

# adapted from https://pdf.sciencedirectassets.com/271055/1-s2.0-S0165027008X00123/1-s2.0-S0165027008002963/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD7gu0Tzrf5A53hCX%2FT4XTjcXuDREu7kYUXWXHw%2Bl%2FB2wIgFvXEgBXqTkkTPozynYFLn5lTE7HRJKfCXRJYl%2F3fCTYqswUILRAFGgwwNTkwMDM1NDY4NjUiDKU9WWn02TMhqxcL9CqQBaY%2Fs9dSTgJduvnMsn33%2FvaLCbkAhgdOc%2FiXD9Chl6vTzN%2BH2efU%2FtbN0cJSO0OdEGnztoEETJc98N3rcAxlGUZD8LSgmhpFcm9uFKCiMjKUI%2FNtBalSu4poxHA2HElVazFSb0cPJm2YGl0QQ2Vg7L%2BcS7bWHgt%2BUbvfhfXI5BYywxWMK1aQE%2B4wu9emt66P0XlYIkTqy8983YcfmU%2F2kAoZ2x9yVqNjSCgVBfnZh%2FYW1%2ByGuKaC%2BouAI2GlnkdlOATyAE5NC6nFRnun2uSOt5gurX5A1KYjwFRXhVSP%2B6K7j9DIajAmf6uc0MX24yZvC2MgnkB69aqGHrHNfONs7VpAEJiOgZSdJrHuT2VAnOQZQ%2BINhEo9bOalQMaRZkEuvAlKpGPCMHdySsl37Np82BDKpw72oHGr6ya1nQUN2I0r886u318w3bZk%2Fzo3Rbh0%2BYLDmy06bmttGPLqZWsab1tnAzJf2DPIpe%2FNKQUCNInM9NQdaOwondCdtQnnAuqmTnxRW0JiWadnNvKA8UTK9O83ZaAWcZnfPT%2FfMcdsKJ9TZL5L0neImugn08XfrEnO4lF%2BuvD716GHIHTekFXk1Zq1X0%2BA6HZpKHUMYgvdI1LTXSgyE35Vdsfyv%2FcHA6k0K7wQBJg9AZlluoKho%2Bzg8WrGyddkwxYKCn6aqkFNs5OqAX%2FORkitKjB5VO3VfWmNSUxhvzr2H1sHy1ne%2BM0lt3WLF7NNf6yfb2JqJhghajrVohF2TYOEzbU1T56g6d4xviYywwypyOvRit6NPBFEwQH4NiRgiyo8hvZtJ0SQy0s1wEm%2B66oUtYPw0ciA%2FPlQ0ALFJ2Sza0h28mih%2B5gJbUNIu2lJxnfTEE4gXbyOOTNvMNnVs6YGOrEB41MacZ6GRxv2E8JBBcD5fLTdMm0ljDAwM%2BkvAzH0IvNEdYdngSI2iAjFj7%2BQYkikCD4t3WnUtzDEXMosEfyQ8KGPH6r5mNoCoMtuWZxfxf7gIvqYlWv7nJUmwtj8u8UwYmaDpMvRTR1ZcPzqyWdyNnOyrQmtIteg3Lh%2FedkKBfzxIzTCMxRZTGRMm9KgiU9eDJP3FCxzYBedX%2FLeDC%2BEOmhdoQfUxSIX%2BSGlMNHxLmPJ&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230804T131801Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYTZA33A2K%2F20230804%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=2e480179e298cbc2f8503ba8cd59d4e0192de2014be69eba619d153541c18842&hash=12f74fe39dc0ae294755514043c6298a9ea76ad046d038881049f63b7b0f11ab&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0165027008002963&tid=spdf-9285f624-4ae7-4577-807c-c8f1f7cb3155&sid=4c56e5754cab454dd3788e15a4f57d3b6f10gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0101580652505657515a&rr=7f171579e9e775c9&cc=gb
def findWaveletCutoff(fs,level):
    return (fs*0.5)/(2**level)

def waveletDenoiseByCutoff(signal,cutoff,fs=20,wavelet='db4',level=4):
    coeff = pywt.wavedec(signal, wavelet, mode="symmetric")
    # decomposition breaks the signal into equal sized bands across the sampling frequency
    # e.g. for 100Hz @ level 2, there are 3 bands 0-33,33-66,66-100Hz
    # so at a cut off frequency of say 33, the coefficients in the 2nd and 3rd group are reduced
    fstep = fs/(level+1)
    print("denoising from cutoff",cutoff,fstep,int(np.ceil(fs/fstep)))
    # caps the value of coefficients to uthresh
    coeff[1+int(np.ceil(fs/fstep)):] = (np.zeros_like(i) for i in coeff[1+int(np.ceil(fs/fstep)):])
    return pywt.waverec(coeff, wavelet, mode='symmetric')

def plotWaveletDecomp(signal,wavelet,level,**kwargs):
##    if level is None:
##        level = pywt.dwt_max_level(len(signal),pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(signal, wavelet, mode="symmetric",level=level)
    approx = coeffs[0]
    details = coeffs[1:]

    f = plt.figure(figsize=(15,24))
    plt.subplot(len(details)+2,1,1)
    plt.plot(signal,'r-')
    plt.title("Original")
    for i in range(len(details)):
        plt.subplot(len(details)+2,1,i+2)
        d = details[len(details)-1-i]
        half = len(d)//2
        xvals = np.arange(-half,-half+len(d))* 2**i
        plt.plot(xvals, d)
        #plt.xlim(xlim)
        plt.title("detail[{}]".format(i))
    plt.subplot(len(details)+2,1,len(details)+2)
    plt.title("approx")
    plt.plot(xvals, approx)
    #plt.xlim(xlim)
    return f

def denoiseTorqueWavelet(torque_path,wavelet='db4',**kwargs):
    if isinstance(wavelet,str):
        if wavelet == 'all':
            input(f"Pywt supports {len(pywt.wavelist(kind='discrete'))} wavelets. Press ENTER to continue")
            wavelet = pywt.wavelist(kind='discrete')
        else:
            wavelet = [wavelet,]
    if isinstance(torque_path,str):
        torque_path = glob(torque_path)
    decomp_level = kwargs.get("level",1)
    cutoff = findWaveletCutoff(20,decomp_level)
    opath = kwargs.get("output_path",None)
    # collect statistics to help describe the effect it is having
    stats = pd.DataFrame(columns=['Filename','Min','Max','Mean','Variance','Wavelet'])
    # iterate over each torque file
    for i,fn in enumerate(torque_path):
        # load torque signal
        # convert to mA
        unfiltered = pd.read_excel(fn).torque
        # store referencec stats
        stats = pd.concat([stats,pd.DataFrame([[os.path.splitext(os.path.basename(fn))[0],unfiltered.min(),unfiltered.max(),unfiltered.mean(),unfiltered.var(),'None']],columns=['Filename','Min','Max','Mean','Variance','Wavelet'])])
        f,ax = plt.subplots(constrained_layout=True)
        ax.plot(unfiltered,'b-',label="Original")
        # denoise signal using each wavelet
        for w in wavelet:
            filtered = waveletDenoiseSignal(unfiltered,w,decomp_level)
            #filtered = waveletDenoiseByCutoff(unfiltered,cutoff,wavelet=w,level=decomp_level)
##            ff = plotWaveletDecomp(unfiltered,w,None)
##            ff.savefig(fr"lueberring wavelets/{os.path.splitext(os.path.basename(fn))[0]}-wavelet-decomp-{w}.png")
##            plt.close(ff)
            stats = pd.concat([stats,pd.DataFrame([[os.path.splitext(os.path.basename(fn))[0],filtered.min(),filtered.max(),filtered.mean(),filtered.var(),w]],columns=['Filename','Min','Max','Mean','Variance','Wavelet'])])
            # adding in 
            ax.plot(filtered,'-',label=w)

            if not (opath is None):
                fw,axw = plt.subplots(constrained_layout=True)
                axw.plot(unfiltered,'b-',label="Original")
                axw.plot(filtered,'r-',label="Filtered")
                axw.set(xlabel="Index",ylabel="Torque (A)",title=f"{os.path.splitext(os.path.basename(fn))[0]} denoised using {w}, auto-cutoff")
                fw.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-denoised-wavlet-{w}.png"))
                plt.close(fw)

        ax.legend()
        ax.set(xlabel="Index",ylabel="Torque (A)")
        f.suptitle(f"Wavelet Denoising of Unfiltered Torque\n{os.path.splitext(os.path.basename(fn))[0]}")

        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-denoised-wavlet-auto-all.png"))
    return stats

def denoiseTorqueWaveletSkimage(torque_path,power_path,is_new=False,wavelet='db4',**kwargs):
    from skimage.restoration import denoise_wavelet
    from skimage.metrics import peak_signal_noise_ratio
    if isinstance(wavelet,str):
        if wavelet == 'all':
            input(f"Pywt supports {len(pywt.wavelist(kind='discrete'))} wavelets. Press ENTER to continue")
            wavelet = pywt.wavelist(kind='discrete')
        else:
            wavelet = [wavelet,]
    decomp_level = kwargs.get("level",1)
    cols = ['Filename','Min','Max','Mean','Variance','PSNR','Denoising Method','Wavelet']
    opath = kwargs.get("output_path",None)
    # collect statistics to help describe the effect it is having
    stats = pd.DataFrame(columns=cols)
    wavelet_rmse_BayesShrink = []
    wavelet_rmse_VisuShrink = []

    ms = kwargs.get("markersize",2)
    mms = ms*8

    wavelet = np.array(wavelet)

    new_sr = kwargs.get("new_sr",100)

    if is_new:
        power_full = read_new_power(power_path,power_only=True)
    else:
        power_full = read_old_power(power_path,power_only=True)
    
    # iterate over each torque file
    for i,fn in enumerate(glob(torque_path)):
        wavelet_rmse_BayesShrink.clear()
        wavelet_rmse_VisuShrink.clear()

        unfiltered = pd.read_excel(fn).torque/1000*48
        # upsample to 100 Hz
        if new_sr>0:
            unfiltered = raise_sr(unfiltered,20,new_sr = new_sr)
        # get power signal
        power_fn = power_full[power_full['Signal Index']==i]['P[W]'].values
        # upsample to 100 Hz
        if new_sr>0:
            power_fn = raise_sr(power_fn,10,new_sr = new_sr)
        # align signals
        unfiltered, power = align_signals(unfiltered,power_fn)
        unorm = unfiltered/unfiltered.max()

##        # load torque signal
##        # convert to mA
##        unfiltered = pd.read_excel(fn).torque
##        unorm = unfiltered/unfiltered.max()
##        # use filtered torque for noise floor
##        ref = pd.read_excel(fn).fil_torque
##        ref /= ref.max()
        #print(unorm.min(),unorm.max())
        # store referencec stats
        stats = pd.concat([stats,pd.DataFrame([[os.path.splitext(os.path.basename(fn))[0],unfiltered.min(),unfiltered.max(),unfiltered.mean(),unfiltered.var(),peak_signal_noise_ratio(unorm,unorm,data_range=1.1),'None','None']],columns=cols)])
        f,ax = plt.subplots(nrows=2,constrained_layout=True)
        ax[0].plot(unfiltered,'b-',label="Original")
        ax[1].plot(unfiltered,'b-',label="Original")
        # denoise signal using each wavelet
        for w in wavelet:
            # skimage expects a 
            filtered = denoise_wavelet(unorm, method='BayesShrink', mode='hard', wavelet_levels=decomp_level, wavelet=w, rescale_sigma='True')
            #print(filtered.min(),filtered.max())
            stats = pd.concat([stats,pd.DataFrame([[os.path.splitext(os.path.basename(fn))[0],filtered.min(),filtered.max(),filtered.mean(),filtered.var(),peak_signal_noise_ratio(unorm,filtered,data_range=1.1),'BayesShrink',w]],columns=cols)])
            # adding in 
            ax[0].plot(filtered*unfiltered.max(),'-',label=w)
            ax[0].legend()
            ax[0].set(xlabel="Index",ylabel="Torque (W)",title="Denoised using BayesShrink")

            wavelet_rmse_BayesShrink.append(rmseDiffLengths(filtered*unfiltered.max(),power))

            if not (opath is None):
                fw,axw = plt.subplots(constrained_layout=True)
                axw.plot(unfiltered,'b-',label="Original")
                axw.plot(filtered*unfiltered.max(),'r-',label=w)
                axw.set(xlabel="Index",ylabel="Torque (W)")
                fw.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]}, denoised using BayesShrink & {w}")
                fw.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-BayesShrink-denoised-using-{w}.png"))
                plt.close(fw)

            filtered = denoise_wavelet(unorm, method='VisuShrink', mode='hard', wavelet_levels=decomp_level, wavelet=w, rescale_sigma='True')
            stats = pd.concat([stats,pd.DataFrame([[os.path.splitext(os.path.basename(fn))[0],filtered.min(),filtered.max(),filtered.mean(),filtered.var(),peak_signal_noise_ratio(unorm,filtered,data_range=1.1),'VisuShrink',w]],columns=cols)])
            # adding in 
            ax[1].plot(filtered*unfiltered.max(),'-',label=w)
            #ax[1].legend()
            ax[1].set(xlabel="Index",ylabel="Torque (W)",title="Denoised using VisuShrink")

            wavelet_rmse_VisuShrink.append(rmseDiffLengths(filtered*unfiltered.max(),power))

            if not (opath is None):
                fw,axw = plt.subplots(constrained_layout=True)
                axw.plot(unfiltered,'b-',label="Original")
                axw.plot(filtered*unfiltered.max(),'r-',label=w)
                axw.set(xlabel="Index",ylabel="Torque (W)")
                fw.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]}, denoised using VisuShrink & {w}")
                fw.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-VisuShrink-denoised-using-{w}.png"))
                plt.close(fw)
            
        f.suptitle(f"Wavelet Denoising of Unfiltered Torque\n{os.path.splitext(os.path.basename(fn))[0]}")

        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-skimage-wavelet-denoised-all.png"))

        # plot wavelet against rmse
        # find the wavelet where the RMSE is smallest
        wavelet_rmse = np.array(wavelet_rmse_BayesShrink)
        ii = np.argmin(wavelet_rmse)
        #best_wavelet.append(wavelet[ii])

        # plot the RMSE for each window size
        ff,ax = plt.subplots(constrained_layout=True,figsize=[12, 4.8])
        ax.plot(range(len(wavelet_rmse)),wavelet_rmse,'-',markersize=ms,markeredgewidth=ms//2)        
        ax.plot(ii,wavelet_rmse[ii],'rx',markersize=mms,markeredgewidth=mms//2)
        ax.set_xticks(range(len(wavelet_rmse)),wavelet)
        ax.set_xlim(0,len(wavelet))

        ax.set(xlabel="Wavelet",ylabel="RMSE",title=f"Effect of Wavelet Size on RMSE using BayesShrink\nmin found using {wavelet[ii]}")
        if not (opath is None):
            ff.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-skimage-wavelet-denoise-BayesShrink-wavelet-rmse.png"))

        wavelet_rmse = np.array(wavelet_rmse_VisuShrink)
        ii = np.argmin(wavelet_rmse)
        #best_wavelet.append(wavelet[ii])

        # plot the RMSE for each window size
        ff,ax = plt.subplots(constrained_layout=True,figsize=[12, 4.8])
        ax.plot(range(len(wavelet_rmse)),wavelet_rmse,'-',markersize=ms,markeredgewidth=ms//2)        
        ax.plot(ii,wavelet_rmse[ii],'rx',markersize=mms,markeredgewidth=mms//2)
        ax.set_xticks(range(len(wavelet_rmse)),wavelet)
        ax.set_xlim(0,len(wavelet))
        ax.set(xlabel="Wavelet",ylabel="RMSE",title=f"Effect of Wavelet Size on RMSE using VisuShrink\nmin found using {wavelet[ii]}")
        if not (opath is None):
            ff.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-skimage-wavelet-denoise-VisuShrink-wavelet-rmse.png"))

    return stats

def plotPSNR(paths):
    from skimage.metrics import peak_signal_noise_ratio
    psnr = []
    fnames = []
    if isinstance(paths,str):
        paths = glob(paths)

    for fn in paths:
        data = pd.read_excel(fn)
        psnr.append(peak_signal_noise_ratio(data.fil_torque/data.fil_torque.max(),data.torque/data.torque.max()))
        fnames.append(os.path.splitext(os.path.basename(fn))[0])
    f,ax = plt.subplots()
    ax.plot(psnr,'x')
    return f

def getRecDiscreteWavelets(th=15,inv=False):
    wv = []
    for w in pywt.wavelist(kind="discrete"):
        # sym and db are recommended for denoising
        if ('sym' in w) or ('db' in w):
            # if there's a dot in the name add to list
            if '.' in w:
                wv.append(w)
                continue
            # attempt to convert last two char to number
            try:
                lvl = int(w[-2:])
            except ValueError:
                lvl = int(w[-1:])
            if inv:
                if lvl<=th:
                    wv.append(w)
            else:
                if lvl>=th:
                    wv.append(w)
    return wv

def plotStats(stats,target='PSNR'):
    import seaborn as sns
    for i,g in stats.groupby('Filename'):
        g.reset_index(inplace=True)
        plt.figure()
        ax = sns.scatterplot(g,x='Wavelet',y='PSNR',hue='Denoising Method')
        ax.set_title(g['Filename'].unique()[0])

# from https://www.youtube.com/watch?v=w0rOvNJW58o
def apply_convolution(sig, window):
    from scipy.signal import convolve
    conv = np.repeat([0., 1., 0.], window)
    print(sig)
    filtered = signal.convolve(sig, conv, mode='same') / window
    return filtered

def denoiseConvolutionSame(paths,wsizes=None):
    from skimage.metrics import peak_signal_noise_ratio
    if isinstance(paths,str):
        paths = glob(paths)
    psnr = []
    for i,fn in enumerate(paths):
        print(fn)
        # load torque signal
        # convert to mA
        unfiltered = pd.read_excel(fn).torque
        ref = unfiltered / unfiltered.max()
        print("unfiltered",unfiltered.shape)
        # create window sizes
        wsizes = list(range(1,len(unfiltered)//2,1))
        f,ax = plt.subplots()
        ax.plot(unfiltered.values,'b-',label="Unfiltered")
        for w in wsizes:
            #denoised = unfiltered.apply(lambda srs: apply_convolution(srs, w))
            conv = np.repeat([0., 1., 0.], w)
            denoised = signal.convolve(unfiltered, conv, mode='same') / w
            ax.plot(denoised,'-',label=f"W={w}")
            # calculate PSNR against no filtering     
            psnr.append(peak_signal_noise_ratio(ref,denoised/denoised.max()))
        f.suptitle(os.path.splitext(os.path.basename(fn))[0])
        # plot PSNR
        fp,axp = plt.subplots()
        axp.plot(psnr)
        fp.suptitle(os.path.splitext(os.path.basename(fn))[0]+" PSNR")

def rmseDiffLengths(A,B):
    from scipy.interpolate import UnivariateSpline
    xa = np.arange(len(A))
    As = UnivariateSpline(xa, A)
    xb = np.arange(len(B))
    Bs = UnivariateSpline(xb, B)
    # create longer x
    ml = max([len(A),len(B)])
    xa = np.linspace(xa.min(),xa.max(),ml)
    xb = np.linspace(xb.min(),xb.max(),ml)
    # generate new vectors
    ya = As(xa)
    yb = Bs(xb)
    # find rmse
    return np.sqrt(np.mean((ya-yb)**2))

def pulsingWindow(w):
    return np.repeat([0., 1., 0.], w)

def chebWinAt(w,at=100):
    return signal.windows.chebwin(w,at)

def kaiserWinAt(w,beta=14.0):
    return signal.windows.kaiser(w,beta=beta)

def taylorWinAt(w,at=125,nbar=50):
    return signal.windows.taylor(w, nbar=nbar, sll=at, norm=False)

def denoiseConvAgainstPower(torque_new_path,power_new_path,torque_worn_path,power_worn_path,**kwargs):
    '''
        Denoise torque by convolving against different window sizes

        Provide windowing function via window_fn. If not specified it defaults to:

        lambda w : np.repeat([0., 1., 0.], w)

        Output save path for files set via output_path. If None, nothing is saved.
    '''
    new_sr = kwargs.get("upsample",100)

    best_wsize = []
    ms = kwargs.get("markersize",2)
    mms = ms*8

    wsizes = list(range(10,51,5))

    win_fn = kwargs.get("window_fn",pulsingWindow)

    opath = kwargs.get("output_path",None)

    power_full = read_old_power(power_worn_path,power_only=True)
    # iterate over each torque file
    for i,fn in enumerate(glob(torque_worn_path)):
        print("processing",fn)
        # load torque signal
        # convert to mA
        unfiltered = pd.read_excel(fn).torque/1000*48
        # upsample to 100 Hz
        if new_sr>0:
            unfiltered = raise_sr(unfiltered,20,new_sr = new_sr)
        # get power signal
        power_fn = power_full[power_full['Signal Index']==i]['P[W]'].values
        # upsample to 100 Hz
        if new_sr>0:
            power_fn = raise_sr(power_fn,10,new_sr = new_sr)
        # align signals
        torque, power = align_signals(unfiltered,power_fn)

        rmse = []
        for w in wsizes:
            conv = 2*win_fn(w)
            denoised = signal.convolve(unfiltered, conv, mode='same') / w

            if not (opath is None):
                fw,axw = plt.subplots(constrained_layout=True)
                axw.plot(unfiltered,'b-',label="Original")
                axw.plot(denoised,'r-',label="Filtered")
                axw.set(xlabel="Index",ylabel="Torque (A)")
                fw.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} wsize {w}")
                fw.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-wsize-{w}.png"))
                plt.close(fw)
            
            #rmse.append(np.sqrt(np.mean((denoised-power_fn)**2)))
            rmse.append(rmseDiffLengths(denoised,power))

        # find the window where the RMSE is smallest
        ii = np.argmin(rmse)
        best_wsize.append(wsizes[ii])

        # plot the RMSE for each window size
        ff,ax = plt.subplots(constrained_layout=True)
        ax.plot(wsizes,rmse,'-',markersize=ms,markeredgewidth=ms//2)
        ax.plot(wsizes[ii],rmse[ii],'rx',markersize=mms,markeredgewidth=mms//2)
        
        wticks = np.array(wsizes)
        ax.set_xlim(wticks.min(),wticks.max())
        ax.set(xlabel="Window Size (index)",ylabel="RMSE",title=f"Effect of Window Size on RMSE min at wsize={wsizes[ii]} (Convolution)\n{win_fn.__name__}")
        
        ff.suptitle(os.path.splitext(os.path.basename(fn))[0]+" (Convolution)")
        if not (opath is None):
            ff.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-wsizes-rmse.png"))

    power_full = read_new_power(power_new_path,power_only=True)
    # iterate over each torque file
    for i,fn in enumerate(glob(torque_new_path)):
        print("processing",fn)
        # load torque signal
        # convert to mA
        unfiltered = pd.read_excel(fn).torque/1000*48
        # upsample to 100 Hz
        if new_sr>0:
            unfiltered = raise_sr(unfiltered,20,new_sr = new_sr)
        # get power signal
        power_fn = power_full[power_full['Signal Index']==i]['P[W]'].values
        # upsample to 100 Hz
        if new_sr>0:
            power_fn = raise_sr(power_fn,10,new_sr = new_sr)
        # align signals
        torque, power = align_signals(unfiltered,power_fn)

        #wsizes = list(range(1,len(torque)//2,1))

        rmse = []
        for w in wsizes:
            conv = 2*win_fn(w)
            denoised = signal.convolve(unfiltered, conv, mode='same') / w
            #rmse.append(np.sqrt(np.mean((denoised-power_fn)**2)))

            if not (opath is None):
                fw,axw = plt.subplots(constrained_layout=True)
                axw.plot(unfiltered,'b-',label="Original")
                axw.plot(denoised,'r-',label="Filtered")
                axw.set(xlabel="Index",ylabel="Torque (A)")
                fw.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} wsize {w}")
                fw.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-wsize-{w}.png"))
                plt.close(fw)
            
            rmse.append(rmseDiffLengths(denoised,power))

        # find the window where the RMSE is smallest
        ii = np.argmin(rmse)
        best_wsize.append(wsizes[ii])

        # plot the RMSE for each window size
        ff,ax = plt.subplots(constrained_layout=True)
        ax.plot(wsizes,rmse,'-',markersize=ms,markeredgewidth=ms//2)
        ax.plot(wsizes[ii],rmse[ii],'rx',markersize=mms,markeredgewidth=mms//2)
        
        wticks = np.array(wsizes)
        ax.set_xlim(wticks.min(),wticks.max())
        ax.set(xlabel="Window Size (index)",ylabel="RMSE",title=f"Effect of Window Size on RMSE min at wsize={wsizes[ii]} (Convolution)\n{win_fn.__name__}")
        ff.suptitle(os.path.splitext(os.path.basename(fn))[0]+" (Convolution)")

        if not (opath is None):
            ff.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-wsizes-rmse.png"))
            plt.close(ff)

    # plot the best size found for each file
    ff,aa = plt.subplots(constrained_layout=True)
    aa.plot(best_wsize,'x',markersize=10,markeredgewidth=10//2)
    paths = glob(torque_new_path) + glob(torque_worn_path)
    aa.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(fn))[0] for fn in paths],rotation=90)
    ff.suptitle(f"Best Convolution Window Size for Worn+New using Win Function {win_fn.__name__}")
    aa.set_ylabel("Convolution Window Size")
    if not (opath is None):
        ff.savefig(os.path.join(opath,f"convolution-{win_fn.__name__}-best-wsizes-rmse.png"))
        plt.close(ff)

def denoiseConvAgainstPowerScale(torque_new_path,power_new_path,torque_worn_path,power_worn_path,**kwargs):
    '''
        Denoise torque by convolving against a fixed window sizes AND scales

        Provide windowing function via window_fn. If not specified it defaults to:

        lambda w : np.repeat([0., 1., 0.], w)

        Output save path for files set via output_path. If None, nothing is saved.

        Inputs:
            torque_new_path : Wildcard path to torque signals from new tools
            power_new_path : Direct path to power analyzer signal for new tools
            torque_worn_path : Wildcard path to torque signals for worn tools
            power_worn_path : Direct path to power analyzer signal for worn tools
            upsample : New artificial sampling rate of signals
    '''
    import seaborn as sns
    from tslearn.metrics import dtw
    new_sr = kwargs.get("upsample",100)

    best_wsize = []
    best_scale = []

    best_wsize_og = []
    best_scale_og = []
    
    smallest_rmse = []
    smallest_rmse_og = []
    smallest_dtw = []
    smallest_dtw_power = []
    smallest_dtw_og = []
    smallest_dtw_power_og = []
    
    ms = kwargs.get("markersize",2)
    mms = ms*8
    # window sizes
    wsizes = np.array(list(range(5,85,5)))
    print("wsizes range",wsizes.min(),wsizes.max())
    # scaling applied to the convolution window
    scales = np.arange(1,3.1,0.1)
    print("scales range",scales.min(),scales.max())
    #scales = np.round(scales,decimals=1)
    # define search vectors
    ww,ss = np.meshgrid(wsizes,scales)
    ww = ww.flatten()
    ss = ss.flatten()

    win_fn = kwargs.get("window_fn",pulsingWindow)

    opath = kwargs.get("output_path",None)

    stats_cols = ['Filename',
                  "Upsampled Torque Max (W)","Upsampled Torque Min (W)","Upsampled Torque Mean (W)","Upsampled Torque Variance (W)","Upsampled Torque Signal Energy",
                  "Convolution Window Size","Scaling Factor","Window Name",
                  "Denoised Upsampled Torque Max (W)","Denoised Upsampled Torque Min (W)","Denoised Upsampled Torque Mean (W)","Denoised Upsampled Torque Variance (W)","Denoised Upsampled Torque Signal Energy",
                  "Denoised Original Torque Max (W)","Denoised Original Torque Min (W)","Denoised Original Torque Mean (W)","Denoised Original Torque Variance (W)","Denoised Original Torque Signal Energy",
                  "Power Filename","Power Index",
                  "RMSE Upsampled","RMSE (Original)",
                  "DTW Upsampled","DTW Upsampled Against Power",
                  "DTW (Original)","DTW Against Power (Original)"]

    stats = pd.DataFrame(columns=stats_cols)

    # initialize power vector
    power_full = read_new_power(power_new_path,power_only=True)
    power_path = power_new_path
    # initialize paths
    paths = glob(torque_new_path)
    fn = paths.pop(0)
    # set counter to 0
    i=0
    # gloval counter
    processed = 0
    # total number of paths
    total_paths = len(glob(torque_new_path) + glob(torque_worn_path))
    # iterate over each torque file
    while(processed < total_paths):
        print("processing",fn)
        fname = os.path.splitext(os.path.basename(fn))[0]
        # load torque signal
        # convert to mA
        unfiltered = pd.read_excel(fn).torque/1000*48
        filtq = pd.read_excel(fn).fil_torque/1000*48
        # upsample to 100 Hz
        if new_sr>0:
            unfiltered_up_sr = raise_sr(unfiltered,20,new_sr = new_sr)
        if new_sr>0:
            filtq_up_sr = raise_sr(filtq,20,new_sr = new_sr)
        # get power signal
        power_fn = power_full[power_full['Signal Index']==i]['P[W]'].values
        # upsample to 100 Hz
        if new_sr>0:
            power_fn_up_sr = raise_sr(power_fn,10,new_sr = new_sr)
        # align signals
        torque, power = align_signals(unfiltered_up_sr,power_fn_up_sr)
        _,filtq_up_sr = align_signals(power_fn_up_sr,filtq_up_sr)

##        f,ax = plt.subplots(constrained_layout=True)
##        ax.plot(torque,'b-',label="Torque")
##        ax.plot(power,'r-',label="Power")
##        ax.set(xlabel="Index",ylabel="Power (W)",title=f"{fname} Aligned Torque and Power Signal")
##        ax.legend()
##        plt.show()

        ## 2d metric matricies
        # rmse for upscaled
        rmse_mat = pd.DataFrame(data=np.zeros((wsizes.shape[0],scales.shape[0])),index=wsizes,columns=scales)
        # rmse for original
        rmse_mat_og = pd.DataFrame(data=np.zeros((wsizes.shape[0],scales.shape[0])),index=wsizes,columns=scales)
        # DTW for upscaled
        dtw_mat = pd.DataFrame(data=np.zeros((wsizes.shape[0],scales.shape[0])),index=wsizes,columns=scales)
        # DTW for upscaled against power
        dtw_mat_power = pd.DataFrame(data=np.zeros((wsizes.shape[0],scales.shape[0])),index=wsizes,columns=scales)
        
        # DTW for original
        dtw_mat_og = pd.DataFrame(data=np.zeros((wsizes.shape[0],scales.shape[0])),index=wsizes,columns=scales)
        # DTW for original against power
        dtw_mat_power_og = pd.DataFrame(data=np.zeros((wsizes.shape[0],scales.shape[0])),index=wsizes,columns=scales)
        
        # find the rmse of each value
        for w,s in zip(ww,ss):
            conv = s*win_fn(w)
            denoised_up_sr = signal.convolve(torque, conv, mode='same') / w
            rmse_mat[s][w] = rmseDiffLengths(denoised_up_sr,power)
            
            dtw_mat[s][w] = dtw(denoised_up_sr,torque)
            dtw_mat_power[s][w] = dtw(denoised_up_sr,power)

            # denoise the original res unfiltered signal
            denoised = signal.convolve(unfiltered, conv, mode='same') / w

            rmse_mat_og[s][w] = rmseDiffLengths(denoised,power_fn)
            
            dtw_mat_og[s][w] = dtw(denoised,unfiltered)
            dtw_mat_power_og[s][w] = dtw(denoised,power_fn)

            stats = pd.concat([stats,pd.DataFrame(data=[[fn,torque.max(),torque.min(),torque.mean(),torque.var(),np.sum(torque**2),
                                                       w,s,win_fn.__name__,
                                                       denoised_up_sr.max(),denoised_up_sr.min(),denoised_up_sr.mean(),denoised_up_sr.var(),np.sum(denoised_up_sr**2),
                                                       denoised.max(),denoised.min(),denoised.mean(),denoised.var(),np.sum(denoised**2),
                                                       power_path,i,
                                                       rmse_mat[s][w],rmse_mat_og[s][w],
                                                       dtw_mat[s][w],dtw_mat_power,
                                                       dtw_mat_og[s][w],dtw_mat_power_og[s][w]]],columns=stats_cols)])

        # find the window where the RMSE is smallest
        ii = rmse_mat.values.argmin()
        wi,si = np.unravel_index(ii,rmse_mat.shape)
        smallest_rmse.append(rmse_mat.values.min())
        # find best
        best_wsize.append(wsizes[wi])
        best_scale.append(scales[si])

        oo = rmse_mat_og.values.argmin()
        wio,sio = np.unravel_index(oo,rmse_mat_og.shape)
        smallest_rmse_og.append(rmse_mat_og.values.min())
        # find best
        best_wsize_og.append(wsizes[wio])
        best_scale_og.append(scales[sio])        

        # update vectors
        smallest_dtw.append(dtw_mat.values.min())
        smallest_dtw_power.append(dtw_mat_power.values.min())
        smallest_dtw_og.append(dtw_mat_og.values.min())
        smallest_dtw_power_og.append(dtw_mat_power_og.values.min())

        # plot the heatmap of rmse values (upscaled)
        f, ax = plt.subplots(constrained_layout=True,figsize=(13, 8))
        ax = sns.heatmap(rmse_mat,annot=True,fmt='.1f', linewidth=.5,ax=ax)
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} RMSE")
        #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.set_xticks(ax.get_xticks(),[f"{x:.1f}" for x in scales])
        ax.set(xlabel="Convolution Scale Factor",ylabel="Convolution Window Size")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-rmse-heatmap.png"))
            plt.close(f)

        # plot the heatmap of rmse values (original)
        f, ax = plt.subplots(constrained_layout=True,figsize=(13, 8))
        ax = sns.heatmap(rmse_mat_og,annot=True,fmt='.1f', linewidth=.5,ax=ax)
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} RMSE (Original Signals)")
        #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.set_xticks(ax.get_xticks(),[f"{x:.1f}" for x in scales])
        ax.set(xlabel="Convolution Scale Factor",ylabel="Convolution Window Size")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-rmse-heatmap-og.png"))
            plt.close(f)

        # plot the heatmap of DTW
        f, ax = plt.subplots(constrained_layout=True,figsize=(13, 8))
        ax = sns.heatmap(dtw_mat,annot=True,fmt='.1f', linewidth=.5,ax=ax)
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} DTW Similarity")
        #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.set_xticks(ax.get_xticks(),[f"{x:.1f}" for x in scales])
        ax.set(xlabel="Convolution Scale Factor",ylabel="Convolution Window Size")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-dtw-heatmap.png"))
            plt.close(f)

        # plot the heatmap of DTW power
        f, ax = plt.subplots(constrained_layout=True,figsize=(13, 8))
        ax = sns.heatmap(dtw_mat_power,annot=True,fmt='.1f', linewidth=.5,ax=ax)
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} DTW Similarity against Power")
        #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.set_xticks(ax.get_xticks(),[f"{x:.1f}" for x in scales])
        ax.set(xlabel="Convolution Scale Factor",ylabel="Convolution Window Size")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-dtw-power-heatmap.png"))
            plt.close(f)

        # plot the heatmap of DTW (Original)
        f, ax = plt.subplots(constrained_layout=True,figsize=(13, 8))
        ax = sns.heatmap(dtw_mat_og,annot=True,fmt='.1f', linewidth=.5,ax=ax)
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} DTW Similarity (Original)")
        #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.set_xticks(ax.get_xticks(),[f"{x:.1f}" for x in scales])
        ax.set(xlabel="Convolution Scale Factor",ylabel="Convolution Window Size")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-dtw-heatmap-og.png"))
            plt.close(f)

        # plot the heatmap of DTW power (Original)
        f, ax = plt.subplots(constrained_layout=True,figsize=(13, 8))
        ax = sns.heatmap(dtw_mat_power_og,annot=True,fmt='.1f', linewidth=.5,ax=ax)
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} DTW Similarity against Power (Original)")
        #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.set_xticks(ax.get_xticks(),[f"{x:.1f}" for x in scales])
        ax.set(xlabel="Convolution Scale Factor",ylabel="Convolution Window Size")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-dtw-power-heatmap-og.png"))
            plt.close(f)

        ## replot best denoised signal
        f,ax = plt.subplots()
        df = pd.read_excel(fn)
        # plot the original signal
        ax.plot(df.distance,unfiltered,'b-',label="Original")
        # plot the oversmoothed torque
        ax.plot(df.distance,filtq,'m-',label="FIL_TORQUE")
        # get the conv window
        conv = best_scale_og[-1]*win_fn(best_wsize_og[-1])
        # denoise signal
        denoised = signal.convolve(unfiltered, conv, mode='same') / best_wsize_og[-1]
        ax.plot(df.distance,denoised,'r-',label="Filtered")
        #tax = ax.twinx()
        power_pow = np.linspace(df.distance.min(),df.distance.max(),power_fn.shape[0])
        ax.plot(power_pow,power_fn,'g-',label="Power Analyzer")
        ax.legend()
        ax.set(xlabel="Position (mm*100)",ylabel="Torque (W)")
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} (Original)\nBest wsize {best_wsize_og[-1]} & scale {best_scale_og[-1]:.2f} according to Original RMSE")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-best-wsize-scale-og.png"))
            plt.close(f)

        f,ax = plt.subplots()
        df = pd.read_excel(fn)
        # plot the original signal
        ax.plot(df.distance,unfiltered,'b-',label="Original")
        # plot the oversmoothed torque
        ax.plot(df.distance,filtq,'m-',label="FIL_TORQUE")
        # get the conv window
        conv = best_scale[-1]*win_fn(best_wsize[-1])
        # denoise signal
        denoised = signal.convolve(unfiltered, conv, mode='same') / best_wsize[-1]
        ax.plot(df.distance,denoised,'r-',label="Filtered")
        #tax = ax.twinx()
        power_pow = np.linspace(df.distance.min(),df.distance.max(),power_fn.shape[0])
        ax.plot(power_pow,power_fn,'g-',label="Power Analyzer")
        ax.legend()
        ax.set(xlabel="Position (mm*100)",ylabel="Torque (W)")
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} (Original)\nBest wsize {best_wsize[-1]} & scale {best_scale[-1]:.2f} according to Upscaled RMSE")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-best-wsize-scale-og-using-upscaled.png"))
            plt.close(f)


        f,ax = plt.subplots()
        # resample the distance vector so it can be plotted against the other upsampled signals
        dist_up_sr = raise_sr(df.distance,20,new_sr = new_sr)
        # plot the upscaled signal
        ax.plot(dist_up_sr,unfiltered_up_sr,'b-',label="Original")
        # plot the upscaled oversmoothed torque
        ax.plot(dist_up_sr,filtq_up_sr,'m-',label="FIL_TORQUE")
        # get the conv window
        conv = best_scale[-1]*win_fn(best_wsize[-1])
        # denoise signal
        denoised = signal.convolve(unfiltered_up_sr, conv, mode='same') / best_wsize[-1]
        ax.plot(dist_up_sr,denoised,'r-',label="Filtered")
        #tax = ax.twinx()
        power_pow = np.linspace(df.distance.min(),df.distance.max(),power_fn_up_sr.shape[0])
        ax.plot(power_pow,power_fn_up_sr,'g-',label="Power Analyzer")
        ax.legend()
        ax.set(xlabel="Position (mm*100)",ylabel="Torque (W)")
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} (Upsampled)\nBest wsize {best_wsize[-1]} & scale {best_scale[-1]:.2f} according to Upscaled RMSE")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-best-wsize-scale-using-upscaled.png"))
            plt.close(f)

        f,ax = plt.subplots()
        # resample the distance vector so it can be plotted against the other upsampled signals
        dist_up_sr = raise_sr(df.distance,20,new_sr = new_sr)
        # plot the upscaled signal
        ax.plot(dist_up_sr,unfiltered_up_sr,'b-',label="Original")
        # plot the upscaled oversmoothed torque
        ax.plot(dist_up_sr,filtq_up_sr,'m-',label="FIL_TORQUE")
        # get the conv window
        conv = best_scale_og[-1]*win_fn(best_wsize_og[-1])
        # denoise signal
        denoised = signal.convolve(unfiltered_up_sr, conv, mode='same') / best_wsize_og[-1]
        ax.plot(dist_up_sr,denoised,'r-',label="Filtered")
        #tax = ax.twinx()
        power_pow = np.linspace(df.distance.min(),df.distance.max(),power_fn_up_sr.shape[0])
        ax.plot(power_pow,power_fn_up_sr,'g-',label="Power Analyzer")
        ax.legend()
        ax.set(xlabel="Position (mm*100)",ylabel="Torque (W)")
        f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Convolution {win_fn.__name__} (Upsampled)\nBest wsize {best_wsize_og[-1]} & scale {best_scale_og[-1]:.2f} according to Original RMSE")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-convolution-window-{win_fn.__name__}-best-wsize-scale.png"))
            plt.close(f)

        # if that was the last path of the list
        if len(paths)==0:
            # move onto worn paths
            paths = glob(torque_worn_path)
            # load power vector
            power_full = read_old_power(power_worn_path,power_only=True)
            power_path = power_worn_path
            # reset counter
            i=-1
        # get next path
        fn = paths.pop(0)
        # increment counter
        i+=1
        processed+=1

    # plot the best size found for each file
    ff,aa = plt.subplots(constrained_layout=True)
    aa.plot(best_wsize,'bx',markersize=10,markeredgewidth=10//2)
    taa = aa.twinx()
    taa.plot(best_scale,'rx',markersize=10,markeredgewidth=10//2)
    taa.set_ylabel("Convolution Scale Factor")
    paths = glob(torque_new_path) + glob(torque_worn_path)
    aa.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(fn))[0] for fn in paths],rotation=90)
    ff.suptitle(f"Best Convolution Window Size & Scale\nfor Worn+New using Window {win_fn.__name__}")
    aa.set_ylabel("Convolution Window Size")
    if not (opath is None):
        ff.savefig(os.path.join(opath,f"convolution-{win_fn.__name__}-best-wsizes-best-scale-rmse.png"))
        plt.close(ff)

    ff,aa = plt.subplots(constrained_layout=True)
    aa.plot(best_wsize_og,'bx',markersize=10,markeredgewidth=10//2)
    taa = aa.twinx()
    taa.plot(best_scale_og,'rx',markersize=10,markeredgewidth=10//2)
    taa.set_ylabel("Convolution Scale Factor")
    paths = glob(torque_new_path) + glob(torque_worn_path)
    aa.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(fn))[0] for fn in paths],rotation=90)
    ff.suptitle(f"Best Convolution Window Size & Scale (Original)\nfor Worn+New using Window {win_fn.__name__}")
    aa.set_ylabel("Convolution Window Size")
    if not (opath is None):
        ff.savefig(os.path.join(opath,f"convolution-{win_fn.__name__}-best-wsizes-best-scale-rmse-og.png"))
        plt.close(ff)

    # plot the smallest rmse values found for each file using the identified best parameter pair
    fr,ar = plt.subplots(constrained_layout=True)
    ar.plot(smallest_rmse,'bx',markersize=10,markeredgewidth=10//2)
    ar.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(fn))[0] for fn in paths],rotation=90)
    ar.set_ylabel("Smallest RMSE Error")
    fr.suptitle(f"Best RMSE (Upsampled)\nfor Worn+New using Window {win_fn.__name__}")
    if not (opath is None):
        fr.savefig(os.path.join(opath,f"convolution-{win_fn.__name__}-best-rmse.png"))
        plt.close(fr)

    fr,ar = plt.subplots(constrained_layout=True)
    ar.plot(smallest_rmse_og,'bx',markersize=10,markeredgewidth=10//2)
    ar.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(fn))[0] for fn in paths],rotation=90)
    ar.set_ylabel("Smallest RMSE Error")
    fr.suptitle(f"Best RMSE (Original)\nfor Worn+New using Window {win_fn.__name__}")
    if not (opath is None):
        fr.savefig(os.path.join(opath,f"convolution-{win_fn.__name__}-best-rmse-og.png"))
        plt.close(fr)

    fr,ar = plt.subplots(constrained_layout=True)
    ar.plot(smallest_dtw,'bx',markersize=10,markeredgewidth=10//2)
    ar.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(fn))[0] for fn in paths],rotation=90)
    ar.set_ylabel("Best DTW Similarity")
    fr.suptitle(f"Best DTW (Upsampled)\nfor Worn+New using Window {win_fn.__name__}")
    if not (opath is None):
        fr.savefig(os.path.join(opath,f"convolution-{win_fn.__name__}-best-dtw.png"))
        plt.close(fr)
    
    fr,ar = plt.subplots(constrained_layout=True)
    ar.plot(smallest_dtw_power,'bx',markersize=10,markeredgewidth=10//2)
    ar.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(fn))[0] for fn in paths],rotation=90)
    ar.set_ylabel("Best DTW Similarity Against Power")
    fr.suptitle(f"Best DTW Similarity Against Power (Upsampled)\nfor Worn+New using Window {win_fn.__name__}")
    if not (opath is None):
        fr.savefig(os.path.join(opath,f"convolution-{win_fn.__name__}-best-dtw-power.png"))
        plt.close(fr)

    fr,ar = plt.subplots(constrained_layout=True)
    ar.plot(smallest_dtw_og,'bx',markersize=10,markeredgewidth=10//2)
    ar.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(fn))[0] for fn in paths],rotation=90)
    ar.set_ylabel("Best DTW Similarity")
    fr.suptitle(f"Best DTW Similarity (Original)\nfor Worn+New using Window {win_fn.__name__}")
    if not (opath is None):
        fr.savefig(os.path.join(opath,f"convolution-{win_fn.__name__}-best-dtw-og.png"))
        plt.close(fr)
    
    fr,ar = plt.subplots(constrained_layout=True)
    ar.plot(smallest_dtw_power_og,'bx',markersize=10,markeredgewidth=10//2)
    ar.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(fn))[0] for fn in paths],rotation=90)
    ar.set_ylabel("Best DTW Similarity Against Power (Original)")
    fr.suptitle(f"Best DTW Against Power (Original)\nfor Worn+New using Window {win_fn.__name__}")
    if not (opath is None):
        fr.savefig(os.path.join(opath,f"convolution-{win_fn.__name__}-best-dtw-power-og.png"))
        plt.close(fr)

    return stats,best_wsize,best_scale

def plotBestWsizeScale(table):
    # search for the globally best windows based on RMSE
    best_global = stats_wins_stack.groupby('Filename').apply(lambda x : x.iloc[x.RMSE.idxmin()])

def plot_spect(z, times,frequencies,cmap="hot",norm=None):
    if cmap is None:
        cmap = plt.get_cmap('Greys')
    # or if cmap is a string, get the actual object
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # create the figure if needed
    fig, ax = plt.subplots(constrained_layout=True)
    #xx,yy = np.meshgrid(times,frequencies)
    #im = ax.contourf(xx,yy,z, None, cmap=cmap, vmin=z.min(),vmax=z.max())
    im = plt.pcolormesh(times,frequencies,z,cmap='hot')
    #im = ax.contourf(times,np.log2(1./frequencies),np.log2(z),extend="both",cmap=cmap)
    #im = ax.imshow(z,norm=LogNorm(),aspect="auto",cmap=cmap)
    try:
        plt.colorbar(im)
    except:
        print("skipping adding colorbar due to exception")
        print("has nan",np.isnan(z).any())
    # make it prettier
    ax.set_yscale('log')
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(frequencies.min(), frequencies.max())

    return ax

def lueberring_drop_scales(fmin=3,fmax=6):
    # drop artifact happens from 3hz +
    return 1./np.arange(fmin,fmax,20)

def getCWAtBC(bandwidth,cfreq,order=None):
    # check inputs
    if isinstance(bandwidth,(int,float)):
        bandwidth = [bandwidth,]
    if isinstance(cfreq,(int,float)):
        cfreq = [cfreq,]
    if isinstance(order,int):
        order = [order,]
    # set flag to check of order was given
    has_order = not (order is None)
    new_wvs = []
    # iterate over families that req a bandwidth and central freq, sometimes order
    for fam,fmt in zip(['cmor','shan','fpsp'],[lambda b,c : f"cmor{b:.1f}-{c:.1f}",lambda b,c : f"shan{b:.1f}-{c:.1f}"]):#,lambda m,b,c : f"fpsp{m}-{b:.1f}-{c:.1f}"]):
        for b in bandwidth:
            for c in cfreq:
                print(fam,b,c)
                # check that order was given
                if fam in ['fpsp']:
                    if has_order:
                        for m in order:
                            print(m)
                            new_wvs.append(fmt(m,b,c))
                else:
                    new_wvs.append(fmt(b,c))
    # iterate over families that req only an order
    if has_order:
        for fam,fmt in zip(['gaus','cgau'],[lambda p : f"gaus{p}",lambda p : f"cgau{p}"]):
            for p in order:
                new_wvs.append(fmt(p))
    return new_wvs

def torqueCWT(torque_path,wavelet='all',detrend=True,normalize=True,**kwargs):
    from scipy import signal
    import pywt
    import scaleogram as scg
    # sampling rate
    fs = 20.0
    dt = 1.0/fs

    if isinstance(torque_path,str):
        torque_path = glob(torque_path)
    if isinstance(wavelet,str):
        if wavelet == 'all':
            input(f"Pywt supports {len(pywt.wavelist(kind='discrete'))} wavelets. Press ENTER to continue")
            wavelet = pywt.wavelist(kind='continuous')
        else:
            wavelet = [wavelet,]

    opath = kwargs.get("output_path","lueberring specs")

    using_scales = False
    using_freqs = False
    if "scales" in kwargs:
        scales=kwargs["scales"]
        using_scales = True
        print("using scales",using_scales)
    elif "target_freqs" in kwargs:
        freqs_list = kwargs["target_freqs"]
        using_freqs = True
        print("using freqs",using_freqs)
    for fn in torque_path:
        unfiltered = pd.read_excel(fn).torque
        ## from https://gist.github.com/MiguelonGonzalez/00416cbf3d7f3eab204766961cf7c8fb
        N = len(unfiltered)
        times = np.arange(N) * dt
        # detrend linearly
        if detrend:
            unfiltered = signal.detrend(unfiltered,type="linear")
        # norm via std
        if normalize:
            stddev = unfiltered.std()
            unfiltered = unfiltered / stddev
        # generate scales for testing
        if using_scales:
            if isinstance(scales,int):
                nOctaves = np.log2(2*np.floor(N/2.0)).astype("int")
                scales_list = 2**np.arange(1, nOctaves, 1.0/scales)
                #scales_list = scales_list[len(scales_list)//2:]
            else:
                scales_list = scales
        # for each wavelet
        for w in wavelet:
            # apply cwt
            print(w)
            if using_freqs:
                scales_list = pywt.frequency2scale(w,freqs_list)/20.0
            try:
                coef, freqs=pywt.cwt(unfiltered,scales_list,w,sampling_period=1./20.)
            except AttributeError:
                print(f"Skipping {w} as it's not complex")
                continue
##            # convert scales to frequencies
##            frequencies = pywt.scale2frequency(w, scales_list) / dt
##            # log norm coefficients
##            z = 20*np.log10(np.abs(coef))
##            # just look at the top half of the freqs
##            z = z[frequencies>(frequencies.max()/2),:]
##            frequencies = frequencies[frequencies>(frequencies.max()/2)]
##            scaleogram.cws(
##            # occasionally gets an 
##            if (z.shape[0])==0:
##                continue
            # plot the scaleogram
            ax = plot_spect(20*np.log10(np.abs(coef)),times,freqs)
            # plot the original signal to compare behaviours
            tax = ax.twinx()
            tax.plot(times,unfiltered,'g-')
            tax.set_ylabel("Torque (A)")
            # set the labels
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)")

##            ax = scg.cws(times,unfiltered,scales_list,w,coikw={'alpha':0.5,'hatch':'/'},yscale='log',
##                         cmap='hot',title=f"{os.path.splitext(os.path.basename(fn))[0]} Scaleogram {w}")
            ax.figure.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]} Scaleogram {w}")
            if not (opath is None):
                try:
                    ax.figure.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-wavelet-scaleogram-{w}.png"))
                except ValueError:
                    print(f"Failed to save fig due to float NaN error")
                plt.close(ax.figure)

def icwt(coef,scales,wtype='morl'):
    ''' inverse continuous wavelet from https://github.com/PyWavelets/pywt/issues/328 '''
    import pywt
    mwf = pywt.ContinuousWavelet(wtype).wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

    r_sum = np.transpose(np.sum(np.transpose(coef)/ scales ** 0.5, axis=-1))
    return r_sum * (1 / y_0)

def filterCWS(data,filters,time=None,scales=None,wavelet='morl',T = 1/1e5,period=1,new_mag='min',periods=None):
    #from scaleogram.wfun import fastcwt
    # check that filters are given
    if not filters:
        raise ValueError("Filters must be not None!")
    # check if length of filters is non-zero
    if len(filters)==0:
        raise ValueError("List of filters must have entries!")
    if time is None:
        # creat time vector
        time = np.arange(0.0,len(data)*T,T,dtype='float16')
    # if scales aren't specified
    if scales is None:
        # create scales based on time
        scales = np.arange(1, min(len(time)/10, 100))
    # calculate CWT
    coefs,_ = pywt.cwt(data,scales,wavelet,T)
    # iterate over the target scales
    for sc in filters:
        # if the user gave a two element range
        # filter using the range
        if len(sc) ==2:
            si=np.where((scales >=sc[0]) & (scales <= sc[1]))[0]
        # if a single element then filter from value onwards
        else:
            si=np.where(scales >=sc[0])[0]
        # set the target scale range to the target value
        if new_mag == 'min':
            coefs[si,:] = coefs.min()
        elif new_mag == 'max':
            coefs[si,:] = coefs.max()
        else:
            coefs[si,:] = new_mag
        coefs[si,:] = 0.5*coefs[si,:]
    # perform inverse
    return icwt(coefs,scales)

def plotWindowFreqResponse(win_fns):
    #from scipy.fft import fft, fftshift
    from scipy.fft import rfft
    f,ax = plt.subplots(nrows=2,ncols=len(win_fns),constrained_layout=True)
    for i,fn in enumerate(win_fns):
        window = fn(51)
        A = rfft(window, 2048) / (len(window)/2.0)
        freq = np.linspace(0., 0.5, len(A))
        #freq = np.linspace(-0.5, 0.5, len(A))
        #response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
        response = 20 * np.log10(np.abs(A / abs(A).max()))
        ax[0,i].plot(window)
        ax[0,i].set(xlabel="Sample",ylabel="Amplitude",title=fn.__name__)
        ax[1,i].plot(freq,response)
        ax[1,i].set(xlabel="Normalized Frequency\n[cycles per sample]",ylabel="Normalized magnitude [dB]",title=f"Freq. Response of\n{fn.__name__} window")

def plotButterResponse(order,fc=0.1):
    b, a = signal.butter(order, fc, 'low', analog=False)
    w, h = signal.freqz(b, a)
    f,ax = plt.subplots()
    #ax.semilogx(w, 20 * np.log10(abs(h)))
    ax.plot(w, 20 * np.log10(abs(h)))
    ax.set(xlabel='Frequency [radians / second]',ylabel='Amplitude [dB]',title=f'Butterworth filter frequency response order={order}')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    #ax.axvline(100, color='green') # cutoff frequency
    return f

# from https://stackoverflow.com/a/63177397
def signaltonoise(a : np.ndarray, axis: int=0, ddof: int=0)-> float:
    '''
        Caclculate Signal-to-Noise Ratio of the signal

        Inputs:
            a : Numpy array
            axis : Along wtih axis of the array to calculate
            ddof : Degree of freedom

        Returns calculated SNR
    '''
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def applyFiltFilt(torque_path:(str,list),fstep: float=0.01,fmin: float=0.1, opath: str="filtfilt" ,**kwargs)->None:
    '''
        Apply a digital filter forwards and backwards across the signal

        The result is a signal with zero phase (no delay)

        Currently a Butterworth filter is used.

        The frequency is normalized with respect to Nyquist frequency

        On the recommendation of the docs scipy.signal.sosfiltfilt is used.

        Inputs:
            torque_path : Wildcard path string or list of paths.
            fstep : Steps in normalized frequency. Default 0.01
            fmin : Minimum normalized frequency to start from. Default 0.1.
            opath : Output folder to save plots to. Default filtfilt.
            order : Model order to use. Default 8.
    '''
    if isinstance(torque_path,str):
        torque_path = glob(torque_path)
    snr_list = []
    fc_list = np.arange(fmin,1.0,fstep)
    order = kwargs.get("order",8)
    f = plotButterResponse(order)
    f.savefig(os.path.join(opath,f"butterworth-frequency-response-order-{order}.png"))
    plt.close(f)
    #nq_freq = 20/2.0
    for fn in torque_path:
        fname = os.path.splitext(os.path.basename(fn))[0]
        unfiltered = pd.read_excel(fn)
        snr_ref = signaltonoise(unfiltered.torque)
        snr_list.clear()
        # iterate over cut off freq
        for fc in fc_list:
            sos = signal.butter(order,fc,output='sos')
            #y = signal.filtfilt(b,a,unfiltered.torque,method='gust')
            y = signal.sosfiltfilt(sos,unfiltered.torque)
            #snr = 10 * np.log10(y / unfiltered.fil_torque)
            snr = signaltonoise(y)
            snr_list.append(snr)
            # plot the filtered signal against the unfiltered signal
            f,ax = plt.subplots()
            ax.plot(unfiltered.distance,unfiltered.torque,'b-',label="Unfiltered")
            ax.plot(unfiltered.distance,y,'r-',label="Filtered")
            ax.plot(unfiltered.distance,unfiltered.fil_torque,'g-',label="Oversmoothed")
            ax.legend()
            ax.set(xlabel="Index",ylabel="Torque (A)",title=f"{fname} Butterworth {order} order fc={fc:.2f}")
            f.savefig(os.path.join(opath,f"{fname}-filtfilt-butter-order-{order}-fc-{fc:.2f}.png"))
            plt.close(f)

        # plot the SNR of the filtered signal
        f,ax = plt.subplots()
        ax.plot(fc_list,snr_list,'b-',label="Filtered SNR")
        ax.hlines(snr_ref,fc_list.min(),fc_list.max(),'r','dashed',label="Ref SNR")
        ax.legend()
        ax.set(xlabel="Cutoff Frequency",ylabel="Signal-to-Noise Ratio",title=f"SNR for {fname} applying\nButterworth Filter {order} order fc={fc:.2f}")
        f.savefig(os.path.join(opath,f"{fname}-filtfilt-butter-order-{order}-fc-{fc:.2f}-snr.png"))
        plt.close(f)

def plotEachSNR(torque_path:(str,list)):
    if isinstance(torque_path,str):
        torque_path = glob(torque_path)
    fname_list = []
    snr_list = []
    for fn in torque_path:
        fname = os.path.splitext(os.path.basename(fn))[0]
        fname_list.append(fname)
        snr_list.append(signaltonoise(pd.read_excel(fn).torque))
    f,ax = plt.subplots(constrained_layout=True)
    ax.plot(snr_list,'bx')
    ax.set_xticks(list(range(len(fname_list))),fname_list,rotation=90)
    ax.set(xlabel="Filename",ylabel="Signal to Noise Ratio",title="SNR for each file")
    return f

def splitByDistance(df,p=1.0):
    '''
        Split the dataframe into groups based on distance periods

        Iterates over periods in pairs and masks the dataframe to the rows that are within the target range

        p0 >= distance > p1

        Inputs:
            df : Lueberring dataframe loaded using pandas
            p : Period in mm

        Returns list of dataframes 
    '''
    pp = p*100
    periods = np.arange(df.distance.min(),df.distance.max()+pp,pp)
    return [df.loc[(df.distance>=p0) & (df.distance<p1)] for p0,p1 in zip(periods,periods[1:])]

def getMaxTorqueChange(groups):
    ''' Find abs max change in torque for each group '''
    return [abs(gp.torque.max()-gp.torque.min()) for gp in groups]

def getTorqueEdgeChanges(groups):
    ''' Finds the abs torque difference between last and first values in each group '''
    return [abs(gp.torque.iloc[-1] - gp.torque.iloc[0]) for gp in groups]

def filterButter(data,fc=0.2,order=8):
    sos = signal.butter(order,fc,output='sos')
    #y = signal.filtfilt(b,a,unfiltered.torque,method='gust')
    y = signal.sosfiltfilt(sos,data)
    return y

def dynamicProgBPS(torque_path:(str,list),order:int=8,fc:float=0.2,opath:str="butterbreak",**kwargs):
    '''
        Smooth and apply dynamic programming breakpoint detection to each file in the path.

        Generates figures showing the original signal, the filtered signal, the breakpoints and the oversmoothed
        signal from the Lueberring data.

        Inputs:
            torque_path : Wildcard path or list of specific paths
            order : Butterworth model order. Default 8
            fc : Nyquist cutoff frequency. Default 0.2.
            opath : Output folder where to save figures. Default butterbreak.
            distance_metric : Distance metric used in dynp calculation. Supported l1, l2 and rbf. Default l2.
            nbps : Number of breakpoints to search for. Default 4.
            min_size : Minimum distance between change points in terms of number of data points. Default 3.
            jump : Creates grid of change points at every jump, 2*jump, 3*jump etc. point. Default 10.
            
    '''
    import ruptures as rpt
    # convert the wildpath to list of paths
    if isinstance(torque_path,str):
        torque_path = glob(torque_path)
    # get filtering function
    filt_fn = kwargs.get("filter",filterButter)

    for fn in torque_path:
        fname = os.path.splitext(os.path.basename(fn))[0]
        # load data
        df = pd.read_excel(fn)
        unfiltered = df.torque
        dist = df.distance
        # filter the data
        filtered = filterButter(unfiltered,fc,order)
        # perform breakpoint detection using the specified settings
        algo = rpt.Dynp(model=kwargs.get("distance_metric","l2"), min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",5)).fit(filtered)
        my_bkps = np.array(algo.predict(n_bkps=kwargs.get("nbps",4)))-1
        #rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
        f,ax = plt.subplots()
        ax.plot(df.distance,unfiltered,'b-',label="Unfiltered")
        ax.plot(dist,filtered,'r-',label="Filtered")
        ax.vlines(df.distance.values[my_bkps],unfiltered.min(),unfiltered.max(),colors='k',linestyles='dashed',label="Breakpoints")
        ax.legend()
        ax.set(xlabel="Distance (mm*100)",ylabel="Torque (mA)",title=f"{fname} {order} order Butterworth filter fc={fc:.2f}")
        if not (opath is None):
            f.savefig(os.path.join(opath,f"{fname}-dyn-prog-butterworth-order-{order}-fc-{fc}-breakpoints.png"))
        plt.close(f)

        # upsample the data using thbe
        if kwargs.get("upsample",True):
            # raise sampling rate
            filtered = raise_sr(filtered,20,new_sr=100,smoothing=0.5)
            dist = raise_sr(dist,20,new_sr=100,smoothing=0.5)
            # find breakpoints
            algo = rpt.Dynp(model=kwargs.get("distance_metric","l2"), min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",5)).fit(filtered)
            my_bkps = np.array(algo.predict(n_bkps=kwargs.get("nbps",4)))-1
            
            f,ax = plt.subplots()
            ax.plot(df.distance,unfiltered,'b-',label="Unfiltered")
            ax.plot(dist,filtered,'r-',label="Filtered")
            ax.vlines(dist[my_bkps],unfiltered.min(),unfiltered.max(),colors='k',linestyles='dashed',label="Breakpoints")
            ax.legend()
            ax.set(xlabel="Distance (mm*100)",ylabel="Torque (mA)",title=f"{fname} {order} order Butterworth filter fc={fc:.2f}")
            if not (opath is None):
                f.savefig(os.path.join(opath,f"{fname}-dyn-prog-butterworth-order-{order}-fc-{fc}-breakpoints-upsampled.png"))
            plt.close(f)

def dynamicProgBPSSlider(torque_path:str,order:int=8,fc:float=0.2,**kwargs):
    from matplotlib.widgets import Slider, Button
    import ruptures as rpt
    # load single file
    fname = os.path.splitext(os.path.basename(torque_path))[0]
    # load data
    df = pd.read_excel(torque_path)
    unfiltered = df.torque
    unf_min = unfiltered.min()
    unf_max = unfiltered.max()

    dist = df.distance
    # filter the data
    filtered = filterButter(unfiltered,fc,order)
    if kwargs.get("upsample",True):
        # raise sampling rate
        filtered = raise_sr(filtered,20,new_sr=100,smoothing=0.5)
        dist = raise_sr(dist,20,new_sr=100,smoothing=0.5)

    fig,ax = plt.subplots(constrained_layout=False)
    ax.set(xlabel="Distance (mm*100)",ylabel="Torque (mA)",title=fname)
    line, = ax.plot(dist,filtered,'b-')

    algo = rpt.Dynp(model=kwargs.get("distance_metric","l2"), min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",5)).fit(filtered)
    my_bkps = np.array(algo.predict(n_bkps=kwargs.get("nbps",4)))-1

    bklines = ax.vlines(dist[my_bkps],unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")  
    fig.subplots_adjust(left=0.25, bottom=0.25)

    axmnps = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    nbps_slider = Slider(
        ax=axmnps,
        label='nbps',
        valmin=1,
        valmax=20,
        valstep=1,
        valinit=kwargs.get("nbps",4),
        valfmt="%d"
    )
    
    axminsz = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    minsz_slider = Slider(
        ax=axminsz,
        label='Min Size',
        valmin=1,
        valmax=100,
        valstep=1,
        valfmt="%d"
    )

    # Make a vertically oriented slider to control the amplitude
    axjump = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    jump_slider = Slider(
        ax=axjump,
        label="Jump",
        valmin=1,
        valmax=20,
        valstep=1,
        valfmt="%d"
    )

    def update(val):
        jump = int(jump_slider.val)
        min_sz = int(minsz_slider.val)
        print(min_sz,jump)
        algo = rpt.Dynp(model=kwargs.get("distance_metric","l2"), min_size=min_sz, jump=jump).fit(filtered)
        my_bkps = np.array(algo.predict(n_bkps=int(nbps_slider.val)))-1
        ax.collections[0].remove()
        bklines = ax.vlines(dist[my_bkps],unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")  
        fig.canvas.draw_idle()

    # register the update function with each slider
    nbps_slider.on_changed(update)
    minsz_slider.on_changed(update)
    jump_slider.on_changed(update)

    plt.show()

def kernelBPSSlider(torque_path:str,order:int=8,fc:float=0.2,**kwargs):
    from matplotlib.widgets import Slider, Button
    import ruptures as rpt
    # load single file
    fname = os.path.splitext(os.path.basename(torque_path))[0]
    # load data
    df = pd.read_excel(torque_path)
    unfiltered = df.torque
    unf_min = unfiltered.min()
    unf_max = unfiltered.max()

    dist = df.distance
    # filter the data
    filtered = filterButter(unfiltered,fc,order)
    if kwargs.get("upsample",True):
        # raise sampling rate
        filtered = raise_sr(filtered,20,new_sr=100,smoothing=0.5)
        dist = raise_sr(dist,20,new_sr=100,smoothing=0.5)

    fig,ax = plt.subplots(constrained_layout=False)
    ax.set(xlabel="Distance (mm*100)",ylabel="Torque (mA)",title=f"{fname} Kernel BPS")
    line, = ax.plot(dist,filtered,'b-')

    params = {}
    if kwargs.get("kernel","cosine") == "rbf":
        params = {"gamma":1e-2}
    
    algo = rpt.KernelCPD(kernel=kwargs.get("kernel","linear"), params=params,min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",5)).fit(filtered)
    my_bkps = np.array(algo.predict(n_bkps=kwargs.get("nbps",4)))-1

    bklines = ax.vlines(dist[my_bkps],unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")  
    fig.subplots_adjust(left=0.25, bottom=0.25)

    axmnps = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    nbps_slider = Slider(
        ax=axmnps,
        label='nbps',
        valmin=1,
        valmax=20,
        valstep=1,
        valinit=kwargs.get("nbps",4),
        valfmt="%d"
    )
    
    axminsz = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    minsz_slider = Slider(
        ax=axminsz,
        label='Min Size',
        valmin=1,
        valmax=100,
        valstep=1,
        valfmt="%d"
    )

    # Make a vertically oriented slider to control the amplitude
    axjump = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    jump_slider = Slider(
        ax=axjump,
        label="Jump",
        valmin=1,
        valmax=20,
        valstep=1,
        valfmt="%d"
    )

    def update(val):
        jump = int(jump_slider.val)
        min_sz = int(minsz_slider.val)
        print(min_sz,jump)
        algo = rpt.KernelCPD(kernel=kwargs.get("kernel","linear"), params=params,min_size=int(minsz_slider.val), jump=int(jump_slider.val)).fit(filtered)
        my_bkps = np.array(algo.predict(n_bkps=int(nbps_slider.val)))-1
        ax.collections[0].remove()
        bklines = ax.vlines(dist[my_bkps],unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")  
        fig.canvas.draw_idle()

    # register the update function with each slider
    nbps_slider.on_changed(update)
    minsz_slider.on_changed(update)
    jump_slider.on_changed(update)

    plt.show()

def peltBPSSlider(torque_path:str,order:int=8,fc:float=0.2,**kwargs):
    from matplotlib.widgets import Slider, Button
    import ruptures as rpt
    # load single file
    fname = os.path.splitext(os.path.basename(torque_path))[0]
    # load data
    df = pd.read_excel(torque_path)
    unfiltered = df.torque
    unf_min = unfiltered.min()
    unf_max = unfiltered.max()

    dist = df.distance
    # filter the data
    filtered = filterButter(unfiltered,fc,order)
    if kwargs.get("upsample",True):
        # raise sampling rate
        filtered = raise_sr(filtered,20,new_sr=100,smoothing=0.5)
        dist = raise_sr(dist,20,new_sr=100,smoothing=0.5)

    fig,ax = plt.subplots(constrained_layout=False)
    ax.set(xlabel="Distance (mm*100)",ylabel="Torque (mA)",title=f"{fname} PELT BPS")
    line, = ax.plot(dist,filtered,'b-')
    
    algo = rpt.Pelt(model=kwargs.get("model","l2"),min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",5)).fit(filtered)
    my_bkps = np.array(algo.predict(pen=kwargs.get("pen",4)))-1

    bklines = ax.vlines(dist[my_bkps],unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")  
    fig.subplots_adjust(left=0.25, bottom=0.25)

    axpen = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    pen_slider = Slider(
        ax=axpen,
        label='pen',
        valmin=1,
        valmax=20,
        valstep=1,
        valinit=kwargs.get("pen",4),
        valfmt="%d"
    )
    
    axminsz = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    minsz_slider = Slider(
        ax=axminsz,
        label='Min Size',
        valmin=1,
        valmax=100,
        valstep=1,
        valfmt="%d"
    )

    # Make a vertically oriented slider to control the amplitude
    axjump = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    jump_slider = Slider(
        ax=axjump,
        label="Jump",
        valmin=1,
        valmax=20,
        valstep=1,
        valfmt="%d"
    )

    def update(val):
        jump = int(jump_slider.val)
        min_sz = int(minsz_slider.val)
        print(min_sz,jump)
        algo = rpt.Pelt(model=kwargs.get("model","l2"),min_size=int(minsz_slider.val), jump=int(jump_slider.val)).fit(filtered)
        my_bkps = np.array(algo.predict(pen=int(pen_slider.val)))-1
        ax.collections[0].remove()
        bklines = ax.vlines(dist[my_bkps],unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")  
        fig.canvas.draw_idle()

    # register the update function with each slider
    pen_slider.on_changed(update)
    minsz_slider.on_changed(update)
    jump_slider.on_changed(update)

    plt.show()

# convert distance from mm to samples
def distToSamples(dist:float,feed_rate:float,rpm:float,sample_rate:float)->int:
    '''
       Convert the distance from mm to number of samples

       FR (mm/s) = FR (mm/rev) * (rpm/60)
       SP (mm/sample) = FR (mm/s) / SR (samples/s)
       D (samples) = round(D (mm) * SP(mm/sample))

       Inputs:
           dist : Distance to convert in mm
           feed_rate : Advance rate of the tool in mm/rev
           rpm : Angular velocity in rot per min.
           sample_rate : Sampling rate of the signal
    '''
    # convert feed rate from mm/rev -> mm/s
    fr_mms = feed_rate * (rpm/60.0)
    # calculate spatial resolution
    spatial_res = fr_mms/sample_rate
    return int(round(dist/spatial_res,0))

def findPlotBpsCalcMinSize(torque_path:str,shapelet:ToolShapelet,**kwargs):
    '''
        Finds and plots the breakpoints in the signal by calculating the min_size and jump using tool
        geometry and process parameters.

        The ToolShapelet instance is used to convert the segments from millimetres to number of samples.

        The min distance between breakpoints is calculated by processing the segments into a single value.
        By default the smallest segment is selected but the user can provide a different method via select_min_dist.
        
        The spacing between breakpoints on the grid is set by the parameter jump. The spacing is converted from
        distance in mm to number of samples using the paramters feed_rate, rpm and sr. The user controls this
        parameter kind of like how much wiggle room they want for possible breakpoints.

        The flags return_data and return_samples control if the found breakpoints are returned. The flag return_data
        is evaluated first. If False, then return_samples is evaluated. If both are False, then no data is returned

        Inputs:
            torque_path : Path to torque file
            shapelet : Instance of ToolShapelet describing the tool used in the signal
            feed_rate : Advance rate used in mm/rev. Default 0.04
            rpm : Angular velocity for file in rpm. Default 4500 rpm.
            sr : Sampling rate of the file in Hz. Default 100.
            jump : Spacing between the breakpoints to be evaluated in mm. Default 0.1 mm
            select_min_dist : Function for selecting the min size from the tool segments. Default min.
            model : Cost model or kernel model. See supported models for set bps_method. Default l2.
            bps_method : Breakpoint method. Default Dynp.
            fc : Nyquist cutoff frequency for Butterworth filter. Default 0.2.
            order : Model order of the Butterworth filter. Default 8.
            upsample : Upsample the signal to a new sampling rate. Default True.
            new_sr : New sampling rate to upsample to. Default 100 Hz.
            mask_th : Masking window size to remove backtracking
            plot_grad : Plot the gradient. Default True.
            return_data : Flag to return the breakpoints in mm alongside the figure. Default True
            return_samples : Flag to return the breakpoints in samples alongside the figure. Default True.

        Returns figure
    '''
    import ruptures as rpt
    from dataparser import loadSetitecXls
    from scipy.signal import find_peaks, peak_widths
    ## get parameters
    fr = kwargs.get("feed_rate",0.04)
    rpm = kwargs.get("rpm",4600)
    sr = kwargs.get("sample_rate",100)
    jump = kwargs.get("jump",0.1)
    mask_th = kwargs.get("mask_th",2.0)

    all_bkps = []
    # convert jump to number of samples
    jump_samples = distToSamples(jump,fr,rpm,sr)
    print("jump (samples)",jump_samples)
    ## find min distance between points in terms of samples
    # get function for determining the best width
    select_min_dist = kwargs.get("select_min_dist",min)
    min_sz_samples = select_min_dist(shapelet.convertSegToSamples(fr,rpm,sr)[0])
    print("min size (samples)",min_sz_samples)
    # get type of BPS method
    bps_method = kwargs.get("bps_method",rpt.Dynp)
    bps = bps_method(kwargs.get("model","l2"),min_sz_samples,jump_samples)
    # load torque file
    fname = os.path.splitext(os.path.basename(torque_path))[0]
    # load data
    print("loading file")
    # try loading as a setitec file
    is_setitec = True
    df = loadSetitecXls(torque_path)[-1]
    # if none of the columns contain the phrase Position then it's a Lueberring file
    # Lueberring has no header
    if not any(['Position' in c for c in df.columns]):
        is_setitec = False
        df = pd.read_excel(torque_path)
        unfiltered = df.torque.values
        dist = df.distance.values
    else:
        unfiltered = df['I Torque (A)'].values
        dist = np.abs(df['Position (mm)'].values)

    unf_min = unfiltered.min()
    unf_max = unfiltered.max()
    # filter the data
    print("butter filter")
    filtered = filterButter(unfiltered,kwargs.get("fc",0.1),kwargs.get("order",10))
    # upsample if the flag is set to True and the new sampling rate does not equal the source
    if kwargs.get("upsample",True) and (sr != kwargs.get("new_sr",100)):
        print("upsampling")
        # raise sampling rate
        filtered = raise_sr(filtered,sr,new_sr=kwargs.get("new_sr",100),smoothing=0.5)
        dist = raise_sr(dist,sr,new_sr=kwargs.get("new_sr",100),smoothing=0.5)
    ## fit and predict breakpoints
    # use numerical difference
    if kwargs.get("use_diff",True):
        print("finding BPS using difference")
        my_bkps = np.array(bps.fit_predict(np.diff(filtered),n_bkps=kwargs.get("bkps",5)))-1
    else:
        print("finding BPS using signal")
        my_bkps = np.array(bps.fit_predict(filtered,n_bkps=kwargs.get("bkps",5)))-1

    ## plot the results
    f,ax = plt.subplots()
    # plot the original signal
    ax.plot(dist,filtered,'b-',label="Filtered Signal")
    if kwargs.get("plot_grad",False):
        tax = ax.twinx()
        tax.plot(dist[:-1],np.diff(filtered),'r-')
        tax.set(ylabel="Gradient")
    # plot the breakpoints
    ax.vlines(dist[my_bkps],unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")
    all_bkps += my_bkps.tolist()
    ## find CFRP breakpoints
    # only done if a Setitec file
    # Lueberring files don't seem to have a CFRP entry step
    if is_setitec:
        cfrp = filtered[0:my_bkps[0]]
        cfrp_dist = dist[0:my_bkps[0]]
        # fit and predict breakpoints
        bps = bps_method(kwargs.get("model","l2"),min_sz_samples,jump_samples)
        if kwargs.get("use_diff",True):
            print("finding CFRP BPS using difference")
            cfrp_diff = np.diff(cfrp)
            # find smallest peak
            ii = np.argmin(cfrp_diff)
            # zero gradient within window around peak
            dii = cfrp_dist[ii]
            # if this value is more than 7.5mm in the signal
            # then it's prob what we're looking for
            if dii>7.5:
                print("removing divit")
                cfrp_diff[((cfrp_dist>=(dii-mask_th)) & (cfrp_dist<=(dii+mask_th)))[:-1]]=0
            my_bkps = np.array(bps.fit_predict(cfrp_diff,n_bkps=kwargs.get("cfrp_nbkps",2)))-1
        else:
            print("finding CFRP BPS using signal")
            my_bkps = np.array(bps.fit_predict(cfrp,n_bkps=kwargs.get("cfrp_nbkps",2)))-1

        all_bkps += my_bkps.tolist()
        # final plotting
        ax.vlines(dist[my_bkps],unf_min,unf_max,colors='g',linestyles='dotted',label="CFRP Breakpoints")
    else:
        print("Skipping finding CFRP breakpoints as file is Lueberring!")
    ax.legend()
    ax.set(xlabel="Distance (mm*100)",ylabel="Torque (mA)",title=f"{fname} min size {min_sz_samples}, jump {jump_samples}\nSR {sr}hz, RPM {rpm}, FR {fr}mm/rev")
    if kwargs.get("return_data",False):
        return f,dist[all_bkps].tolist()
    elif kwargs.get("return_samples",False):
        return f,all_bkps
    else:
        return f

def findTemplateBreakpoints(torque_path:str,shapelet:ToolShapelet,**kwargs)->pd.DataFrame:
    '''
        Finds the breakpoints in the signal and assembles a dataframe of features about the data between the breakpoints

        It calculates the min_size and jump using tool geometry and process parameters.

        The ToolShapelet instance is used to convert the segments from millimetres to number of samples.

        The min distance between breakpoints is calculated by processing the segments into a single value.
        By default the smallest segment is selected but the user can provide a different method via select_min_dist.
        
        The spacing between breakpoints on the grid is set by the parameter jump. The spacing is converted from
        distance in mm to number of samples using the paramters feed_rate, rpm and sr. The user controls this
        parameter kind of like how much wiggle room they want for possible breakpoints.

        Inputs:
            torque_path : Path to torque file
            shapelet : Instance of ToolShapelet describing the tool used in the signal
            feed_rate : Advance rate used in mm/rev. Default 0.04
            rpm : Angular velocity for file in rpm. Default 4500 rpm.
            sr : Sampling rate of the file in Hz. Default 20.
            jump : Spacing between the breakpoints to be evaluated in mm. Default 0.1 mm
            select_min_dist : Function for selecting the min size from the tool segments. Default min.
            model : Cost model or kernel model. See supported models for set bps_method. Default l2.
            bps_method : Breakpoint method. Default Dynp.
            fc : Nyquist cutoff frequency for Butterworth filter. Default 0.2.
            order : Model order of the Butterworth filter. Default 8.
            upsample : Upsample the signal to a new sampling rate. Default True.
            new_sr : New sampling rate to upsample to. Default 100 Hz.
            mask_th : Masking window size to remove backtracking in mm. Default 2.0

        Returns pandas dataframe
    '''
    import ruptures as rpt
    from dataparser import loadSetitecXls
    from scipy.signal import find_peaks, peak_widths
    from numpy.polynomial import polynomial as P
    from sklearn.preprocessing import MinMaxScaler
    ## get parameters
    fr = kwargs.get("feed_rate",0.04)
    rpm = kwargs.get("rpm",4600)
    sr = kwargs.get("sample_rate",20)
    jump = kwargs.get("jump",0.1)
    mask_th = kwargs.get("mask_th",2.0)
    new_sr = kwargs.get("new_sr",100)
    nbkps = kwargs.get("bkps",5)
    nbkps_cfrp = kwargs.get("cfrp_nbkps",2)

    all_bkps = []
    bkps_labels = []
    # convert jump to number of samples
    jump_samples = distToSamples(jump,fr,rpm,sr)
    ## find min distance between points in terms of samples
    # get function for determining the best width
    select_min_dist = kwargs.get("select_min_dist",min)
    min_sz_samples = select_min_dist(shapelet.convertSegToSamples(fr,rpm,sr)[0])
    # get type of BPS method
    bps_method = kwargs.get("bps_method",rpt.Dynp)
    bps = bps_method(kwargs.get("model","l2"),min_sz_samples,jump_samples)
    # load torque file
    fname = os.path.splitext(os.path.basename(torque_path))[0]
    # load data
    print("loading file")
    # try loading as a setitec file
    is_setitec = True
    df = loadSetitecXls(torque_path)[-1]
    # if none of the columns contain the phrase Position then it's a Lueberring file
    # Lueberring has no header
    if not any(['Position' in c for c in df.columns]):
        is_setitec = False
        df = pd.read_excel(torque_path)
        unfiltered = df.torque.values
        dist = df.distance.values
    else:
        unfiltered = df['I Torque (A)'].values
        dist = np.abs(df['Position (mm)'].values)

    print(f"Identified file as {'Setitec' if is_setitec else 'Lueberring'}")
    
    unf_min = unfiltered.min()
    unf_max = unfiltered.max()
    # filter the data
    print("butter filter")
    filtered = filterButter(unfiltered,kwargs.get("fc",0.1),kwargs.get("order",10))
    
    # upsample if the flag is set to True and the new sampling rate does not equal the source
    if kwargs.get("upsample",True) and (sr != new_sr):
        print("upsampling")
        # raise sampling rate
        filtered = raise_sr(filtered,sr,new_sr=new_sr,smoothing=0.5)
        dist = raise_sr(dist,sr,new_sr=new_sr,smoothing=0.5)
    ## fit and predict breakpoints
    # use numerical difference
    if kwargs.get("use_diff",True):
        print("finding BPS using difference")
        my_bkps = np.array(bps.fit_predict(np.diff(filtered),n_bkps=nbkps))-1
    else:
        print("finding BPS using signal")
        my_bkps = np.array(bps.fit_predict(filtered,n_bkps=nbkps))-1

    all_bkps += my_bkps.tolist()
    bkps_labels += len(my_bkps)*['Mat2',]
    ## find CFRP breakpoints
    # only done if a Setitec file
    # Lueberring files don't seem to have a CFRP entry step
    if is_setitec:
        cfrp = filtered[0:my_bkps[0]]
        cfrp_dist = dist[0:my_bkps[0]]
        # fit and predict breakpoints
        bps = bps_method(kwargs.get("model","l2"),min_sz_samples,jump_samples)
        if kwargs.get("use_diff",True):
            print("finding CFRP BPS using difference")
            cfrp_diff = np.diff(cfrp)
            # find smallest peak
            ii = np.argmin(cfrp_diff)
            # zero gradient within window around peak
            dii = cfrp_dist[ii]
            # if this value is more than 7.5mm in the signal
            # then it's prob what we're looking for
            if dii>7.5:
                print("removing divit")
                cfrp_diff[((cfrp_dist>=(dii-mask_th)) & (cfrp_dist<=(dii+mask_th)))[:-1]]=0
            my_bkps = np.array(bps.fit_predict(cfrp_diff,n_bkps=nbkps_cfrp))-1
        else:
            print("finding CFRP BPS using signal")
            my_bkps = np.array(bps.fit_predict(cfrp,n_bkps=nbkps_cfrp))-1

        all_bkps += my_bkps.tolist()
    else:
        print("Skipping finding CFRP breakpoints as file is Lueberring!")
    bkps_labels += len(my_bkps)*['CFRP',]
##    print("unfiltered",unfiltered.shape,unfiltered.dtype)
##    print("filtered",filtered.shape,filtered.dtype)
##    print("dist",dist.shape,dist.dtype)

    all_bkps = np.array(all_bkps)
    bkps_labels = np.array(bkps_labels)
    ii = np.argsort(all_bkps)
    all_bkps = all_bkps[ii]
    bkps_labels = bkps_labels[ii]

    print(f"collected {len(all_bkps)} bkps total")
    print(all_bkps)
    print(bkps_labels)
    print("collecting features")
    df_all_bkps = []
    
    # iterate over breakpoints as pairs
    for a,b,la,lb in zip(all_bkps[::2],all_bkps[1::2],bkps_labels[::2],bkps_labels[1::2]):
        dist_clip = dist[a:b]
        uf_clip = unfiltered[a:b]
        f_clip = filtered[a:b]

        if dist_clip.shape[0]==0:
            dist_clip = np.array(dist[a],dist[b])

        if uf_clip.shape[0]==0:
            uf_clip = np.array(unfiltered[a],unfiltered[b])

        if f_clip.shape[0]==0:
            f_clip = np.array(filtered[a],filtered[b])
        # set up basic attributes
        row = [a,b,abs(a-b),abs(dist[a]-dist[b]),dist_clip[0],dist_clip[-1]]
        ## get unfiltered torque features
        if uf_clip.shape[0]>0:
            uf_feats = [abs(uf_clip.max()-uf_clip.min()),uf_clip.min(),uf_clip.max(),np.mean(uf_clip),np.std(uf_clip),uf_clip.max()/unfiltered.max()]
        else:
            print(f"Warning! Setting unfiltered feats to -1 as section {a} -> {b} has size 0!")
            uf_feats = [-1,-1,-1,-1,-1]
        ## get filtered torque features
        if f_clip.shape[0]>0:
            f_feats = [abs(f_clip.max()-f_clip.min()),f_clip.min(),f_clip.max(),np.mean(f_clip),np.std(f_clip),f_clip.max()/filtered.max()]
        else:
            print(f"Warning! Setting filtered feats to -1 as section {a} -> {b} has size 0!")
            f_feats = [-1,-1,-1,-1,-1]
        ## get distance features
        # get gradient from 1d polt fitted to data
        if uf_clip.shape[0]>0 and f_clip.shape[0]>0 and dist_clip.shape[0]>0:
            filt_grad = P.polyfit(dist_clip,f_clip,1,full=False)[-1]
            unfilt_grad = P.polyfit(dist_clip,uf_clip,1,full=False)[-1]
            # norm data
            dist_norm = MinMaxScaler().fit_transform(dist_clip.reshape(-1, 1)).flatten()
            unfilt_torque_norm = MinMaxScaler().fit_transform(uf_clip.reshape(-1, 1)).flatten()
            filt_torque_norm = MinMaxScaler().fit_transform(f_clip.reshape(-1, 1)).flatten()
            # find grad for norm data
            filt_grad_nom = P.polyfit(dist_norm,filt_torque_norm,1,full=False)[-1]
            unfilt_grad_norm = P.polyfit(dist_norm,unfilt_torque_norm,1,full=False)[-1]
            # make feature vector
            grad_feats = [unfilt_grad,filt_grad,unfilt_grad_norm,filt_grad_nom]
        else:
            print(f"Warning! Setting gradient feats to -1 as section {a} -> {b} has size 0!")
            grad_feats = [-1,-1,-1,-1]
        # form final row
        row = row + uf_feats + f_feats + grad_feats + [f"{la} to {lb}",]
        # add to list of data
        df_all_bkps.append(row)
    df = pd.DataFrame(df_all_bkps,columns = ["A","B","Range (samples)","Range (mm)","Dist Min (mm)","Dist Max (mm)",
                                             "Unfiltered Torque Range (mA)","Unfiltered Min Torque (mA)","Unfiltered Max Torque (mA)","Unfiltered Average Torque (mA)","Unfiltered Torque Std (mA)","Unfiltered Normalized Max Torque",
                                             "Filtered Torque Range (mA)","Filtered Min Torque (mA)","Filtered Max Torque (mA)","Filtered Average Torque (mA)","Filtered Torque Std (mA)","Filtered Normalized Max Torque",
                                             "Unfiltered Fitted Gradient","Filtered Fitted Gradient","Normalized Unfiltered Fitted Gradient","Normalized Filtered Fitted Gradient","Material Label"])
    return df

def plotTemplateBreakpoints(torque_path:str,opath:str,ts:ToolShapelet=None,**kwargs):
    '''
        Iterates over all files in the given path, finds and plots the breakpoints saving the generated figure in the specified folder

        Inputs:
            torque_path : Wildcard path to folder of torque files.
            opath : Folder to store the results in
            ts : ToolShapelet used. If None, a dummy one is used. Default None
            **kwargs : See findPlotBpsCalcMinSize
    '''
    if ts is None:
        ts = ToolShapelet([1,2,3],[1,2,3])

    for fn in glob(torque_path,recursive=True):
        f = findPlotBpsCalcMinSize(fn,ts,**kwargs)
        f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-template-extraction-breakpoints.png"))
        plt.close(f)

def collectBpsFeats(path:str,ts=None,**kwargs)->pd.DataFrame:
    '''
        Iterates over all files in the given path, finds and extracts features about the breakpoints storing them in a single dataframe

        A new column called Filename is added storing the full file path so the breakpoints can be traced back

        Inputs:
            torque_path : Wildcard path to folder of torque files.
            opath : Folder to store the results in
            ts : ToolShapelet used. If None, a dummy one is used. Default None
            skip_last_pair : Drop the last two breakpoints in each dataframe. Default False.l
            **kwargs : See findTemplateBreakpoints

        Returns pandas dataframe
    '''
    if ts is None:
        ts = ToolShapelet([1,2,3],[1,2,3])

    dfs = []
    for fn in glob(path,recursive=True):
        print(fn)
        df = findTemplateBreakpoints(fn,ts,**kwargs)
        if kwargs.get("skip_last_pair",False):
            df.drop(df.tail(2).index,inplace=True)
        df['Filename'] = fn
        dfs.append(df)
    all_feats = pd.concat(dfs)
    all_feats.reset_index(inplace=True)
    return all_feats

def plotBpsFeats(all_feats:pd.DataFrame,**kwargs):
    '''
        Plot the features collected about the breakpoints

        Creates a pairplot of all features using sns.pairplot

        Creates scatter plots for each column plotting against index

        The Material Label column is used to colour code the data

        Inputs:
            all_feats: See collectBpsFeats
            opath : Output folder for plots. Default None.
            palette : Seaborn palette used for plotting. Default bright.
    '''
    import seaborn as sns
    plot_feats = all_feats.drop(columns=['A','B'],inplace=False)
    opath = kwargs.get("opath",None)
    sns.pairplot(plot_feats,hue='Material Label',palette=kwargs.get('palette','bright'),corner=True,diag_kind='kde')
    if opath:
        plt.gcf().savefig(os.path.join(opath,"bps-all-feats-pairplot.png"))
        plt.close('all')

    # get columns of interest
    for c in plot_feats.columns:
        if (c != 'Filename') and (c != 'index') and (c != 'Material Label'):
            f,ax = plt.subplots(constrained_layout=True)
            sns.scatterplot(plot_feats,x=all_feats.index,y=c,hue='Material Label',palette=kwargs.get('palette','bright'))
            f.suptitle(c)
            if opath:
                f.savefig(os.path.join(opath,f"bps-feats-{c}-scatter.png"))
                plt.close('all')

def testShapeletAngleRange(path:str,diameter:float,angle:np.ndarray,**kwargs):
    '''
        Create a series of tool shapelets using the range of angles and search for breakpoints.

        A figure is created with the angle along the x-axis and the found breakpoints on the y-axis.
        The purpose is to observe the effect of tool geometry on the breakpoint locations as it's used
        to set the min distance parameter.

        If an output path is specifed, then all figures from findPlotBpsCalcMinSize are saved at that location
        as well as the final scatter plot. If no output is given, then no files are saved and all are kept loaded
        and the final scatter plot is returned.

        Inputs:
            path : Path to single XLS file
            diameter : Tool diameter used for all shapelets in mm.
            angle : Array of angle values to test.
            opath : Output path to save all generated figures.

        Returns final scatter plot if opath is None else returns None.
    '''
    opath = kwargs.get("opath",None)
    all_bps = []
    angle_plot = []
    for theta in angle:
        # make the tool shapelet using the current angle
        ts = ToolShapelet([theta,],[diameter,],mode="angle_diameter")
        # plot the bps
        f,bps = findPlotBpsCalcMinSize(path,ts,**kwargs)
        print(bps)
        all_bps += bps
        angle_plot += len(bps)*[theta,]
        if opath:
            f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(path))[0]}-template-bps-angle-{theta}.png"))
            plt.close(f)
    f,ax = plt.subplots()
    ax.scatter(angle_plot,all_bps)
    ax.set(xlabel="Half Tool Angle (degrees)",ylabel="Breakpoint Location (mm)",title=f"{os.path.splitext(os.path.basename(path))[0]} BPS vs Angle")
    if not opath:
        return f
    f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(path))[0]}-angle-effect.png"))
    plt.close(f)

def findPlotBpsSetitecDynamic(torque_path:str,shapelet:ToolShapelet,**kwargs):
    '''
        Finds and plots the breakpoints in the signal by calculating the min_size and jump using tool
        geometry and process parameters.

        The ToolShapelet instance is used to convert the segments from millimetres to number of samples.

        The min distance between breakpoints is calculated by processing the segments into a single value.
        By default the smallest segment is selected but the user can provide a different method via select_min_dist.
        
        The spacing between breakpoints on the grid is set by the parameter jump. The spacing is converted from
        distance in mm to number of samples using the paramters feed_rate, rpm and sr. The user controls this
        parameter kind of like how much wiggle room they want for possible breakpoints.

        Inputs:
            torque_path : Path to torque file
            shapelet : Instance of ToolShapelet describing the tool used in the signal
            feed_rate : Advance rate used in mm/rev. Default 0.04
            rpm : Angular velocity for file in rpm. Default 4500 rpm.
            sr : Sampling rate of the file in Hz. Default 100.
            jump : Spacing between the breakpoints to be evaluated in mm. Default 0.1 mm
            select_min_dist : Function for selecting the min size from the tool segments. Default min.
            model : Cost model or kernel model. See supported models for set bps_method. Default l2.
            bps_method : Breakpoint method. Default Dynp.
            fc : Nyquist cutoff frequency for Butterworth filter. Default 0.2.
            order : Model order of the Butterworth filter. Default 8.
            upsample : Upsample the signal to a new sampling rate. Default True.
            new_sr : New sampling rate to upsample to. Default 100 Hz.
            mask_th : Masking window size to remove backtracking
            plot_grad : Plot the gradient. Default True.

        Returns figure
    '''
    import ruptures as rpt
    from dataparser import loadSetitecXls
    from scipy.signal import find_peaks, peak_widths
    ## get parameters
    fr = kwargs.get("feed_rate",0.04)
    rpm = kwargs.get("rpm",4600)
    jump = kwargs.get("jump",0.1)
    mask_th = kwargs.get("mask_th",2.0)

    all_bkps = []
    # load torque file
    fname = os.path.splitext(os.path.basename(torque_path))[0]
    # load data
    print("loading file")
    # try loading as a setitec file
    is_setitec = True
    df = loadSetitecXls(torque_path)
    df_data = df[-1]
    # get sampling rate from file
    sr = float(df[0]['Sample Rate (Hz)'][0])
    # if none of the columns contain the phrase Position then it's a Lueberring file
    # Lueberring has no header
    if not any(['Position' in c for c in df_data.columns]):
        raise ValueError("File is not Setitec! Method is designed to use the metadata from the file")
    else:
        unfiltered = df['I Torque (A)'].values
        dist = np.abs(df['Position (mm)'].values)

    # filter the programs to the ones actually used
    metadata = pd.DataFrame.from_dict(df[7])
    metadata = metadata[metadata['Step Nb'].isin(df_data['Step (nb)'].unique())]
    # RPM is already in rots per min and Feed Speed in mm/s
    # convert jump to number of samples
##    jump_samples = distToSamples(jump,fr,rpm,sr)
##    print("jump (samples)",jump_samples)
##    ## find min distance between points in terms of samples
##    # get function for determining the best width
##    select_min_dist = kwargs.get("select_min_dist",min)
##    min_sz_samples = select_min_dist(shapelet.convertSegToSamples(fr,rpm,sr)[0])
##    print("min size (samples)",min_sz_samples)
##    # get type of BPS method
##    bps_method = kwargs.get("bps_method",rpt.Dynp)
##    bps = bps_method(kwargs.get("model","l2"),min_sz_samples,jump_samples)

    unf_min = unfiltered.min()
    unf_max = unfiltered.max()
    # filter the data
    print("butter filter")
    filtered = filterButter(unfiltered,kwargs.get("fc",0.1),kwargs.get("order",10))
    # upsample if the flag is set to True and the new sampling rate does not equal the source
    if kwargs.get("upsample",True) and (sr != kwargs.get("new_sr",100)):
        print("upsampling")
        # raise sampling rate
        filtered = raise_sr(filtered,sr,new_sr=kwargs.get("new_sr",100),smoothing=0.5)
        dist = raise_sr(dist,sr,new_sr=kwargs.get("new_sr",100),smoothing=0.5)
    ## fit and predict breakpoints
    # use numerical difference
    if kwargs.get("use_diff",True):
        print("finding BPS using difference")
        my_bkps = np.array(bps.fit_predict(np.diff(filtered),n_bkps=kwargs.get("bkps",5)))-1
    else:
        print("finding BPS using signal")
        my_bkps = np.array(bps.fit_predict(filtered,n_bkps=kwargs.get("bkps",5)))-1

    ## plot the results
    f,ax = plt.subplots()
    # plot the original signal
    ax.plot(dist,filtered,'b-',label="Filtered Signal")
    if kwargs.get("plot_grad",False):
        tax = ax.twinx()
        tax.plot(dist[:-1],np.diff(filtered),'r-')
        tax.set(ylabel="Gradient")
    # plot the breakpoints
    ax.vlines(dist[my_bkps],unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")
    all_bkps += my_bkps.tolist()
    ## find CFRP breakpoints
    # only done if a Setitec file
    # Lueberring files don't seem to have a CFRP entry step
    if is_setitec:
        cfrp = filtered[0:my_bkps[0]]
        cfrp_dist = dist[0:my_bkps[0]]
        # fit and predict breakpoints
        bps = bps_method(kwargs.get("model","l2"),min_sz_samples,jump_samples)
        if kwargs.get("use_diff",True):
            print("finding CFRP BPS using difference")
            cfrp_diff = np.diff(cfrp)
            # find smallest peak
            ii = np.argmin(cfrp_diff)
            # zero gradient within window around peak
            dii = cfrp_dist[ii]
            # if this value is more than 7.5mm in the signal
            # then it's prob what we're looking for
            if dii>7.5:
                print("removing divit")
                cfrp_diff[((cfrp_dist>=(dii-mask_th)) & (cfrp_dist<=(dii+mask_th)))[:-1]]=0
            my_bkps = np.array(bps.fit_predict(cfrp_diff,n_bkps=kwargs.get("cfrp_nbkps",2)))-1
        else:
            print("finding CFRP BPS using signal")
            my_bkps = np.array(bps.fit_predict(cfrp,n_bkps=kwargs.get("cfrp_nbkps",2)))-1

        all_bkps += my_bkps.tolist()
        # final plotting
        ax.vlines(dist[my_bkps],unf_min,unf_max,colors='g',linestyles='dotted',label="CFRP Breakpoints")
    else:
        print("Skipping finding CFRP breakpoints as file is Lueberring!")
    ax.legend()
    ax.set(xlabel="Distance (mm*100)",ylabel="Torque (mA)",title=f"{fname} min size {min_sz_samples}, jump {jump_samples}\nSR {sr}hz, RPM {rpm}, FR {fr}mm/rev")
    if kwargs.get("return_data",True):
        return f,all_bkps
    else:
        return f

## from https://github.com/jthiem/overlapadd/blob/master/olafilt.py
##L_I = b.shape[0]
### Find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
##L_F = 2<<(L_I-1).bit_length()

if __name__ == "__main__":
    angle = np.arange(45,85,1)/2
    #torque_path = "922123_100r1_5245.xlsx"
    torque_path = r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/922123_100r1_5245.xlsx"
    #### load power signal and extract target period ####
    # read the cvs file
    #path_L_new = 'LOG0017.CSV'
    path_L_new = r"luebering_data_test/4B Luebbering/New cutter/Power analyser log files/LOG0017.CSV"

    for fn in glob(r"C:\Users\david\Downloads\MSN660-20230912T134035Z-001\MSN660\*.xls"):
        testShapeletAngleRange(fn, # using single file for testing
                               2.0, # dummy diameter of the tool
                               np.arange(45,85,1)/2, # range of angle to test. div by 2 as it's from full angle to angle from centre line
                               is_rads=False,   # ensuring that the angles are treated as degrees
                               opath="angle-test", # output path for plots
                               sample_rate=100.0, # sampling rate of the signal
                               return_data=True,
                               return_samples=False)
    
##    #plotTemplateBreakpoints(r"C:\Users\uos\Downloads\MSN660-20230912T134035Z-001\MSN660\*.xls","bps-feats",sample_rate=100.0,plot_grad=True)
##    all_feats = collectBpsFeats(r"C:\Users\uos\Downloads\MSN660-20230912T134035Z-001\MSN660\*.xls",sample_rate=100.0)
##    plotBpsFeats(all_feats,opath="bps-feats")
##        except:
##            print("Skipping ",fn)
##            plt.close('all')
##    peltBPSSlider(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/922123_100r1_5245.xlsx")
##    df = pd.read_excel(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data\922123_100r1_5264.xlsx")
##    gps = splitByDistance(df)
##    max_edge_change = getTorqueEdgeChanges(gps)
##    max_change = getMaxTorqueChange(gps)
##    periods = np.arange(df.distance.min(),df.distance.max()+100,100)
##    f,ax = plt.subplots()
##    tax = ax.twinx()
##    ax.plot(periods[:-1],max_edge_change,'r-',label='Edge Change')
##    ax.plot(periods[:-1],max_change,'g-',label='Max Change')
##    ax.legend()
##    tax.plot(df.distance,df.torque,'b-')
##    plt.show()
    #optimal_window_tool(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx",r"luebering_data_test/4B Luebbering/Worn cutter/Power analyser log files/LOG0018.CSV",False)
    #plotWinSizeEnergy(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx",r"luebering_data_test/4B Luebbering/New cutter/Power analyser log files/LOG0017.CSV",True)

##    plotWinSizeEnergyBoth(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx",r"luebering_data_test/4B Luebbering/New cutter/Power analyser log files/LOG0017.CSV",
##                          r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx",r"luebering_data_test/4B Luebbering/Worn cutter/Power analyser log files/LOG0018.CSV",apply_tukey=False)

##    denoiseTorqueWavelet(glob(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx")+glob(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx"),
##                        wavelet=getRecDiscreteWavelets(15,inv=False),output_path="lueberring wavelet")
##
##    plt.close('all')

##    stats = denoiseTorqueWaveletSkimage(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx",
##                                        r"luebering_data_test/4B Luebbering/Worn cutter/Power analyser log files/LOG0018.CSV",
##                                        is_new=False,
##                                         wavelet=getRecDiscreteWavelets(15,inv=False),output_path="lueberring wavelet")
##    plt.close('all')

##    denoiseConvolutionSame(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx")
##
##    stats_wins = []
##    #[signal.windows.hann,signal.windows.blackman,signal.windows.flattop,pulsingWindow,chebWinAt,kaiserWinAt,signal.windows.hamming,taylorWinAt]
##    for win_fn in [signal.windows.hann,signal.windows.blackman,signal.windows.flattop,pulsingWindow,chebWinAt,kaiserWinAt,signal.windows.hamming,taylorWinAt]:
##        stats,best_wsize,best_scale = denoiseConvAgainstPowerScale(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx",r"luebering_data_test/4B Luebbering/New cutter/Power analyser log files/LOG0017.CSV",
##                              r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx",r"luebering_data_test/4B Luebbering/Worn cutter/Power analyser log files/LOG0018.CSV",
##                                window_fn=win_fn,output_path="lueberring convolution")
##        plt.close('all')
##        stats_wins.append(stats)
##    stats_wins_stack = pd.concat(stats_wins)

##    torqueCWT(glob(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx")+glob(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx"),
##              wavelet=getCWAtBC(2,[0.5,1,2],4),
##              # target frequencies for the scales
##              #scales=12)
##              target_freqs=np.arange(0.05,1.0,0.01))

    #plot_fft(glob(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx")[0])
##    for fn in glob(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx") + glob(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx"):
##        f = plot_fft_diff(fn)
##        f.savefig(f"{os.path.splitext(os.path.basename(fn))[0]}-rfft-difference.png")
##        plt.close(f)
    
##    plotWindowFreqResponse([signal.windows.hann,signal.windows.blackman,signal.windows.flattop,pulsingWindow,chebWinAt,kaiserWinAt,signal.windows.hamming,taylorWinAt])
##    plotWindowFreqResponse([chebWinAt,kaiserWinAt,signal.windows.hamming,taylorWinAt])

    ##dynamicProgBPS(glob(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx")+glob(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx"))

    #for o in range(2,9):
    #    applyFiltFilt(glob(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx")+glob(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx"),fstep=0.1,order=o)
    #plotEachSNR(glob(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx")+glob(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx"))
