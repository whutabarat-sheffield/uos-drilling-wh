import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import seaborn as sns
from abyss.rolling_gradient import rolling_gradient_v2
from scipy.fft import rfft, rfftfreq, fft, fftfreq
from glob import glob
import os
from scipy import signal

#https://stackoverflow.com/a/8260297
def window_rms(a, window_size):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))

def window_rms_pd(a, window_size):
    return pd.Series(a).pow(2).rolling(window_size).mean().apply(np.sqrt,raw=True)

# apply rms smoothing to the target file using the specified window
def rms_smooth(path,ws=10, use_alt=True):
    df = pd.read_excel(path)
    rms= (window_rms_pd if use_alt else window_rms)(df.torque,ws)
    f,ax = plt.subplots()
    ax.plot(df.torque,'b-')
    ax.plot(rms,'r-')
    return f

# apply RMS smoothing to the signal and then find the rolling gradient
def rms_smooth_rg(path,ws=20,N=10, use_alt=True):
    df = pd.read_excel(path)
    # perform rms smoothing
    rms= (window_rms_pd if use_alt else window_rms)(df.torque,ws)
    f,ax = plt.subplots(ncols=2)
    # plot the torque against rms smoothed
    ax[0].plot(df.torque,'b-',label="Original")
    ax[0].plot(rms,'r-',label="RMS")
    ax[0].set(xlabel="Index",ylabel="Torque",title="RMS")
    ax[0].legend()
    # plot the rms
    ax[1].plot(rms,'b-')
    if isinstance(N,int):
        N = [N,]
    tax = ax[1].twinx()
    # plot the rolling gradient for each target window size
    for n in N:
        tax.plot(rolling_gradient_v2(rms,n),label=f"N={n}")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}\nRMS + Rolling Gradient")
    return f

# calculate the rolling gradient of the signal
# reverse flag is for doing the rolling gradient in reverse
def try_rg(path,key='fil_torque',N=10,reverse=False):
    from rolling_gradient import rolling_gradient_v2
    if isinstance(N,int):
        N = [N,]

    df = pd.read_excel(path)
    f,ax = plt.subplots()
    ax.plot(df.distance,df[key],'b-')
    tax = ax.twinx()
    for n in N:
        if reverse:
            tax.plot(df.distance,rolling_gradient_v2(df.fil_torque.values[::-1],n),label=f"N={n}")
        else:
            tax.plot(df.distance,rolling_gradient_v2(df.fil_torque.values,n),label=f"N={n}")
    tax.legend()
    ax.set(xlabel="Distance",ylabel=key)
    tax.set_ylabel("Rolling Gradient")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}\nRolling Gradient {key} N={N}")
    return f

# try finding breakpoints using dynamic programming at the specified settings
def try_dynp(path,k,key='fil_torque',model="l2", min_size=10,jump=5):
    df = pd.read_excel(path)
    data = df[key].values
    algo = rpt.Dynp(model=model,min_size=min_size,jump=jump).fit(data)
    result = algo.predict(k)
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    f,ax = plt.subplots()
    ax.plot(df.distance,data,'b-')
    ax.plot(df.distance[result],data[result],'rx',markersize=22)
    ax.set(xlabel="Distance",ylabel=key)
    f.suptitle(f'{os.path.splitext(os.path.basename(path))[0]}\nDynp BPS model={model},min_size={min_size},jump={jump}')
    return f

# try finding breakpoints using PELT at the specified settings
def try_pelt(path,pen=1,key='fil_torque',model="l2",**kwargs):
    df = pd.read_excel(path)
    data = df[key].values
    algo = rpt.Pelt(model=model,min_size=kwargs.get("min_size",10),jump=kwargs.get("jump",5)).fit(data)
    try:
        result = algo.predict(pen)
    except ValueError:
        print(f'Failed to find PELT breakpoints using model {model}, min_size {kwargs.get("min_size",10)}, size {kwargs.get("jump",5)}')
        return
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    f,ax = plt.subplots()
    ax.plot(df.distance,data,'b-')
    ax.plot(df.distance[result],data[result],'rx',markersize=22)
    ax.set(xlabel="Distance",ylabel=key)
    f.suptitle(f'{os.path.splitext(os.path.basename(path))[0]}\nPelt BPS min_size={kwargs.get("min_size",50)},jump={kwargs.get("jump",5)}')
    return f

def kernel_seg(path,k,key='fil_torque',model='linear',min_size=10,jump=5):
    df = pd.read_excel(path)
    data = df[key].values
    algo_c = rpt.KernelCPD(kernel=model, min_size=min_size,jump=jump).fit(data)
    result = algo_c.predict(k)
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    f,ax = plt.subplots()
    ax.plot(df.distance,data,'b-')
    ax.plot(df.distance[result],data[result],'rx',markersize=22)
    ax.set(xlabel="Distance",ylabel=key)
    f.suptitle(f'{os.path.splitext(os.path.basename(path))[0]}\nKernel BPS model={model},min_size={min_size},jump={jump}')
    return f

def binary_segmentation(path,k,key='fil_torque',model="linear",min_size=10,jump=5):
    df = pd.read_excel(path)
    data = df[key].values

    algo_c = rpt.Binseg(model,min_size=min_size,jump=jump).fit(data)
    result = algo_c.predict(k)
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    f,ax = plt.subplots()
    ax.plot(df.distance,data,'b-')
    ax.plot(df.distance[result],data[result],'rx',markersize=22)
    ax.set(xlabel="Distance",ylabel=key)
    f.suptitle(f'{os.path.splitext(os.path.basename(path))[0]}\nBinSeg BPS model={model},min_size={min_size},jump={jump}')
    return f

def bottom_up_seg(path,k,key='fil_torque',model="linear",min_size=10,jump=5):
    df = pd.read_excel(path)
    data = df[key].values

    algo_c = rpt.BottomUp(model,min_size=min_size,jump=jump).fit(data)
    result = algo_c.predict(k)
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    f,ax = plt.subplots()
    ax.plot(df.distance,data,'b-')
    ax.plot(df.distance[result],data[result],'rx',markersize=22)
    ax.set(xlabel="Distance",ylabel=key)
    f.suptitle(f'{os.path.splitext(os.path.basename(path))[0]}\nBottomUp BPS model={model},min_size={min_size},jump={jump}')
    return f

def window_sliding_seg(path,k,winSize=5,key='fil_torque',model="linear",min_size=10,jump=5,msz=90):
    df = pd.read_excel(path)
    data = df[key].values
    algo_c = rpt.Window(width=winSize,model=model,min_size=min_size,jump=jump).fit(data)
    result = algo_c.predict(k)
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    f,ax = plt.subplots()
    ax.plot(df.distance,data,'b-')
    ax.plot(df.distance[result],data[result],'rx',markersize=msz)
    ax.set(xlabel="Distance",ylabel=key)
    f.suptitle(f'{os.path.splitext(os.path.basename(path))[0]}\nWindow Seg BPS model={model},min_size={min_size},jump={jump}')
    return f

def try_all_rupture(path,k,pen=1,model="l2",winSize=5,key='fil_torque',msz=70,**kwargs):
    df = pd.read_excel(path)
    data = df[key].values

    f,ax = plt.subplots()
    ax.plot(df.distance,data,'b-')

    algo = rpt.Dynp(model=model,min_size=kwargs.get("min_size",10),jump=kwargs.get("jump",5)).fit(data)
    result = algo.predict(k)
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    ax.scatter(df.distance[result],data[result],s=msz,label='dynp')

    algo = rpt.Pelt(model=model,min_size=kwargs.get("min_size",10),jump=kwargs.get("jump",5)).fit(data)
    try:
        result = algo.predict(pen)
    except ValueError:
        print(f'Failed to find PELT breakpoints using model {model}, min_size {kwargs.get("min_size",10)}, size {kwargs.get("jump",5)}')
        return
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    ax.scatter(df.distance[result],data[result],s=msz,label='pelt')

    algo_c = rpt.KernelCPD(kernel=model, min_size=kwargs.get("min_size",10),jump=kwargs.get("jump",5)).fit(data)
    result = algo_c.predict(k)
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    ax.scatter(df.distance[result],data[result],s=msz,label='kernel')

    algo_c = rpt.Binseg(model,min_size=kwargs.get("min_size",10),jump=kwargs.get("jump",5)).fit(data)
    result = algo_c.predict(k)
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    ax.scatter(df.distance[result],data[result],s=msz,label='binary')

    algo_c = rpt.BottomUp(model,min_size=kwargs.get("min_size",10),jump=kwargs.get("jump",5)).fit(data)
    result = algo_c.predict(k)
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    ax.scatter(df.distance[result],data[result],s=msz,label='bottomup')

    algo_c = rpt.Window(width=winSize,model=model,min_size=kwargs.get("min_size",10),jump=kwargs.get("jump",5)).fit(data)
    result = algo_c.predict(k)
    #result = np.array(result)-1
    result = np.array(list(map(lambda x : x-1 if x>(data.shape[0]-1) else x,result)))
    ax.scatter(df.distance[result],data[result],s=msz,label='window')

    ax.legend()
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]} Rupture Breakpoints model={model}")
    return f

# find the FFT of the unfiltered signal
def plot_fft(path):
    df = pd.read_excel(path)
    yf = rfft(df.torque)
    # sample rate from the time vector
    xf = rfftfreq(df.shape[0],50/1000)

    f,ax = plt.subplots(ncols=2)
    ax[0].plot(xf,np.abs(yf))
    ax[0].set_yscale('log')
    ax[0].set(xlabel="Frequency (Hz)",ylabel="Magnitude",title="Amplitude")

    ax[1].plot(xf,np.angle(yf))
    ax[1].set(xlabel="Frequency (Hz)",ylabel="Phase (rads)",title="Phase")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]} FFT")
    return f

# plot the freq response of a lowpass digital filter
def plot_lowpass_filter(order=4,cutoff=7.5):
    b,a = signal.butter(order,cutoff/(0.5*20),'low',analog=False)
    #b,a = signal.butter(order,cutoff,'low',analog=False,fs=20)
    w,h = signal.freqz(b,a)
    f,ax = plt.subplots()
    plt.semilogx(w,20*np.log10(np.abs(h)))
    ax.set(xlabel="Frequency (radians/sec)",ylabel="Amplitude (dB)")
    f.suptitle(f"Butterworth filter Low Pass Filter fc={cutoff}Hz")
    plt.grid(which='both',axis='both')
    plt.axvline(cutoff,color='green')
    #plt.axvline(cutoff,color='green')
    plt.show()

# plot the freq response of a highpass digital filter
def plot_highpass_filter(order=4,cutoff=7.5):
    b,a = signal.butter(order,cutoff/(0.5*20),'hp',analog=False)
    #b,a = signal.butter(order,cutoff,'low',analog=False,fs=20)
    w,h = signal.freqz(b,a)
    f,ax = plt.subplots()
    plt.semilogx(w,20*np.log10(np.abs(h)))
    ax.set(xlabel="Frequency (radians/sec)",ylabel="Amplitude (dB)")
    f.suptitle(f"Butterworth filter High Pass Filter fc={cutoff}Hz")
    plt.grid(which='both',axis='both')
    plt.axvline(cutoff,color='green')
    #plt.axvline(cutoff,color='green')
    plt.show()

# apply a lowpass digital filter to the unfiltered torque
def apply_lowpass_filter(path,order=4,cutoff=7.5):
    df = pd.read_excel(path)
    sos = signal.butter(order,cutoff/(0.5*20),'low',analog=False,output='sos')
    filtered = signal.sosfilt(sos, df.torque)
    f,ax = plt.subplots(nrows=2)
    ax[0].plot(df.distance,df.torque)
    ax[0].set(xlabel="Distance",ylabel="Torque",title="Unfiltered")
    ax[1].plot(df.distance,filtered)
    ax[1].set(xlabel="Distance",ylabel="Torque",title=f"Low Pass Filtered fc={cutoff}Hz")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}")
    f.tight_layout()
    return f

# apply a highpass digital filter to the unfiltered torque
def apply_highpass_filter(path,order=4,cutoff=7.5):
    df = pd.read_excel(path)
    sos = signal.butter(order,cutoff/(0.5*20),'hp',analog=False,output='sos')
    filtered = signal.sosfilt(sos, df.torque)
    f,ax = plt.subplots(nrows=2)
    ax[0].plot(df.distance,df.torque)
    ax[0].set(xlabel="Distance",ylabel="Torque",title="Unfiltered")
    ax[1].plot(df.distance,filtered)
    ax[1].set(xlabel="Distance",ylabel="Torque",title=f"High Pass Filtered fc={cutoff}Hz")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}")
    f.tight_layout()
    return f

def find_all_highpass_energy(paths,order=4,cutoff=9.0):
    if isinstance(paths,str):
        paths = glob(paths)
    energy = []
    for fn in paths:
        df = pd.read_excel(fn)
        sos = signal.butter(order,cutoff/(0.5*20),'hp',analog=False,output='sos')
        filtered = signal.sosfilt(sos, df.torque)
        energy.append(np.sum(filtered**2))
    f,ax = plt.subplots(constrained_layout=True)
    ax.scatter(range(len(energy)),energy,marker='x',s=50)
    ax.set(xlabel="Path Index",ylabel="Signal Energy")
    f.suptitle(f"Signal Energy of Tourque High Pass Filtered fc={cutoff}Hz")
    ax.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(p))[0] for p in paths])
    ax.tick_params(axis='x', rotation=90)
    return f

# plot the stft of the unfiltered torque
def plot_stft(path,log_scale=False,plot_signal=False):
    df = pd.read_excel(path)
    data = df.torque.values
    freq,t,Zxx = signal.stft(data,fs=20.0,nperseg=data.shape[0])
    f,ax = plt.subplots()
    if log_scale:
        ax.pcolormesh(t,freq,10*np.log10(np.abs(Zxx)),shading='gouraud',cmap='hot')
    else:
        ax.pcolormesh(t,freq,np.abs(Zxx),shading='gouraud',cmap='hot')
    if plot_signal:
        tax = ax.twinx()
        tax.plot(df.timestamp/1000.0,df.torque,'y-')
    ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}\nSTFT Magnitude")
    return f

# plot the spectrogram
def plot_spect(path):
    df = pd.read_excel(path)
    freq, times, spectrogram = signal.spectrogram(df.torque,20,nperseg=len(df.torque))
    f,ax = plt.subplots(constrained_layout=True)
    I = ax.imshow(spectrogram,aspect='auto',cmap='hot',origin='lower')
    plt.colorbar(I)
    ax.set(xlabel="Time Window (s)",ylabel="Frequency Band")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]} Spectrogram")
    return f

# plot welch
def plot_welch(path):
    df = pd.read_excel(path)
    freqs,psd = signal.welch(df.torque,20,nperseg=len(df.torque))
    f,ax = plt.subplots(constrained_layout=True)
    plt.semilogx(freqs,psd)
    ax.set(xlabel="Frequency (Hz)",ylabel="Power")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]} Welch PSD")
    return f

def plot_all_welch(paths):
    if isinstance(paths,str):
        paths = glob(paths)
    f,ax = plt.subplots(constrained_layout=True)
    for fn in paths:
        df = pd.read_excel(fn)
        freqs,psd = signal.welch(df.torque,20,nperseg=len(df.torque))
        plt.semilogx(freqs,psd)
    ax.set(xlabel="Frequency (Hz)",ylabel="Power")
    f.suptitle(f"Lubbering Welch PSD")
    return f

# plot the histogram
def plot_hist(path, bins=30, normed=True):
    df = pd.read_excel(path)
    f,ax = plt.subplots(constrained_layout=True)
    plt.hist(df.torque,bins,density=True)
    ax.set(xlabel="Torque",ylabel="Normed Population")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]} Histogram")
    return f

# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
def kernel_smooth(path):
    pass

# spline interpolation to raise sampling rate
def raise_sr(path,new_sr=100,smoothing=0.5,no_plot=True):
    from scipy.interpolate import UnivariateSpline
    # if it's a path load it in
    if isinstance(path):
        df = pd.read_excel(path)
        dist = df.distance
        torque = df.torque
    else:
        dist = path.distance
        torque = path.torque
    # define splint
    spl = UnivariateSpline(dist,torque)
    spl.set_smoothing_factor(smoothing)
    # new distance values
    new_dist = np.linspace(dist.min(),dist.max(),int(dist.shape[0]*(new_sr/20)))
    if no_plot:
        return spl(new_dist)
    f,ax = plt.subplots()
    ax.plot(dist,torque,'b-',label="Original")
    ax.plot(new_dist,spl(new_dist),'r-',label="Interpolated")
    ax.set(xlabel="Distance",ylabel="Torque")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]} Spline Interp. new fs={new_sr}")
    return f

# calculate the energy of the unfiltered torque
def calc_energy(path):
    df = pd.read_excel(path)
    return np.sum(df.torque.values**2)

# iterate over each file in the path and calculate signal energy
# plot the results as a scatter plot
def plot_mult_energy(paths):
    if isinstance(paths,str):
        paths = glob(paths,recursive=True)

    energy = [calc_energy(fn) for fn in paths]
    f,ax = plt.subplots(constrained_layout=True)
    ax.scatter(range(len(energy)),energy,s=100)
    ax.set(xlabel="Path Index",ylabel="Signal Energy")
    f.suptitle("Signal Energy of Unfiltered Torque")
    ax.set_xticks(range(len(paths)),[os.path.splitext(os.path.basename(p))[0] for p in paths])
    ax.tick_params(axis='x', rotation=90)
    return f

def plot_dist_diff_hist(paths):
    if isinstance(paths,str):
        paths = glob(paths)
    dist = []
    for fn in paths:
         dd = np.diff(pd.read_excel(fn).distance.values)
         dist.append(pd.DataFrame({'Dist Diff':dd,'File':len(dd)*[os.path.splitext(os.path.basename(fn))[0],]}))
    dist_all = pd.concat(dist)
    dist_all['Distance Change (mm)'] = dist_all['Dist Diff']/100
    ax = sns.histplot(dist_all,x='Distance Change (mm)')
    ax.figure.suptitle(f"Histogram of Distance Changes (mm) for Lubbering Data")
    return ax.figure
        
if __name__ == "__main__":
    pass
##    # check email from Paul for zip file of data
##    for fn in glob(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx")+glob(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx"):
##
##        ## try different rupture models ##
##        for m in ["l1","l2","rbf","cosine","normal","ar"]:
##            f = try_dynp(fn,3,model=m)
##            f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-dynp-model-{m}.png")
##            plt.close(f)
##        
##        for m in ["l1","l2","rbf","normal","ar"]:
##            f = try_pelt(fn,model=m)
##            if not (f is None):
##                f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-pelt-model-{m}.png")
##                plt.close(f)
##
##        for m in ["linear","rbf","cosine"]:
##            f = kernel_seg(fn,3,model=m)
##            f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-kernel-model-{m}.png")
##            plt.close(f)
##
##        for m in ["l1","l2","rbf","cosine","normal","ar"]:
##            f = binary_segmentation(fn,3,model=m)
##            f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-binset-model-{m}.png")
##            plt.close(f)
##
##        for m in ["l1","l2","rbf","cosine","normal"]:
##            f = window_sliding_seg(fn,3,model=m)
##            f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-window-model-{m}.png")
##            plt.close(f)
##
##        f = try_all_rupture(fn,3,model="rbf")
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-rupture-all.png")
##        plt.close(f)
##
##        ## investigate frequency domain ##
##        f = plot_fft(fn)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-fft.png")
##        plt.close(f)
##
##        f = plot_stft(fn,plot_signal=True)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-stft.png")
##        plt.close(f)
##
##        f = plot_stft(fn,log_scale=True,plot_signal=True)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-stft-log-scale.png")
##        plt.close(f)
##
##        f = apply_lowpass_filter(fn,cutoff=7.5)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-low-pass-filter-7_5.png")
##        plt.close(f)
##
##        f = apply_lowpass_filter(fn,cutoff=1.0)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-low-pass-filter-1_0.png")
##        plt.close(f)
##
##        f = apply_highpass_filter(fn,cutoff=9)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-high-pass-filter-9_0.png")
##        plt.close(f)
##
##        f = plot_spect(fn)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-spect.png")
##        plt.close(f)
##
##        f = plot_welch(fn)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-welch.png")
##        plt.close(f)
##
##        # plot the histogram
##        f = plot_hist(fn)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-hist.png")
##        plt.close(f)
##
        # investigate the rolling gradient
##        f = try_rg(fn,N=[3,5])
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-rolling-gradient-N-3-5.png")
##        plt.close(f)
##
##        f = try_rg(fn,N=10)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-rolling-gradient.png")
##        plt.close(f)
##
##        # try some different smoothing methods
##        f = rms_smooth_rg(fn)
##        f.savefig(f"luebering_data_test/plots/{os.path.splitext(os.path.basename(fn))[0]}-rms-rolling-gradient.png")
##        plt.close(f)
##        
##    f= plot_mult_energy(glob(r"luebering_data_test/4B Luebbering/New cutter/Luebbering data/*.xlsx")+glob(r"luebering_data_test/4B Luebbering/Worn cutter/Luebbering data/*.xlsx"))
##    f.savefig(r"luebering_data_test/plots/lueberring_4B_signal_energy.png")
##    plt.close(f)
