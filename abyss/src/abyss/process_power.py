import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

def loadPowerCSV(path,split=True):
    # load full dataframe into pandas
    df = pd.read_csv(path,sep=';',skiprows=13)
    # convert timestamp to seconds
    tt = pd.to_timedelta(df['Timestamp']).astype('timedelta64[ns]')
    tt -= tt.min()
    df['Timestamp'] = tt
    if not split:
        return df
    meanIR = np.ma.masked_invalid(df['IRange[A]'].values).max()
    # nans are in groups of 3
    # find where middle indicies for nans are
    ii = [0,] + np.where(df['URMS[V]'].isnull())[0][1::3].tolist()
    # split into groups
    gps = [df.iloc[cA:cB+1].dropna() for cA,cB in zip(ii,ii[1:])]
    # filter groups to those with high IRange
    #gps = list(filter(lambda x : ((np.ma.masked_invalid(x['IRange[A]'].values).max()==meanIR) and (x.shape[0]>50)),gps))
    #gps = list(filter(lambda x : (np.ma.masked_invalid(x['IRange[A]'].values).max()>meanIR),gps))
    #print([gg.shape[0] for gg in gps])
    gps = list(filter(lambda x : x.shape[0]>50,gps))
    gps = list(filter(lambda x : (np.ma.masked_invalid(x['IRange[A]'].values).max()>(meanIR*0.3)),gps))
    return gps

def plotPowerCSV(path,as_groups=False,interp_nans=False):
    gps = loadPowerCSV(path,as_groups)
    # get keys except for timestamp
    keys = (gps[0] if as_groups else gps).keys()[:-1] 
    nk = len(keys)
    # get how many rows and columns are needed for the plot
    row = int(np.ceil(np.sqrt(nk)))
    col = int(np.floor(np.sqrt(nk)))
    # create figure
    f = plt.figure(constrained_layout=True)
    # iterate over keys adding axes
    for ki,kk in enumerate(keys,start=1):
        ax = f.add_subplot(row,col,ki)
        if as_groups:
            for gi,gg in enumerate(gps):
                if interp_nans:
                    # replace and infs with NaNs
                    if np.isinf(gg[kk].values).any():
                        gg[kk].values[np.isinf(gg[kk].values)] = np.nan
                    # interpolate to fill nan values
                    gg[kk] = gg[kk].interpolate()
                ax.plot(gg['Timestamp'].values,gg[kk].values)
        else:
            if interp_nans:
                # replace and infs with NaNs
                if np.isinf(gps[kk].values).any():
                    gps[kk].values[np.isinf(gps[kk].values)] = np.nan
                # interpolate to fill nan values
                gps[kk] = gps[kk].interpolate()
            ax.plot(gps['Timestamp'].values,gps[kk].values)
        ax.set(xlabel="Time (s)",ylabel=kk,title=kk)
    # set figure title to filename
    f.suptitle(os.path.splitext(os.path.basename(path))[0])
    return f

def plotPowerOverlap(path,interp_nans=False):
    if isinstance(path,str):
        path = glob(path)
    # load first file to get keys
    gps = loadPowerCSV(path[0],False)
    # get keys except for timestamp
    keys = gps.keys()[:-1]
    nk = len(keys)
    # get how many rows and columns are needed for the plot
    row = int(np.ceil(np.sqrt(nk)))
    col = int(np.floor(np.sqrt(nk)))
    # create figure
    f = plt.figure(constrained_layout=True)
    axes = {}
    for fn in path:
        gps = loadPowerCSV(fn,False)
        for ki,kk in enumerate(keys,start=1):
            if not (kk in axes):
                axes[kk] = f.add_subplot(row,col,ki)
            ax = axes[kk]
            #ax.plot(gps['Timestamp'].values,gps[kk].values)
            idx = np.linspace(0,1,len(gps[kk].values))
            if interp_nans:
                # replace and infs with NaNs
                if np.isinf(gps[kk].values).any():
                    gps[kk].values[np.isinf(gps[kk].values)] = np.nan
                # interpolate to fill nan values
                gps[kk] = gps[kk].interpolate()
            ax.plot(idx,gps[kk].values,label=os.path.splitext(os.path.basename(fn))[0])
    for kk,aa in axes.items():
        aa.set(xlabel="Normalised Index",ylabel=kk,title=kk)
        aa.legend()
    return f

def dtwAlign(path):
    from tslearn import metrics
    # make list of paths
    paths = glob(path)
    # set reference
    A = loadPowerCSV(paths.pop(0),False)
    B = loadPowerCSV(paths.pop(0),False)
    # load and condition IRMS
    AI = A['IRMS[A]'].values.flatten()
    AI = np.nan_to_num(AI,posinf=np.ma.masked_invalid(AI).max())
    BI = B['IRMS[A]'].values.flatten()
    BI = np.nan_to_num(BI,posinf=np.ma.masked_invalid(BI).max())
    # find path to align the two signals
    align,_ = metrics.dtw_subsequence_path(AI,BI)
    # extract values
    AA = []
    BB = []
    for ii,jj in align:
        AA.append(AI[ii])
        BB.append(BI[jj])
    AA = np.array(AA)
    BB = np.array(BB)
    aligned = [AA,BB]

    ref = AI

    for fn in paths:
        # set reference
        B = loadPowerCSV(fn,False)
        BI = B['IRMS[A]'].values.flatten()
        BI = np.nan_to_num(BI,posinf=np.ma.masked_invalid(BI).max())
        # find path to align the two signals
        align,_ = metrics.dtw_subsequence_path(BI,ref)
        # extract values
        BB = []
        for ii,jj in align:
            BB.append(BI[jj])
        BB = np.array(BB)
        aligned.append(BB)
    # add to the list
    f,ax = plt.subplots()
    for gg in aligned:
        ax.plot(gg)
    return f
            
if __name__ == "__main__":
    from tslearn import metrics
    #dtwAlign(r"C:\Users\uos\Downloads\AA*.csv")
    #ax.set_ylim(min(np.ma.masked_invalid(AA).min(),np.ma.masked_invalid(BB).min()),max(np.ma.masked_invalid(AA).max(),np.ma.masked_invalid(BB).max()))
    plotPowerOverlap(glob(r"C:\Users\uos\Downloads\AA*.csv")[-2:])
    for fn in glob(r"C:\Users\uos\Downloads\AA*.csv"):
        plotPowerCSV(fn,False,False)
        plotPowerCSV(fn,True,False)
        plt.show()
