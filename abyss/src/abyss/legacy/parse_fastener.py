import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import os

def loadFastenerTxt(path:str)->pd.core.frame.DataFrame:
    '''
        Load a fastener curve stored in a text file and convert it to a
        pandas dataframe

        Inputs:
            path : Complete path to file

        Returns pandas dataframe
    '''
    with open(path) as file:
        # skip 4 lines
        for _ in range(4):
            file.readline()
        # set the columns
        cols = ["Label",]+file.readline().strip().split('\t')
        nc = len(cols)
        # append data to a list
        data = []
        for line in file:
            d = line.strip().split('\t')
            # if there's no label in first column
            if len(d) != nc:
                d = [None,]+ list(map(float,d))
            else:
                d = [d[0],]+list(map(float,d[1:]))
            data.append(d)
    # convert to a dataframe
    return pd.DataFrame(data,columns=cols)

def batchLoadFastenerTxt(path:str)->pd.DataFrame:
    '''
        Iterate over folder of fastener text files and combine them together
        into a single dataframe

        An additional column called Filepath is added so each curve can be
        separated

        Inputs:
            path : Wildcard path to folder of text files

        Returns pandas dataframe
    '''
    dfs = []
    for fn in glob(path):
        df = loadFastenerTxt(fn)
        df['Filepath'] = fn
        dfs.append(df)
    return pd.concat(dfs)

def plotFastenerCurve(df:[str,pd.DataFrame],**kwargs)->dict:
    '''
        Plot a dataframe containing a single fastener curve

        Creates a series of plots of different variables against time and angle.
        Plots are stored in a dict where the key can be used in the saved filename

        Inputs:
            df : Path to fastener text file or already loaded dataframe
            limit_angle : Flag to limit the x-axis to the angle limits. Default False.
            append_str : Append the given string to the start of the keys. Default ""
            

        Return dict of plots
    '''
    if isinstance(df,str):
        df = loadFastenerTxt(df)
    plots = {}
    vline_cols = {'Am':'r','Sa':'g'}
    limit_angle = kwargs.get("limit_angle",False)
    append_str = kwargs.get("append_str","")
    # extract label locations
    label_dict = {l:df[df.Label==l] for l in df.Label.unique() if l}
    # get torque min max for vlines
    tmax = df['Torque (Nm)'].max()
    tmin = df['Torque (Nm)'].min()

    amax = df['Angle (°)'].max()
    amin = df['Angle (°)'].min()

    ## plot torque angle curve
    f,ax = plt.subplots()
    ax.plot(df['Angle (°)'].values,df['Torque (Nm)'].values,'b-',label=r"Torque/Angle")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    if limit_angle:
        ax.set_xlim(amin,amax)
    ax.set(xlabel="Angle (°)",ylabel="Torque (Nm)",title="Torque-Angle curve")
    plots[append_str+'torque-angle'] = f

    ## plot torque time curve
    f,ax = plt.subplots()
    ax.plot(df['Time (s)'].values,df['Torque (Nm)'].values,'b-',label=r"Torque/Time")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    ax.set(xlabel="Time (s)",ylabel="Torque (Nm)",title="Torque-Time curve")
    plots[append_str+'torque-time'] = f

    ## plot current curve
    tmax = df['Current (A)'].max()
    tmin = df['Current (A)'].min()
    
    f,ax = plt.subplots()
    ax.plot(df['Time (s)'].values,df['Current (A)'].values,'b-',label=r"Current/Time")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    ax.set(xlabel="Time (s)",ylabel="Current (A)",title="Current-Time curve")
    plots[append_str+'current-time'] = f

    f,ax = plt.subplots()
    ax.plot(df['Angle (°)'].values,df['Current (A)'].values,'b-',label=r"Current/Time")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    if limit_angle:
        ax.set_xlim(amin,amax)
    ax.set(xlabel="Angle (°)",ylabel="Current (A)",title="Current-Angle curve")
    plots[append_str+'current-angle'] = f

    ## plot torque current
    f,ax = plt.subplots()
    ax.plot(df['Torque (Nm)'].values,df['Current (A)'].values,'b-',label=r"Torque/Current")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Torque (Nm)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    ax.set(xlabel="Torque (Nm)",ylabel="Torque (Nm)",title="Torque-Current curve")
    plots[append_str+'torque-current'] = f

    ## plot T rate curve
    f,ax = plt.subplots()
    tmax = df['T. rate (Nm/°)'].max()
    tmin = df['T. rate (Nm/°)'].min()

    ax.plot(df['Time (s)'].values,df['T. rate (Nm/°)'].values,'b-',label=r"T. rate (Nm/°)/Time")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    ax.set(xlabel="Time (s)",ylabel="T. rate (Nm/°)",title="T. rate (Nm/°) Time curve")
    plots[append_str+'Trate-time'] = f

    f,ax = plt.subplots()
    ax.plot(df['Angle (°)'].values,df['T. rate (Nm/°)'].values,'b-',label=r"T. rate (Nm/°)/Angle")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    if limit_angle:
        ax.set_xlim(amin,amax)
    ax.set(xlabel="Angle (°)",ylabel="T. rate (Nm/°)",title="T. rate (Nm/°) Angle curve")
    plots[append_str+'Trate-angle'] = f

    ## plot Tension curve
    tmax = df['Tension (daN)'].max()
    tmin = df['Tension (daN)'].min()

    f,ax = plt.subplots()
    ax.plot(df['Time (s)'].values,df['Tension (daN)'].values,'b-',label=r"Tension (daN)/Time")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    ax.set(xlabel="Time (s)",ylabel="Tension (daN)",title="Tension (daN) Time curve")
    plots[append_str+'tension-time'] = f

    f,ax = plt.subplots()
    ax.plot(df['Angle (°)'].values,df['Tension (daN)'].values,'b-',label=r"Tension (daN)/Time")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    if limit_angle:
        ax.set_xlim(amin,amax)
    ax.set(xlabel="Angle (°)",ylabel="Tension (daN)",title="Tension (daN) Angle curve")
    plots[append_str+'tension-angle'] = f

    ## plot speed curve
    f,ax = plt.subplots()
    tmax = df['Speed (rpm)'].max()
    tmin = df['Speed (rpm)'].min()

    ax.plot(df['Time (s)'].values,df['Speed (rpm)'].values,'b-',label=r"Speed (rpm)/Time")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    ax.set(xlabel="Time (s)",ylabel="Speed (rpm)",title="Speed (rpm) Time curve")
    plots[append_str+'speed-time'] = f

    f,ax = plt.subplots()
    ax.plot(df['Angle (°)'].values,df['Speed (rpm)'].values,'b-',label=r"Speed (rpm)/Angle")
    if (tmax!=0) or (tmin!=0):
        for k,v in label_dict.items():
            ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=k)
    ax.legend()
    if limit_angle:
        ax.set_xlim(amin,amax)
    ax.set(xlabel="Angle (°)",ylabel="Speed (rpm)",title="Speed (rpm) Angle curve")
    plots[append_str+'speed-angle'] = f

    return plots

def plotSaveFastenerCurve(path:str,opath:str,**kwargs):
    '''
        Load a fastener curve, plot and save plots of the variables

        See plotFastenerCurve

        Inputs:
            path : Path to fastener text file.
            opath : Output folder to save the plots
            **kwargs : See plotFastenerCurve
    '''
    if not opath:
        raise ValueError("Path has to be non-empty!")
    plots = plotFastenerCurve(path,**kwargs)
    for label,p in plots.items():
        p.savefig(os.path.join(opath,f"{label}-fastener-curve.png"))
        plt.close(p)

def plotOverlapFastenerCurve(df:[pd.DataFrame,list],key:str,use_time:bool=False,use_relative:bool=True,limit_angle:bool=False)->[matplotlib.figure.Figure,dict]:
    '''
        Plot the target variable from the series of dataframes on the same axis

        The key has to be either:
            - 'all' indicating all variables
            - Specific column present in all dataframes

        If it's a single dataframe, it must have a Filepath column so the function knows how to parse them.

        If a specific key, the user can set the flags use_time and use_relative to control whethe the x-axis is Time (s) and relative to the smallest value respecively.
        The use_relative flag is useful when plotting using the Angle column which doesn't always start at 0.

        If key is 'all', then a dictionary of figures similar to plotFastenerCurve is created where a series of figures is made and the values from each dataframe are plotted
        on the same axis

        Inputs:
            df : Single dataframe from batchLoadFastenerTxt or a list of dataframes.
            key : Target variable to plot. Has to exist or be 'all',
            use_time : Use values of Time (s) for the x-axis. Default False
            use_relative : Use relative to start values so the curves align. Default True.
            limit_angle : When plotting against Angle, set the x-axis limits to the data limits. Default False

        Returns a single figure if key is not all else a dictionary of figures is returned
    '''
    # check for null key
    if not key:
        raise ValueError("Key cannot be empty or null!")
    if isinstance(df,pd.DataFrame):
        if not ('Filepath' in df.columns):
            raise KeyError("Missing Filename column! Do not know how to parse dataframe into separate curves")
    # if a specific key
    if key != 'all':
         # check that the key exists
        if isinstance(df,list):
            if not all([key in d.columns for d in df]):
                raise KeyError("Target key does not exist in all dataframes!")
        else:
            if not (key in df.columns):
                raise KeyError("Target key does not exist in dataframe!")
        f,ax = plt.subplots()
        if isinstance(df,list):
            for i,d in enumerate(df):
                x = d["Time (s)" if use_time else 'Angle (°)']
                if use_relative:
                    x -= x.min()
                ax.plot(x,d[key],label=f"DF {i+1}")
        elif isinstance(df,pd.DataFrame):
            for fn,d in df.groupby("Filepath"):
                x = d["Time (s)" if use_time else 'Angle (°)']
                if use_relative:
                    x -= x.min()
                ax.plot(x,d[key],label=os.path.splitext(os.path.basename(fn))[0])
        ax.legend()
        ax.set(xlabel=("Relative " if use_relative else "") + ("Time (s)" if use_time else 'Angle (°)'),ylabel=key,title=f"Overlap plot of {key}")
        return f
    else:
        # setup dict to store plots
        plots = {}
        for k in ['torque-angle','torque-time','current-time','current-angle','torque-current','Trate-time','Trate-angle','tension-time','tension-angle','speed-time','speed-angle']:
            plots[k] = plt.subplots()
        # set label line cols
        vline_cols = {'Am':'r','Sa':'k'}
        # set iteration function
        if isinstance(df,pd.DataFrame):
            def get_next_df(df):
                for fn,dd in df.groupby("Filepath"):
                    append_str = os.path.splitext(os.path.basename(fn))[0]
                    if use_relative:
                        dd['Time (s)'] -= dd['Time (s)'].min()
                        dd['Angle (°)'] -= dd['Angle (°)'].min()
                    yield dd,append_str
        else:
            def get_next_df(df):
                # split by file path
                for i,dd in enumerate(df):
                    append_str = f"DF {i+1}"
                    if use_relative:
                        dd['Time (s)'] -= dd['Time (s)'].min()
                        dd['Angle (°)'] -= dd['Angle (°)'].min()
                    yield dd,append_str
        # split by file path
        for dd,append_str in get_next_df(df):
            # extract label locations
            #print(dd,append_str,dd.Label.unique())
            label_dict = {l:dd[dd.Label==l] for l in dd.Label.unique() if l}
            # get torque min max for vlines
            tmax = dd['Torque (Nm)'].max()
            tmin = dd['Torque (Nm)'].min()

            amax = dd['Angle (°)'].max()
            amin = dd['Angle (°)'].min()

            ## plot torque angle curve
            f,ax = plots['torque-angle']
            ax.plot(dd['Angle (°)'].values,dd['Torque (Nm)'].values,'-',label=fr"{append_str} Torque/Angle")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            if limit_angle:
                ax.set_xlim(amin,amax)
            ax.set(xlabel=("Relative " if use_relative else "") + "Angle (°)",ylabel="Torque (Nm)",title="Torque-Angle curve")

            ## plot torque time curve
            f,ax = plots['torque-time']
            ax.plot(dd['Time (s)'].values,dd['Torque (Nm)'].values,'-',label=fr"{append_str} Torque/Time")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            ax.set(xlabel=("Relative " if use_relative else "") + "Time (s)",ylabel="Torque (Nm)",title="Torque-Time curve")

            ## plot current curve
            tmax = dd['Current (A)'].max()
            tmin = dd['Current (A)'].min()
            
            f,ax = plots['current-time']
            ax.plot(dd['Time (s)'].values,dd['Current (A)'].values,'-',label=fr"{append_str} Current/Time")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            ax.set(xlabel=("Relative " if use_relative else "") + "Time (s)",ylabel="Current (A)",title="Current-Time curve")

            f,ax = plots['current-angle']
            ax.plot(dd['Angle (°)'].values,dd['Current (A)'].values,'-',label=fr"{append_str} Current/Time")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            if limit_angle:
                ax.set_xlim(amin,amax)
            ax.set(xlabel=("Relative " if use_relative else "") + "Angle (°)",ylabel="Current (A)",title="Current-Angle curve")

            ## plot torque current
            f,ax = plots['torque-current']
            ax.plot(dd['Torque (Nm)'].values,dd['Current (A)'].values,'-',label=fr"{append_str} Torque/Current")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Torque (Nm)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            ax.set(xlabel=("Relative " if use_relative else "") + "Torque (Nm)",ylabel="Torque (Nm)",title="Torque-Current curve")

            ## plot T rate curve
            f,ax = plots['Trate-time']
            tmax = dd['T. rate (Nm/°)'].max()
            tmin = dd['T. rate (Nm/°)'].min()

            ax.plot(dd['Time (s)'].values,dd['T. rate (Nm/°)'].values,'-',label=fr"{append_str} T. rate (Nm/°)/Time")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            ax.set(xlabel=("Relative " if use_relative else "") + "Time (s)",ylabel="T. rate (Nm/°)",title="T. rate (Nm/°) Time curve")

            f,ax = plots['Trate-angle']
            ax.plot(dd['Angle (°)'].values,dd['T. rate (Nm/°)'].values,'-',label=fr"{append_str} T. rate (Nm/°)/Angle")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            if limit_angle:
                ax.set_xlim(amin,amax)
            ax.set(xlabel=("Relative " if use_relative else "") + "Angle (°)",ylabel="T. rate (Nm/°)",title="T. rate (Nm/°) Angle curve")

            ## plot Tension curve
            tmax = dd['Tension (daN)'].max()
            tmin = dd['Tension (daN)'].min()

            f,ax = plots['tension-time']
            ax.plot(dd['Time (s)'].values,dd['Tension (daN)'].values,'-',label=fr"{append_str} Tension (daN)/Time")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            ax.set(xlabel=("Relative " if use_relative else "") + "Time (s)",ylabel="Tension (daN)",title="Tension (daN) Time curve")

            f,ax = plots['tension-angle']
            ax.plot(dd['Angle (°)'].values,dd['Tension (daN)'].values,'-',label=fr"{append_str} Tension (daN)/Time")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            if limit_angle:
                ax.set_xlim(amin,amax)
            ax.set(xlabel=("Relative " if use_relative else "") + "Angle (°)",ylabel="Tension (daN)",title="Tension (daN) Angle curve")

            ## plot speed curve
            f,ax = plots['speed-time']
            tmax = dd['Speed (rpm)'].max()
            tmin = dd['Speed (rpm)'].min()

            ax.plot(dd['Time (s)'].values,dd['Speed (rpm)'].values,'-',label=fr"{append_str} Speed (rpm)/Time")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Time (s)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            ax.set(xlabel=("Relative " if use_relative else "") + "Time (s)",ylabel="Speed (rpm)",title="Speed (rpm) Time curve")

            f,ax = plots['speed-angle']
            ax.plot(dd['Angle (°)'].values,dd['Speed (rpm)'].values,'-',label=fr"{append_str} Speed (rpm)/Angle")
            if (tmax!=0) or (tmin!=0):
                for k,v in label_dict.items():
                    ax.vlines(v['Angle (°)'].values,tmin,tmax,vline_cols[k],linestyles='dashed',label=f"{append_str} "+k)
            ax.legend()
            if limit_angle:
                ax.set_xlim(amin,amax)
            ax.set(xlabel=("Relative " if use_relative else "") + "Angle (°)",ylabel="Speed (rpm)",title="Speed (rpm) Angle curve")
     
        return plots

def findAreaUnderCurve(df:[str,pd.DataFrame],x='Angle (°)',y='Torque (Nm)'):
    if isinstance(df,str):
        df = loadFastenerTxt(df)
    return np.trapz(df[y],x=df[x])

def scatterAreaUnderCurve(df:[pd.DataFrame,list],x:str='Angle (°)',y:str='Torque (Nm)',**kwargs)->matplotlib.figure.Figure:
    if isinstance(df,pd.DataFrame):
        auc = [findAreaUnderCurve(df[df.Filepath == fn],x,y) for fn in df.Filepath.unique()]
        xx = [os.path.splitext(os.path.basename(fn))[0] for fn in df.Filepath.unique()]
    else:
        auc = [findAreaUnderCurve(dd,x,y) for dd in df]
        xx = list(range(len(auc)))
    f,ax = plt.subplots(constrained_layout=True)
    ax.scatter(list(range(len(xx))),auc,**kwargs)
    if isinstance(df,pd.DataFrame):
        ax.set_xticks(list(range(len(xx))),xx,rotation=90)
        ax.set_xlabel("Filenames")
    else:
        ax.set_xlabel("File Index")
    ax.set(ylabel=f"Area Under {x}-{y} curve",title=f"Area Under {x}-{y} curve")
    return f

def isOKCurve(path:str)->bool:
    '''
        Parse a fastener curve text file and find if it contains OK

        The string of OK indicates that it's classified as an OK file

        Inputs:
            path : Complete path to file

        Returns bool
    '''
    with open(path) as file:
        # skip 4 lines
        for _ in range(2):
            file.readline()
    return file.readline().strip().split(' ')[-1] == 'OK'

def howManyOKCurves(path:str)->int:
    '''
        Iterate over curves in a folder and count how many are
        regarded as OK curves

        Uses isOKCurve to determine this

        Inputs:
            path : Wildcard path to folder of text files

        Returns int  
    '''
    ct = 0
    for fn in glob(path):
        ct += int(isOKCurve(path))
    return ct

def howManyNOKCurves(path:str)->int:
    '''
        Iterate over curves in a folder and count how many are
        regarded as NOK curves

        Uses isOKCurve to determine this

        Inputs:
            path : Wildcard path to folder of text files

        Returns int  
    '''
    ct = 0
    for fn in glob(path):
        ct += int(not isOKCurve(path))
    return ct

def getTriggerTorque(path:str)->float:
    '''
        Parse a fastener curve text file and find the trigger torque

        The trigger torque is the required torque required to start recording.

        Torque is in Nm.

        Inputs:
            path : Complete path to file

        Returns float
    '''
    with open(path) as file:
        # skip 4 lines
        for _ in range(3):
            file.readline()
        return float(file.readline().strip().split('Trigger torque: ')[-1].split('\t')[0].split(' Nm')[0])

def getDatetimeStamp(path:str,as_dt:bool=True)->(tuple,dt):
    '''
        Parse a fastener curve text file and find the datetime stamp at the top of the file

        The datetime stamp is the adjusted creation datetime stamp. The one
        later in the file is from the controller's internal DT stamp which is
        often out of date.

        Datetime stamp is converted to a datetime.datetime object if
        as_dt flag is set to True.

        Inputs:
            path : Complete path to file

        Returns float
    '''
    with open(path) as file:
        pts = file.readline().strip().split(' ')
    date = pts[-2]
    time = pts[-1]
    if as_dt:
        return dt.strptime(date+' '+time,"%d/%m/%y %H:%M:%S")
    else:
        return date,time

if __name__ == "__main__":
    #df = batchLoadFastenerTxt(r"C:\Users\david\Downloads\AMRC - 01\*.txt")
    #df = loadFastenerTxt(r"C:\Users\david\Downloads\AMRC - 01\AMRC - 0101001.txt")
##    for fn in glob(r"C:\Users\david\Downloads\AMRC - 01\*.txt"):
##        ap = os.path.splitext(os.path.basename(fn))[0]
##        plotSaveFastenerCurve(fn,"AMRC-01-plots",append_str=ap+"-")
    #print(getDatetimeStamp(r"C:\Users\uos\Downloads\AMRC - 01\AMRC - 0101001.txt"))
    pass
