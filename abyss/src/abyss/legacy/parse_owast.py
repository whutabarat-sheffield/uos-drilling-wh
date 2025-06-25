import xml.etree.ElementTree as ET
import pandas as pd
import abyss.dataparser as dp
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from dataparser import printTDMSInfo, iterTDMSFile, loadSetitecXls
from glob import glob
import numpy as np
import os
import cv2

def parseXMLToDict(path):
    tree = ET.parse(path)
    root = tree.getroot()
    clusters = {}
    for c in root:
        if c.tag == 'Cluster':
            clusters.update(parseClusterToDict(c))
        elif c.tag == 'Array':
            clusters.update(parseArrayToDict(c))
        elif c.tag == 'String':
            clusters[c.find('Name').text] = c.find('Val').text
    return clusters

# parse a cluster from the xml document
def parseClusterToDict(cluster):
    data = {}
    for c in cluster:
        if c.tag == 'NumElts':
            continue
        if c.tag == 'Name':
            tag_name = c.text
            data[tag_name] = {}
            continue
        if c.tag == 'Cluster':
            tag_name = c.find('Name').text
            data[tag_name] = parseClusterToDict(c)
            continue
        # if a data value
        name = c.find('Name').text
        val = c.find('Val').text
        data[tag_name][name] = val
    return data

# parse an array from the xml document
def parseArrayToDict(cluster):
    data = {}
    for c in cluster:
        if c.tag == 'Dimsize':
            continue
        if c.tag == 'Name':
            tag_name = c.text
            data[tag_name] = {}
            continue
        if c.tag == 'Cluster':
            tag_name = c.find('Name').text
            data[tag_name] = parseClusterToDict(c)
            continue
        # if a data value
        name = c.find('Name').text
        val = c.find('Val').text
        data[tag_name][name] = val
    return data

def combineCommon(cdicts):
    # clusters have common keys
    comb = {}
    for k,v in cdicts:
        # if dictionary has not been initialised
        # use keys from first dictionary 
        if not comb:
            comb = {key : [val,] for key,val in v.items()}
            continue
        # if it has been initialised
        # iterate over keys and a
        for key,val in v.items():
            pass

def plotTDMSFile(path,**kwargs):
    '''
        Iterates over the channels of the given TDMS file and plots each channel on a different axis

        Generates and saves a figure with the saved figure having the same name as the source file

        Inputs:
            path : Full path to TDMS file
            opath : Output path. Default ''
            save_fig : Flag to save files to opath. Default True.
    '''
    nchannels = 0
    with TdmsFile(path) as file:
        for g in file.groups():
            nchannels += len(g.channels())
    sq = np.sqrt(nchannels)
    nr = int(np.ceil(sq))
    nc = int(np.floor(sq))
    f,ax = plt.subplots(nrows=nr,ncols=nc,constrained_layout=True,figsize=(6*nc,6*nr))
    df = pd.DataFrame()
    if nchannels>1:
        ax = ax.flatten()
    else:
        ax = [ax,]
    for aa,(group,channel,data,units) in zip(ax,iterTDMSFile(path,ret_units=True)):
        aa.plot(data.index,data.values.flatten())
        df[rf"{group}\{channel}"] = data.values.flatten()
        aa.set(xlabel="Time (s)",ylabel=f"{channel} ({units})",title=rf"{group}\{channel}")
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}")
    if kwargs.get("save_fig",True):
        opath = kwargs.get("opath",'')
        f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(path))[0]}.png"))
        #df.to_csv(os.path.join(opath,f"{os.path.splitext(os.path.basename(path))[0]}.csv"),chunksize=100)
    plt.close(f)

def _exportTDMS(path,opath='OWAST'):
    '''
        Load TDMS file into a pandas and export as TDMS

        Alternative is using nptdms.TdmsFile.as_dataframe method. This approach is more to ensure
        a custom column formatting
    '''
    df = pd.DataFrame()
    for group,channel,data,units in iterTDMSFile(path,ret_units=True):
        df[rf"{group}\{channel}"] = data.values.flatten()
    df.to_csv(os.path.join(opath,f"{os.path.splitext(os.path.basename(path))[0]}.csv"),chunksize=100)

def exportTDMSToCSV(path,**kwargs):
    '''
        Export folder of TDMS files to individual CSV files

        Uses multiprocessing.Pool

        Inputs:
            path : Wildcard path to TDMS files
            workers : Number of workers to use
    '''
    import multiprocessing as mp
    with mp.Pool(kwargs.get("workers",8)) as pool:
        pool.map(_exportTDMS,glob(path))

def plotPlugData(path):
    data = pd.read_csv(path)
    # convert timestamp to number of seconds from start
    data.Timestamp = pd.to_datetime(data.Timestamp,format="%d/%m/%Y %H:%M:%S")
    ts = diff = (data.Timestamp - pd.to_datetime('1990', format='%Y')).dt.total_seconds()
    ts -= ts.min()
    # make plot
    f,ax = plt.subplots(ncols=len(data.columns)-1,sharex=True,constrained_layout=True,figsize=(16,6))
    for aa,cc in zip(ax.flatten(),data.columns[1:]):
        aa.plot(ts,data[cc])
        aa.set(xlabel="Timestamp (s)",ylabel=cc,title=cc)
    f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}")
    f.savefig(f"{os.path.splitext(os.path.basename(path))[0]}.png")
    plt.close(f)

def plotMaxImageEntropy(path):
    '''
        Loads video, calculates max entropy of each frame and plots it

        Entropy is calculated using skimage.filters.rank.entropy
        and a skimage.morphology.disk(10)

        Inputs:
            path : Input video file path
    '''
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open file {path}")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{length} frames")
    ent = []
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret:
            ent.append(entropy(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), disk(10)).max())
        else:
            raise ValueError("Unable to open image")
    f,ax = plt.subplots()
    ax.plot(ent)
    ax.set(xlabel="Frame Index",ylabel="Max Entropy",title=os.path.splitext(os.path.basename(path))[0])
    return f

def _combine(sf,tf,search,opath):
    '''
        Load the Spreadsheet of EADU files and corresponding TDMS files and combine the files together
    
        Used in combineEADUAndOWASTMT

        Inputs:
            sf : Name of Setitec Stroke file to search for
            tf : Name of TDMS folder to search for
            search : Path to search for sf and tf in
            opath : Where to write the combined compressed file to
    '''
    sp = os.path.join(search,"*",sf)
    eadu = glob(os.path.join(search,"*",sf))
    if len(eadu)==0:
        print(f"Failed to find file {sp}!")
        return
    eadu = eadu[0]
    edata = loadSetitecXls(eadu,"auto_data")
    # search for tdms folder
    tp = os.path.join(search,"*",tf,"*.tdms")
    owast = glob(tp)
    if len(owast)==0:
        print(f"Failed to find TDMS folder {tp}!")
        return
    # for each
    channels = []
    nchannels=0
    maxsz = 0
    for fn in owast:
        for group,channel,data,units in iterTDMSFile(fn,ret_units=True):
            #print(f"{maxsz} vs {len(data)}")
            maxsz = max(maxsz,len(data))
            channels.append(rf"{group}\{channel}")
    nchannels = len(channels)
    print("max: ",maxsz)
    print("nchannels ",nchannels)
    # construct dataframe
    df = pd.DataFrame(np.zeros((maxsz,nchannels)),columns=channels)
    # iterate over files again
    for fn in owast:
        # iterate through file
        for group,channel,data,units in iterTDMSFile(fn,ret_units=True):
            df[rf"{group}\{channel}"].iloc[:data.values.shape[0]] = data.values.flatten()
    # combine EADU dataframe and OWAST dataframe together
    print("combining with EADU data")
    dfcomb = pd.concat([edata,df],axis=1)
    # save compressed
    print(f"writing to {os.path.join(opath,os.path.splitext(os.path.basename(sf))[0])+'.gz'}")
    dfcomb.to_csv(os.path.join(opath,os.path.splitext(os.path.basename(sf))[0])+".gz",index=False,compression=compression_opts,chunksize=1000)

def combineEADUAndOWASTMT(path,search=rf"OWAST_Drilling_Trials\3_Drilling",opath='',workers=2):
    '''
        Load the Spreadsheet of EADU files and corresponding TDMS files and combine the files together

        Multiprocessing version of combineEADUAndOWAST

        WARNING: This program takes a while to run and uses a lot of RAM due to the amount of data involved.

        It is recommended to keep the number of workers low due to the size of the files involved.

        Inputs:
            path : Input path to custom spreadsheet e.g. OWAST_Drilling_Trials\3_Drilling\Copy of Drilling Trials_Experiments_Sheet.xlsx
            search : Path to folder containing EADU and TDMS files. Default OWAST_Drilling_Trials\3_Drilling
            opath : Where to write the GZ files to. Default local.
            workers : Number of workers to use. Default 2.
    '''
    import multiprocessing as mp
    compression_opts = dict(method='gzip',compresslevel=9) 
    data = pd.read_excel(path,header=0,usecols="B:C")
    data.dropna(inplace=True)
    mp.Pool(workers).starmap(_combine,[(sf,tf,search,opath) for sf,tf in zip(data['Stroke file'].values,data['TDMS folder'].values)])

def findMaxSize(path):
    '''
        Scan TDMS file and find the number of channels and the max size of each dataset

        Input:
            path : Filepath to TDMS file

        Returns max channel size and number of channels
    '''
    maxsz = 0
    nc = 0
    for _,_,data,_ in iterTDMSFile(path,ret_units=True):
        maxsz = max(maxsz,len(data))
        nc += 1
    return maxsz,nc

def combineEADUAndOWAST(path,search=rf"OWAST_Drilling_Trials\3_Drilling",opath=''):
    '''
        Load the Spreadsheet of EADU files and corresponding TDMS files and combine the files together

        The custom spreadsheets has columns of the Stroke files and folder name containing the OWAST TDMS files.
        This function iterates over each row, searches for the EADU file and TDMS folder in the given search path.
        If it finds files, then it loads them into pandas DataFrames, combines them together and saves as a compressed
        GZ files.

        To account for the different lengths, the dataframe is initialized to a rectangular matrix of zeros of the max length
        and each vector populates a portion of it

        e.g. if the max length is 100x100 and one of the signals called test is 10 elements long, then it is updated by
        df["test"].iloc[:10] = test.

        The padding is to avoid NaN values and it is left to the user to handle stretching and different sampling rates.

        WARNING: This program takes a while to run and uses a lot of RAM due to the amount of data involved

        Inputs:
            path : Input path to custom spreadsheet e.g. OWAST_Drilling_Trials\3_Drilling\Copy of Drilling Trials_Experiments_Sheet.xlsx
            search : Path to folder containing EADU and TDMS files. Default OWAST_Drilling_Trials\3_Drilling
            opath : Where to write the GZ files to. Default local.
    '''
    compression_opts = dict(method='gzip',compresslevel=9) 
    data = pd.read_excel(path,header=0,usecols="B:C")
    data.dropna(inplace=True)
    for sf,tf in zip(data['Stroke file'].values,data['TDMS folder'].values):
        sp = os.path.join(search,"*",sf)
        eadu = glob(os.path.join(search,"*",sf))
        if len(eadu)==0:
            print(f"Failed to find file {sp}!")
            continue
        eadu = eadu[0]
        edata = loadSetitecXls(eadu,"auto_data")
        # search for tdms folder
        tp = os.path.join(search,"*",tf,"*.tdms")
        owast = glob(tp)
        if len(owast)==0:
            print(f"Failed to find TDMS folder {tp}!")
            continue
        # for each
        channels = []
        paths = []
        nchannels=0
        maxsz = 0
        for fn in owast:
            for group,channel,data,units in iterTDMSFile(fn,ret_units=True):
                pp = os.path.splitext(os.path.basename(fn))[0]
                maxsz = max(maxsz,len(data))
                channels.append(rf"{pp}\{group}\{channel}")
        nchannels = len(channels)
        # construct dataframe
        df = pd.DataFrame(np.zeros((maxsz,nchannels)),columns=channels)
        # iterate over files again
        for fn in owast:
            pp = os.path.splitext(os.path.basename(fn))[0]
            # iterate through file
            for group,channel,data,units in iterTDMSFile(fn,ret_units=True):
                df[rf"{pp}\{group}\{channel}"].iloc[:data.values.shape[0]] = data.values.flatten()
        # combine EADU dataframe and OWAST dataframe together
        print("combining with EADU data")
        dfcomb = pd.concat([edata,df],axis=1)
        # save compressed
        print(f"writing to {os.path.join(opath,os.path.splitext(os.path.basename(sf))[0])+'.gz'}")
        dfcomb.to_csv(os.path.join(opath,os.path.splitext(os.path.basename(sf))[0])+".gz",index=False,compression=compression_opts,chunksize=100000)

def buildDB(path,**kwargs):
    raise NotImplementedError("Not finished yet!")
    # load file
    data = pd.read_excel(path,sheet_name="Drilling trials",header=0,usecols="A:J")
    dirname = os.path.dirname(path)
    # drop rows containing NaN
    data.dropna(how='any',inplace=True)
    # the column Name of Test contains the filename template for the EADU data
    # get the stroke and time paths
    stroke_paths = []
    time_paths= []
    tdms_paths = []
    for nt in data['Copy names to save EADU data'].values:
        pt = glob(os.path.join(dirname,'*',f"{nt}*.xls"),recursive=True)
        if len(pt)==0:
            print(f"Failed to find Stroke and Time files for {nt}!")
            stroke_paths.append("None")
            time_paths.append("None")
        else:
            stroke_paths.append(pt[0])
            time_paths.append(pt[1])
        # split the hole index, program number and tool
        parts = nt.split('_')
        prog = parts[-1]
        tool = parts[-2]
        hole = parts[3]
        #print(hole,tool,prog)
        # create directory search term for TDMS data
        tdms_dir = f"{hole}_*_{tool}_{prog}*"
        pt = glob(os.path.join(dirname,"*",tdms_dir,"*.*"),recursive=True)
        if len(pt)==0:
            print(f"Failed to find TDMS files for {tdms_dir}")
            tdms_paths.append(list(set([os.path.dirname(p) for p in pt])))
        else:
            tdms_paths.append([])
    # add stroke and time paths    
    data["Stroke Paths"] = stroke_paths
    data["Time Paths"] = time_paths
    data["TDMS Paths"] = tdms_paths
    
        
if __name__ == "__main__":
##    clusters = parseXMLToDict(r"OWAST_EADU_Experiments\Experiments\02-Parameter finding trials\2023.02.17\Fast_to_Slow_A3_B4.xml")
##    for fn in glob(r"OWAST_EADU_Experiments\Experiments\01-Air Trials\*\*\*\*.tdms"):
##        printTDMSInfo(fn)
##        plotTDMSFile(fn)
    #for fn in glob(r"OWAST_EADU_Experiments\Experiments\01-Air Trials\Plug Data\*.csv"):
    #    plotPlugData(fn)
    #exportTDMSToCSV(r"OWAST_Drilling_Trials\3_Drilling\*_OWAST\*\*.tdms")
##    for gn in glob(r"D:\Work\ACSE-EADU_data_analytics\scripts\abyss\src\OWAST_Drilling_Trials\3_Drilling\*_OWAST\*\*.tdms"):
##        print(gn)
##        plotTDMSFile(gn,opath="OWAST")
##        plt.close('all')
##    plt.show()
    #combineEADUAndOWAST(r"OWAST_Drilling_Trials\3_Drilling\Copy of Drilling Trials_Experiments_Sheet.xlsx")
    combineEADUAndOWAST(r"OWAST_Drilling_Trials\3_Drilling\Copy of Drilling Trials_Experiments_Sheet.xlsx")
    
