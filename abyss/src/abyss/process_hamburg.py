import abyss.dataparser as dp
import numpy as np
import os
from glob import glob
from plotting import plotSetitecHistory
import matplotlib.pyplot as plt
from abyss.modelling import depth_est_rolling

# get holenumber
def isAir(fn):
    fname = os.path.splitext(os.path.basename(fn))[0]
    # global and local hole numbers are in the first and last parts of the filename
    pts = fname.split('_')
    # if it has three numerical last parts then it's an all AIR file and can be skipped
    return all([c.isnumeric() for c in pts[-3:]])

def isNotAir(fn):
    return not isAir(fn)

def findBadFilesInDir(path):
    skipped = []
    for fn in sorted(glob(os.path.join(path,'*.xls')),key=lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])):
        # load file
        try:
            data = loadSetitecXls(fn,version='auto')
        except Exception as e:
            skipped.append((fn,str(e)))
    return skipped

def findBadFilesMP(path):
   import multiprocessing as mp
   uqdirs = set([os.path.dirname(fn) for fn in glob(path,recursive=True)])
   skipped = mp.Pool(2).map(findBadFilesInDir,uqdirs)
   sk = []
   return map(skipped.extend,sk)

def getHead(path):
    try:
        head= dp.getSetitecXLSLocalHead(path)
    except:
        return -1
    return -1 if head is None else head

def groupFilesByHead(path,mode="name"):
    heads = {}
    if isinstance(path,str):
        paths = glob(path)
    else:
        paths = path
    print(f"original {len(paths)}")
    for fn in paths:
        tag = dp.getSetitecXLSHeadName(fn) if mode == "name" else dp.getSetitecXLSHeadTag(fn)
        if tag is None:
            tag = -1
        heads[tag] = heads.get(tag,[])
        heads[tag].append(fn)
    # sort by local head count
    # remove air files
    # remove files missing local head count
    for k in heads.keys():
        heads[k] = sorted(filter(lambda x : x,list(filter(isNotAir,heads[k]))),key=getHead)
    return heads
    
def depthEstimateHead(head_dict,N=30,depth_exp=20.0,depth_win=8.0,use_empty=False):
    depth_cum = {k : [] for k in head_dict.keys()}
    for k,v in head_dict.items():
        for fn in v:
            data = dp.loadSetitecXls(fn,"auto_data")
            xdata = np.abs(data['Position (mm)'].values)
            ydata = data['I Torque (A)'].values
            if use_empty:
                ydata += data['I Torque Empty (A)'].values
            depth_cum[k].append(depth_est_rolling(ydata,xdata,NA=N,depth_exp=depth_exp,depth_win=depth_win))
    return depth_cum

def plotCumulativeDepth(path,N=30,depth_exp=20.0,depth_win=5.0,use_empty=False):
    heads = groupFilesByHead(path)
    for k,v in heads.items():
        print(f"{k} {len(v)}")
    print(f"Filtered {sum([len(v) for v in heads.values()])}")
    dum = depthEstimateHead(heads,N,depth_exp,depth_win,use_empty)
    for k,v in dum.items():
        if len(v)>1:
            f,ax = plt.subplots(constrained_layout=True)
            ax.plot(v,'ro-')
            ss = np.cumsum(v)
            tax = ax.twinx()
            tax.plot(ss)
            ax.set(xlabel="Local Head Number",ylabel="Estimated Depth (mm)")
            tax.set_ylabel("Cumulative Drilled Depth (mm)")
            f.savefig(f"hamburg-{k}-cummulative-drilled.png")
            plt.close(f)

if __name__ == "__main__":
    path = r'\\fr0-vsiaas-5706\EDU_Pipe\EDU_Incoming_Backup\Hamburg\SETITEC\*\*.xls'
    #plotSetitecHistory(path)
    plotCumulativeDepth(path)
    plt.show()
			