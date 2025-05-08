import numpy as np
import pandas as pd
from find_opt_smoothing import findPlotBpsCalcMinSize, loadDrillingFile, getTorquePosData, ToolShapelet
from glob import glob
from itertools import permutations
import matplotlib.pyplot as plt

def assembleStacks(path:str,ts=None,**kwargs):
    '''
        
    '''
    paths = []
    cfrp = []
    mat = []
    cfrp_tq = []
    mat_tq = []
    ids = []
    uqids = []
    if ts is None:
        ts = ToolShapelet([1,2,3],[1,2,3])
    for fn in glob(path):
        f,bps = findPlotBpsCalcMinSize(fn,ts,return_data=False,return_samples=True,**kwargs)
        # ensure unique breakpoints
        bps = sorted(list(set(bps)))
        #print("bps",bps)
        plt.close(f)
        # load data
        df = loadDrillingFile(fn)[0]
        # first three bps are for CFRP
        # so we can separate CFRP portion of signal
        #cfrp_part = tuple(tuple(df['Position (mm)'][:bps[2]].tolist()),)
        cfrp_part = df['Position (mm)'][:bps[2]].tolist()
        # the remaining part is the 2nd mat
        #mat_part = tuple(tuple(df['Position (mm)'][bps[2]:].tolist()),)
        mat_part = df['Position (mm)'][bps[2]:].tolist()
        #print("mat cfrp part",len(cfrp_part),len(mat_part))
        # add to lists
        cfrp.append(cfrp_part)
        cfrp_tq.append(df['I Torque (A)'][:bps[2]].tolist())
        # append id of 0 for CFRP
        ids.append(0)
        paths.append(fn)
        # hash has to be performed on an immutable object
        uqids.append(hash(tuple(cfrp_part)))
        mat.append(mat_part)
        mat_tq.append(df['I Torque (A)'][bps[2]:].tolist())
        #print("torque part",len(cfrp_tq[-1]),len(mat_tq[-1]))
        #input()
        # append id of 1 for 2nd mat
        ids.append(1)
        paths.append(fn)
        uqids.append(hash(tuple(mat_part)))
    # combine parts together
    mats_all = cfrp + mat
    tq_all = cfrp_tq + mat_tq
    #print(len(paths),len(mats_all),len(ids),len(uqids))
    # form into dataframe
    return pd.DataFrame(list(zip(paths,mats_all,tq_all,uqids,ids)),columns=["Path","Position (mm)","Torque (A)","Unique ID","Portion ID"])

def makeCombinations(mat_parts):
    # get pairwise combinations of unique ids acting as combinations of material sections
    uq_combos = list(permutations(mat_parts["Unique ID"].unique(),2))
    # iterate over combinations
    # assembling into a dataframe
    combo_paths = []
    combined_ids = []
    pos_stacks = []
    tq_stacks = []
    for uu in zip(uq_combos):
        ua,ub = uu[0]
        # filter to target ids
        filt = mat_parts[mat_parts["Unique ID"].isin((ua,ub))]
        # store which paths the combos are from
        combo_paths.append(tuple(filt["Path"].unique()))
        # store which IDs
        combined_ids.append((ua,ub))
        # combine stacks together via appending
        # no smoothing
        pos_stacks.append(filt.iloc[0]["Position (mm)"]+filt.iloc[1]["Position (mm)"])
        tq_stacks.append(filt.iloc[0]["Torque (A)"]+filt.iloc[1]["Torque (A)"])
        #print(len(pos_stacks[-1]),len(tq_stacks[-1]))
    mat_combos = pd.DataFrame(list(zip(pos_stacks,tq_stacks,combo_paths,combined_ids)),columns=["Position (mm)","Torque (A)","Combined Paths","Combined IDs"])
    return mat_combos

def plotCombos(mat_combos,sort_pos=False):
    f,ax = plt.subplots()
    for i in range(mat_combos.shape[0]):
        row = mat_combos.iloc[i]
        pos = np.abs(row["Position (mm)"])
        tq = np.array(row["Torque (A)"])
        if sort_pos:
            ii = np.argsort(pos)
            pos = pos[ii]
            tq = tq[ii]
        #print(len(row["Position (mm)"]),len(row["Torque (A)"]))
        ax.plot(pos,tq,'-')
    return f

if __name__ == "__main__":
    df = assembleStacks(r"C:\Users\david\Downloads\MSN660-20230912T134035Z-001\MSN660\*.xls")
    combos = makeCombinations(df)
    plotCombos(combos)
    
