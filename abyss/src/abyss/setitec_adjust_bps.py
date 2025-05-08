from abyss.dataparser import loadSetitecXls
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ruptures as rpt
from find_opt_smoothing import filterButter, ToolShapelet, distToSamples, raise_sr
from glob import glob

def findMetadata(df):
    '''
        Find the metadata dict in the results of loadSetitecXls and the key

        The found dict is converted to a dataframe.

        This is to handle the different versions of Setitec

        Inputs:
            df : List of stuff returned by loadSetitecXls

        Returns key and DataFrame of step data
    '''
    for dd in df:
        #if any(['Step' in k for k in dd.keys()]):
        for k in dd.keys():
            if 'Step' in k:
                return k,pd.DataFrame.from_dict(dd)

def plotStepEdges(path,use_fig=None):
    '''
        Plot pos-torque signal and add vertical lines for where the program steps end

        Create and return a figure with the position torque data and the dashed lines

        Inputs:
            path : Path to setitec file
            use_fig : Draw on the given figure rather than creating a new one. Default None.

        Returns figure
    '''
    df = loadSetitecXls(path)
    key,metadata = findMetadata(df)
    data = df[-1]
    # filter metadata
    metadata = metadata[metadata[key].isin(data['Step (nb)'].unique())]

    if not use_fig:
        f,ax = plt.subplots()
    else:
        f = use_fig
        ax = f.axes[0]
    pos = np.abs(data['Position (mm)'].values)
    tq = data['I Torque (A)'].values
    ax.plot(pos,tq,'-')
    # find where the limits are
    locs = [np.abs(data[data['Step (nb)']==c]["Position (mm)"]).max() for c in data['Step (nb)'].unique()]
    ax.vlines(locs,data['I Torque (A)'].min(),data['I Torque (A)'].max(),'r',linestyles='dashed',label="Step Changes")
    ax.set(xlabel="Position (mm)",ylabel="Torque (A)",title=f"{os.path.splitext(os.path.basename(path))[0]}")
    return f

def getNumSteps(path):
    '''
        Get the number of programming steps in the file

        The user can give the path or the already loaded Setitec dataframe

        Inputs:
            path : Path to setietc file or loaded dataframe

        Returns number of unique programming setps
    '''
    if isinstance(path,str):
        return len(loadSetitecXls(path)[-1]['Step (nb)'].unique())
    return len(path['Step (nb)'].unique())

def maskProgramStep(df,step,window=4.0):
    '''
        Find and mask where the specified programming step ends and replace the data

        The data is replaced by a fitted line connecting the first and last values in the range

        The window period is specified in mm by window parameter. The final positon of the target program
        step is the centre of the window and half the window size is used either side (i.e. pos +/- window/2).
        That period is replaced.

        The specified step is the step ID.

        Inputs:
            df : Setitec path string or already loaded dataframe
            step : Target step.
            window : Masking period in mm. Default 4.0 mm.

        Returns dataframe with replaced data
    '''
    if step<1:
        raise ValueError("Step code has to be +ve!")
    if isinstance(df,str):
        df = loadSetitecXls(df)[-1]
    if not (step in df['Step (nb)'].unique()):
        raise ValueError("Target step does not exist in the target file!")

    # find where step ends
    pos_min = df[df['Step (nb)']==step]['Position (mm)'].min()
    # mask either side
    mask = df[(df['Position (mm)']>=(pos_min-(window/2))) & (df['Position (mm)']<=(pos_min+(window/2)))].index

    #min_tq = df.iloc[mask]['I Torque (A)'].min()
    min_tq = df.iloc[mask[-1]]['I Torque (A)']
    #max_tq = df.iloc[mask]['I Torque (A)'].max()
    max_tq = df.iloc[mask[0]]['I Torque (A)']
    min_pos = df.iloc[mask]['Position (mm)'].min()
    max_pos = df.iloc[mask]['Position (mm)'].max()

    # fit line between start and end point
    #coefs = np.polyfit([max_pos,min_pos],[min_tq,max_tq],1)
    coefs = np.polyfit([abs(min_pos),abs(max_pos)],[min_tq,max_tq],1)

    # replace data with interpolation
    #pos = np.linspace(max_pos,min_pos,len(mask))
    pos = np.abs(df.iloc[mask]['Position (mm)'])
    tq_rep = np.poly1d(coefs)(pos)
    #df.loc[mask,'Position (mm)']=pos
    df.loc[mask,'I Torque (A)']=tq_rep
    
    return df

def maskProgramStepIndex(df,stepIndex,window=4.0):
    '''
        Same as maskProgramStep but you specify which step by index rather than by ID.

        e.g. if the step ids are 1,2,3 and user specifies 1 when the step ID 2 is replaced.

        Inputs:
            df : Setitec path string or already loaded dataframe
            stepIndex : Target step index.
            window : Masking period in mm. Default 4.0 mm.
    '''
    if isinstance(df,str):
        df = loadSetitecXls(df)[-1]
    return maskProgramStep(df,df['Step (nb)'].unique()[stepIndex],window)

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

        SAME AS MAIN VERSION BUT HAS maskProgramStep TO REMOVE PULLBACK

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
            mask_th : Masking window size to remove backtracking. Default 4.0mm
            plot_grad : Plot the gradient. Default True.
            return_pos : Flag to return breakpoints as position rather than index. Default True.
            bps_th : Min distance threshold for removing neighbouring breakpoints. Default 0.5 mm

        Returns figure and found breakpoints
    '''
    import ruptures as rpt
    ## get parameters
    fr = kwargs.get("feed_rate",0.04)
    rpm = kwargs.get("rpm",4600)
    sr = kwargs.get("sample_rate",100)
    jump = kwargs.get("jump",0.1)
    mask_th = kwargs.get("mask_th",1.0)

    all_bkps = []
    # convert jump to number of samples
    jump_samples = distToSamples(jump,fr,rpm,sr)
    #print("jump (samples)",jump_samples)
    ## find min distance between points in terms of samples
    # get function for determining the best width
    select_min_dist = kwargs.get("select_min_dist",min)
    min_sz_samples = select_min_dist(shapelet.convertSegToSamples(fr,rpm,sr)[0])
    #print("min size (samples)",min_sz_samples)
    # get type of BPS method
    bps_method = kwargs.get("bps_method",rpt.Dynp)
    bps = bps_method(kwargs.get("model","l2"),min_sz_samples,jump_samples)
    # load torque file
    fname = os.path.splitext(os.path.basename(torque_path))[0]
    # load data
    #print("loading file")
    # try loading as a setitec file
    is_setitec = True
    df = loadSetitecXls(torque_path)[-1]
    # filter
    df = maskProgramStepIndex(df,1,mask_th)
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
    #print("butter filter")
    filtered = filterButter(unfiltered,kwargs.get("fc",0.1),kwargs.get("order",10))
    # upsample if the flag is set to True and the new sampling rate does not equal the source
    if kwargs.get("upsample",True) and (sr != kwargs.get("new_sr",100)):
        #print("upsampling")
        # raise sampling rate
        filtered = raise_sr(filtered,sr,new_sr=kwargs.get("new_sr",100),smoothing=0.5)
        dist = raise_sr(dist,sr,new_sr=kwargs.get("new_sr",100),smoothing=0.5)
    ## fit and predict breakpoints
    # use numerical difference
    if kwargs.get("use_diff",True):
        #print("finding BPS using difference")
        diff = np.diff(filtered)
        #diff = filterButter(diff,kwargs.get("fc",0.01),kwargs.get("order",10))
        my_bkps = np.array(bps.fit_predict(diff,n_bkps=kwargs.get("bkps",5)))-1
    else:
        #print("finding BPS using signal")
        my_bkps = np.array(bps.fit_predict(filtered,n_bkps=kwargs.get("bkps",5)))-1

    ## plot the results
    f,ax = plt.subplots()
    # plot the original signal
    ax.plot(dist,filtered,'b-',label="Filtered Signal")
    if kwargs.get("plot_grad",False):
        tax = ax.twinx()
        tax.plot(dist[:-1],diff,'r-')
        tax.set(ylabel="Gradient")
    # plot the breakpoints
    #ax.vlines(dist[my_bkps],unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")
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
            #print("finding CFRP BPS using difference")
            cfrp_diff = np.diff(cfrp)
            #cfrp_diff = filterButter(cfrp_diff,kwargs.get("fc",0.01),kwargs.get("order",10))
            my_bkps = np.array(bps.fit_predict(cfrp_diff,n_bkps=kwargs.get("cfrp_nbkps",2)))-1
        else:
            #print("finding CFRP BPS using signal")
            my_bkps = np.array(bps.fit_predict(cfrp,n_bkps=kwargs.get("cfrp_nbkps",2)))-1

        all_bkps += my_bkps.tolist()
        # final plotting
        #ax.vlines(dist[my_bkps],unf_min,unf_max,colors='g',linestyles='dotted',label="CFRP Breakpoints")
    else:
        print("Skipping finding CFRP breakpoints as file is Lueberring!")
    # sort in ascending order
    all_idx = all_bkps
    all_bkps = np.sort(dist[all_bkps])
    # filter to remove neighbouring breakpoints
    all_bkps = filterBPSDropDist(all_bkps,th=kwargs.get("bps_th",0.2))

    all_bkps = dist[pullbackBP(diff,all_idx)]
    # find the indicies again after correction
    all_new_idx = [np.argmin(np.abs(dist-b)) for b in all_bkps]
    print("bkps grad",diff[all_new_idx])
    all_grad = diff[all_new_idx]
    # draw vertical lines where breakpoiints are
    ax.vlines(all_bkps,unf_min,unf_max,colors='k',linestyles='dashed',label="Breakpoints")
    ax.legend()
    ax.set(xlabel="Distance (mm*100)",ylabel="Torque (mA)",title=f"{fname} min size {min_sz_samples}, jump {jump_samples}\nSR {sr}hz, RPM {rpm}, FR {fr}mm/rev")
    if kwargs.get("return_pos",True):
        return f,all_bkps,all_grad
    else:
        return f,all_idx

def filterBPSDropDist(bps,th=0.1):
    '''
        Filter breakpoints to only those a certain distance from each other

        Due to the method of finding breakpoints, they can often overlap or be very close to each other (within 1 sample).
        However, due to the sampling rate and process parameters a distance of 1 sample can be valid so the breakpoints need to be
        filtered based on physical distance.

        If two breakpoints are within th mm of each other, they are replaced by the mean of the two breakpoints

        Inputs:
            bps : Breakpoints in mm
            th : Min required distance in mm. Default 0.1mm.

        Returns list of filtered breakpoints
    '''
    # check threshold
    if th<=0:
        raise ValueError("BPS distance threshold cannot be <=0!")
    # ensure it's an array
    bps = np.array(bps)
    # make list to hold new bps
    new_bps = []
    # check if any of the breakpoints are within the threshold of each other
    if np.any(np.diff(bps)<th):
        # iterate over breakpoints in pairs
        for a,b in zip(bps[::2],bps[1::2]):
            # if they are within th mm of each other
            if abs(a-b)<th:
                new_bps.append(np.mean([a,b]))
            # add pair to list
            else:
                new_bps.extend([a,b])
        # iteration stops just before last BP
        # add onto the end
        new_bps.append(bps[-1])
    return new_bps

def pullbackBP(tq,bps_idx,th=1e-8,maxits=1000):
    # list of corrected breakpoints
    new_bps = []
    # number of indicies
    nbps = len(tq)
    # iterate over each bp
    for bi in bps_idx:
        its=0
        # store starting index
        start_idx=bi
        # set current index
        current_idx=bi
        # set direction to moving to the left
        direction = -1
        # whilst current index doesn't go out of bounds
        while (current_idx >=0) and (current_idx < (nbps-1)) and (its<maxits):
            # set past location to current
            past_idx = bi
            # move index in direction
            current_idx = bi+direction
            # find different in height
            diff_height = tq[current_idx] - tq[past_idx]
            # if already close to zero
            if diff_height<=th:
                break
            # if the height is increasing
            # change direction
            elif diff_height>0:
                direction *= -1
            its+=1
        if its==maxits:
            print("reached iteration limit! Not changing new bp")
            new_bps.append(start_idx)
        else:
            print(f"setting new breakpoint to {current_idx} vs {start_idx}")
            new_bps.append(current_idx)
        #new_bps.append(start_idx if its==maxits else current_idx)
    return new_bps

# from https://centre-borelli.github.io/ruptures-docs/examples/merging-cost-functions/
class CostL2OnSingleDim(rpt.base.BaseCost):
    """This cost function detects mean-shift on a single dimension of a multivariate signal."""

    # The 2 following attributes must be specified for compatibility.
    model = "CostL2OnSingleDim"
    min_size = 1

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal.reshape(-1, 1)
        return self

    def error(self, start, end) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise rpt.exceptions.NotEnoughPoints
        if end - start == 1:
            return 0.0
        return self.signal[start:end].var(axis=0).sum() * (end - start)

def minmax_scale(array: np.ndarray) -> np.ndarray:
    """Scale each dimension to the [0, 1] range."""
    return (array - np.min(array, axis=0)) / (
        np.max(array, axis=0) - np.min(array, axis=0) + 1e-8
    )

def plotWindowCost(path,WINDOW_SIZE=None,nbkps=8,**kwargs):
    # if the window size is not set
    # calculate them from process parameters
    if WINDOW_SIZE is None:
        # process parameters
        fr = kwargs.get("feed_rate",0.04)
        rpm = kwargs.get("rpm",4600)
        sr = kwargs.get("sample_rate",100)
        # get the shapelet
        shapelet = kwargs.get("shapelet",None)
        if shapelet is None:
            raise ValueError("Tool shapelet has to be specified!")
        # get the min size
        select_min_dist = kwargs.get("select_min_dist",min)
        WINDOW_SIZE = select_min_dist(shapelet.convertSegToSamples(fr,rpm,sr)[0])
    # load file
    df = loadSetitecXls(path)[-1]
    # remove pullback
    df = maskProgramStepIndex(df,1,1.0)
    # get the position and torque data
    pos = np.abs(df['Position (mm)'].values)
    tq = df['I Torque (A)'].values
    tt = df['I Thrust (A)'].values
    # define cost function
    #cost_function = CostL2OnSingleDim(dim=0)
    cost_function = rpt.costs.CostCosine
    cost_function_2 = rpt.costs.CostL2
    # window cost search method
    algo_on_dim_0 = rpt.Window(width=WINDOW_SIZE, custom_cost=cost_function, jump=1).fit(tq)
    algo_on_dim_1 = rpt.Window(width=WINDOW_SIZE, custom_cost=cost_function_2, jump=1).fit(tq)
    # find predicted breakpoints
    bkps_pred = algo_on_dim_0.predict(n_bkps=nbkps - 1)  # the number of changes is known
    # display signal and changes
    fig, axes = rpt.display(tq, bkps_pred)
    _ = axes[0].set_title(
        (
            f"""Detection of mean-shifts using only Dimension 0:\n"""
        )
    )

    fnoise,ax = plt.subplots()
    nplot = np.r_[np.zeros(WINDOW_SIZE // 2),algo_on_dim_0.score,np.zeros(WINDOW_SIZE // 2)]
    print("nplot",nplot.shape)
    ax.plot(pos,nplot,'b-')
    nplot = np.r_[np.zeros(WINDOW_SIZE // 2),algo_on_dim_1.score,np.zeros(WINDOW_SIZE // 2)]
    ax.plot(pos,nplot,'m-')
    tax = ax.twinx()
    tax.plot(pos,tq,'r-')
    ax.set_xmargin(0)
    ax.set_title(f"Score for Costs {cost_function.__name__} & {cost_function_2.__name__} wsize={WINDOW_SIZE}, nbkps={nbkps}")

    # intersection aggregation
    score_arr = np.c_[algo_on_dim_0.score, algo_on_dim_1.score]
    algo_expert_union = rpt.Window(width=WINDOW_SIZE, jump=1).fit(tq)
    algo_expert_union.score = (minmax_scale(score_arr)).min(axis=1)  # scaling + pointwise min
    # only one change point is shared by both dimensions
    bkps_intersection_predicted = algo_expert_union.predict(n_bkps=1)
    # display the intersected score
    fig, ax = plt.subplots()
##    ax.plot(
##        np.r_[
##            np.zeros(WINDOW_SIZE // 2),
##            algo_expert_intersection.score,
##            np.zeros(WINDOW_SIZE // 2),
##        ]
##    )

    ax.plot(
        np.r_[
            np.zeros(WINDOW_SIZE // 2), algo_expert_union.score, np.zeros(WINDOW_SIZE // 2)
        ]
    )
    
    ax.set_xmargin(0)
    _ = ax.set_title(
        (
            """Aggregated score (intersection of experts)\n"""
        )
    )
    return fig

if __name__ == "__main__":
    ts = ToolShapelet([1,2,3],[1,2,3])
    ns = []
    torqued_AB = []
    
    for fn in [glob(r"C:\Users\david\Downloads\MSN660-20230912T134035Z-001\MSN660\*.xls")[22],]:
##        f = plotWindowCost(fn,None,10,shapelet=ts)
##        break
##        df = maskProgramStep(fn,1)
##        f,ax = plt.subplots()
##        ax.plot(df['Position (mm)'],df['I Torque (A)'],'r-',label="Replace")
##        df = loadSetitecXls(fn)[-1]
##        ax.plot(df['Position (mm)'],df['I Torque (A)'],'b-',label="Original")
##        ax.legend()
##        break
        f,bps = findPlotBpsCalcMinSize(fn,ts,return_pos=False,plot_grad=True)
        print("nbps",len(bps))
        print(bps)
        f.savefig(f"breakpoints-with-masks/{os.path.splitext(os.path.basename(fn))[0]}-bps-with-fill.png")
        df = loadSetitecXls(fn)[-1]
        torqued_AB.append(abs(df['I Torque (A)'].values[bps[2]-1]-df['I Torque (A)'].values[bps[2]]))
        #plt.close(f)
        break
##
##        df = loadSetitecXls(fn)[-1]
##        bps = np.sort(np.abs(df['Position (mm)'][bps].values))
##        print("filtering BPS")
##        new_bps = filterBPSDropDist(bps)
##        print(bps,len(bps),"vs",new_bps,len(new_bps))
##
##        #f = plotStepEdges(fn,f)
 #       plt.gca().vlines(new_bps,df['I Torque (A)'].min(),df['I Torque (A)'].max(),'m',linestyle='dashed',label="Filtered BPS")
 #       plt.gca().legend()
 #       f.savefig(f"breakpoints-with-masks/{os.path.splitext(os.path.basename(fn))[0]}-bps-with-fill.png")
        #plotStepEdges(fn,f)
        #ns.append(getNumSteps(fn))
    #plt.close('all')
    plt.show()
    
