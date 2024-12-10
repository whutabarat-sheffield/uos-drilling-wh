import numpy as np
import math
from tslearn.metrics import cdist_dtw, dtw, dtw_path
from dataparser import loadSetitecXls
from glob import glob
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib import cm
import os

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
        self._av = None
        self._scale = 1.0
        # data vectors representing tool
        self._pos = np.empty((0,))
        self._time = np.empty((0,))
        self._signal = np.empty((0,))

    def tlength(self):
        ''' Return the tool length by summing the tool lengths together '''
        return math.fsum(self._dims)

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
            

def plotDTWCostPath(s_y1,s_y2,path,sim=None):
    # adapted from https://tslearn.readthedocs.io/en/stable/auto_examples/metrics/plot_dtw.html#sphx-glr-auto-examples-metrics-plot-dtw-py
    f = plt.figure(figsize=(8, 8))
    # definitions for the axes
    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    
    # scale the width of the axes depending on which one is larger
    if s_y1.shape[0] > s_y2.shape[0]:
        height = 0.65
        width = height * min([s_y1.shape[0],s_y2.shape[0]])/max([s_y1.shape[0],s_y2.shape[0]])
    else:
        width = 0.65
        height = width * min([s_y1.shape[0],s_y2.shape[0]])/max([s_y1.shape[0],s_y2.shape[0]])
  
    bottom_h = bottom + height + 0.02
    # rectangles defining axes around the edge of the plot to draw the signals
    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]
    # create axes
    ax_gram = f.add_axes(rect_gram)
    ax_s_x = f.add_axes(rect_s_x)
    ax_s_y = f.add_axes(rect_s_y)
    # calculate distances between the two signals
    if sim is None:
        sim = cdist(s_y1.reshape((-1,1)), s_y2.reshape((-1,1)))
    # plot axes
    ax_gram.imshow(sim, origin='lower')
    ax_gram.axis("off")
    ax_gram.autoscale(False)
    # plot a line showing the path
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-",
                 linewidth=3.)
    # plot the second signal horizontally across the top
    ax_s_x.plot(np.arange(s_y2.shape[0]), s_y2, "b-", linewidth=3.)
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, s_y2.shape[0] - 1))
    # plot first signal verttically
    ax_s_y.plot(- s_y1, np.arange(s_y1.shape[0]), "b-", linewidth=3.)
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, s_y1.shape[0] - 1))
    return f

def plotDWTConnectionPath(s_y1,s_y2,path,score=None,cmap="hot",offset=False,**kwargs):
    from matplotlib.patches import ConnectionPatch
    if score is None:
        score = cdist(min([s_y1,s_y2],key=lambda x:x.shape[0]).reshape((-1,1)), max([s_y1,s_y2],key=lambda x:x.shape[0]).reshape((-1,1)))
    smax = score.max()
    cmap = cm.get_cmap(cmap)
    print(score.shape)
    print(max(path,key=lambda x : x[0]),max(path,key=lambda x : x[1]))
    # create figure
    fig = plt.figure(figsize=(9,10))
    # create two axes with an empty axes inbetween as a buffer
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(313)
    # plot the two signals
    if offset:
        ii = np.arange(0,s_y1.shape[0],1)+((s_y2.shape[0]//2)-(s_y1.shape[0]//2))
        ax1.set_xlim(0,s_y2.shape[0])
    else:
        ii = np.arange(0,s_y1.shape[0],1)
    # iterate over the path coordinates
    for x_i, y_j in path:
        con = ConnectionPatch(
            xyA=(ii[x_i],s_y1[x_i] ), xyB=(y_j, s_y2[y_j]), coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2, color=cmap(score[x_i,y_j]/smax))
        ax2.add_artist(con)
    ax1.plot(ii,s_y1,'k-',zorder=len(path)+1)
    ax1.set(xlabel=kwargs.get("s1_xlabel",""),ylabel=kwargs.get("s1_ylabel",""),title=kwargs.get("s1_title",""))
    ax2.plot(s_y2,'r-')
    ax2.set(xlabel=kwargs.get("s2_xlabel",""),ylabel=kwargs.get("s2_ylabel",""),title=kwargs.get("s2_title",""))
    fig.suptitle(kwargs.get("ftitle","DTW Tool Shapelet vs Signal"))
    return fig

def findShapletBrute(fn,tool_dims,av,key="torque",calc_score=False,**kwargs):
    # load file
    print("loading data")
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
    print("creating shapelet")
    _,spos,shapelet = ToolShapelet(tool_dims).generate(av,(search.max()-search.min())/2)
    shapelet += search[0]
    print("search ",search.shape," shapelet ",shapelet.shape)
    #
    f,ax = plt.subplots()
    ax.plot(shapelet)
    ax.set(xlabel="Sample",ylabel=key,title=kwargs.get("title","Tool Model"))
    path,_ = dtw_path(shapelet,search)
    score = None
    if calc_score:
        score = cdist_dtw(shapelet,search,n_jobs=10,verbose=1)
    print("path: ",len(path))
    print("score ",score.shape if score else None)
    plotDTWCostPath(shapelet,search,path,score)
    plotDWTConnectionPath(shapelet,search,path)

def compareShapeletSamples(fn,tool_dims,av,key="torque",ratio=0.2,mirror=False,plot_res=False,**kwargs):
    # load file
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
    if scale == 'max':
        scale = search.max() - search[0]
    elif scale == 'min':
        scale = search.min()
    elif (scale == 'mid') or (scale is None):
        scale = (search.max()-search.min())/2

    _,spos,shapelet = tool.generate(av,scale,mirror=mirror)
    shapelet += search[0]

    # convert to a dataframe
    search = pd.DataFrame.from_dict({'pos':np.abs(data['Position (mm)'].values.flatten()),'signal':search})

    if plot_res:
        f,ax = plt.subplots(ncols=2,constrained_layout=True)
        ax[0].plot(spos,shapelet)
        ax[0].set(xlabel="Position (mm)",ylabel=key,title=kwargs.get("stitle","Tool Shapelet"))
        ax[1].plot(np.abs(search['pos'].values.flatten()),search['signal'],'b-')
        tax = ax[1].twinx()

    def _score(x):
        x.dropna()
        x = x['signal']
        return dtw(shapelet,x)
    
    if isinstance(ratio,float):
        ratio = [ratio,]
    for rr in ratio:
        score = search.rolling(int(rr*shapelet.shape[0])).apply(_score).values.flatten()
        np.nan_to_num(score,False)
        if len(ratio)==1:
            tax.plot(ii,score,'r-',label=f"N={rr:.2f}")
        else:
            tax.plot(ii,score,label=f"N={rr:.2f}")
    if plot_res:
        tax.legend()
        #tax.set_ylim(0,score.max())
        ax[1].set(xlabel="Position (mm)",ylabel=key,title="Score")
        tax.set_ylabel("DTW Score")
        rt_str = '-'.join([f"{pp:.2f}" for pp in ratio])    
        f.suptitle(kwargs.get("title",f"DTW Pos Score av={av},tl={math.fsum(tool_dims)},per={rt_str},{'mirrored' if mirror else ''}"))
        f.savefig(f"dtw_tool/{os.path.splitext(os.path.basename(fn))[0]}-av-{av}-tl-{rt_str}{'-mirror' if mirror else ''}-sp-{kwargs.get('shapelet','unknown')}-rolling-samples.png")
    if len(per)==1:
        return pos[0],score[0]

def compareShapeletPos(fn,tool_dims,av,key="torque",per=1.0,mirror=False,plot_res=False,**kwargs):
    # load file
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
    if scale == 'max':
        scale = search.max() - search[0]
    elif scale == 'min':
        scale = search.min()
    elif (scale == 'mid') or (scale is None):
        scale = (search.max()-search.min())/2

    _,spos,shapelet = tool.generate(av,scale,mirror=mirror)
    shapelet += search[0]

    # convert to a dataframe
    search = pd.DataFrame.from_dict({'pos':np.abs(data['Position (mm)'].values.flatten()),'signal':search})

    if plot_res:
        f,ax = plt.subplots(ncols=2,constrained_layout=True)
        ax[0].plot(spos,shapelet)
        ax[0].set(xlabel="Position (mm)",ylabel=key,title=kwargs.get("stitle","Tool Shapelet"))
        ax[1].plot(np.abs(search['pos'].values.flatten()),search['signal'],'b-')
        tax = ax[1].twinx()

    def _score(x):
        x.dropna()
        x = x['signal']
        return dtw(shapelet,x)
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
        f.suptitle(kwargs.get("title",f"DTW Pos Score av={av},tl={math.fsum(tool_dims)},per={per_str},{'mirrored' if mirror else ''}"))
        f.savefig(f"dtw_tool/{os.path.splitext(os.path.basename(fn))[0]}-av-{av}-tl-{per_str}{'-mirror' if mirror else ''}-sp-{kwargs.get('shapelet','unknown')}.png")
    if len(per)==1:
        return pos[0],score[0]

def compareShapeletPosTryPeriods(fn,tool_dims,av,key="torque",per=1.0,mirror=False,**kwargs):
    # load file
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
    if scale == 'max':
        scale = search.max() - search[0]
    elif scale == 'min':
        scale = search.min()
    elif (scale == 'mid') or (scale is None):
        scale = (search.max()-search.min())/2

    _,spos,shapelet = tool.generate(av,scale,mirror=mirror)
    shapelet += search[0]

    # convert to a dataframe
    search = pd.DataFrame.from_dict({'pos':np.abs(data['Position (mm)'].values.flatten()),'signal':search})

    def _score(x):
        x.dropna()
        x = x['signal']
        return dtw(shapelet,x)
    scores = []
    posits = []
    pos_max = []
    pos_min = []
##    for pp in per:
##        # group values into groups of size per mm
##        gps = search.groupby(search['pos'].apply(lambda x : pp*round(x/pp)))
##        pos = np.abs(list(gps.indices.keys()))+pp/2
##        score = gps.apply(_score).values.flatten()
##        scores.append(score)
##        posits.append(pos)
##        pos_max.append(pos[score.argmax()])
##        pos_min.append(pos[score.argmin()])

    def _find(pp):
        # group values into groups of size per mm
        gps = search.groupby(search['pos'].apply(lambda x : pp*round(x/pp)))
        pos = np.abs(list(gps.indices.keys()))+pp/2
        score = gps.apply(_score).values.flatten()
        scores.append(score)
        posits.append(pos)
        return pos[score.argmax()],pos[score.argmin()]
    pos = list(map(_find,per))
    pos_max = [x[0] for x in pos]
    pos_min = [x[1] for x in pos]

    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    ax[0].plot(spos,shapelet)
    ax[0].set(xlabel="Position (mm)",ylabel=key,title=kwargs.get("stitle","Tool Shapelet"))
    ax[1].plot(per,pos_max,'b')
    ax[1].set(xlabel="Period (mm)",ylabel="Max Score Position (mm)",title="Max Score Position")
    ax[2].plot(per,pos_min,'r')
    ax[2].set(xlabel="Period (mm)",ylabel="Min Score Position (mm)",title="Min Score Position")
    f.suptitle(f"{os.path.splitext(os.path.basename(fn))[0]},av={av}")
    return f

def compareShapeletPosSlider(fn,tool_dims,av,key="torque",per=1.0,mirror=False,**kwargs):
    from matplotlib.widgets import Slider
    # load file
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
    if scale == 'max':
        scale = search.max() - search[0]
    elif scale == 'min':
        scale = search.min()
    elif (scale == 'mid') or (scale is None):
        scale = (search.max()-search.min())/2

    _,spos,shapelet = tool.generate(av,scale,mirror=mirror)
    shapelet += search[0]

    # convert to a dataframe
    search = pd.DataFrame.from_dict({'pos':np.abs(data['Position (mm)'].values.flatten()),'signal':search})

    f,ax = plt.subplots(ncols=2,constrained_layout=True)
    ax[0].plot(spos,shapelet)
    ax[0].set(xlabel="Position (mm)",ylabel=key,title=f"Tool Shapelet, tl={tool.tlength():.2f}")
    ax[1].plot(np.abs(search['pos'].values.flatten()),search['signal'],'b-')
    f.suptitle(os.path.splitext(os.path.basename(fn))[0])
    tax = ax[1].twinx()

    maxline = ax[1].axvline(0,color='k')
    minline = ax[1].axvline(0,color='k')

    f.subplots_adjust(left=0.25, bottom=0.25)
    # Make a horizontal slider to control the frequency.
    axper = f.add_axes([0.25, 0.1, 0.65, 0.03])
    per_slider = Slider(
        ax=axper,
        label='Period (mm)',
        valmin=0.1,
        valmax=search['pos'].values.max(),
        valinit=tool.tlength()/2
    )

    def _score(x):
        x.dropna()
        x = x['signal']
        return dtw(shapelet,x)
    
    def update(val):
        per = per_slider.val
        # group values into groups of size per mm
        gps = search.groupby(search['pos'].apply(lambda x : per*round(x/per)))
        pos = np.abs(list(gps.indices.keys()))+per/2
        score = gps.apply(_score).values.flatten()
        ax[1].set_xlim(pos.min(),pos.max())
        if tax.lines:
            tax.lines.pop(0)
        tax.plot(pos,score,'r-')
        maxline.set_xdata([pos[score.argmax()],pos[score.argmax()]])
        minline.set_xdata([pos[score.argmin()],pos[score.argmin()]])
        ax[1].set_title(f"per={per:.2f} mm")
        f.canvas.draw_idle()
   
    ax[1].set(xlabel="Position (mm)",ylabel=key)
    #tax.set_ylabel("DTW Score")
    update(0)
    per_slider.on_changed(update)
    plt.show()

def convolveShapelet(fn,tool_dims,av,key="torque"):
    # load file
    print("loading data")
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
    print("creating shapelet")
    _,spos,shapelet = ToolShapelet(tool_dims).generate(av,1.0)
    shapelet += search.min()
    # convolve shapelet with signal
    res = np.convolve(search,shapelet,mode='same')
    f,ax = plt.subplots()
    ax.plot(res)

def plotCompareShapeletPos(path,dims,av,per=None,**kwargs):
    if per is None:
        per = math.fsum(dims)/2
    # vectors of results
    smax_pos = []
    smin_pos = []
    smax = []
    smin = []
    maxt = []
    mint = []
    fsig,axsig = plt.subplots()
    
    for fn in glob(path):
        # compare according to shaplet at target tool res
        pos,ss = compareShapeletPos(fn,dims,av,per=per,**kwargs)
        plt.close('all')
        # find where the score is max
        ii = np.argmax(ss)
        smax_pos.append(pos[ii])
        smax.append(ss[ii])
        # find where the score is min
        ii = np.argmin(ss)
        smin_pos.append(pos[ii])
        smin.append(ss[ii])
        # get max thrust and torque
        data = loadSetitecXls(fn,"auto_data")
        signal = data[f"I {kwargs.get('key','torque').capitalize()} (A)"].values.flatten()
        if "I {kwargs.get('key','torque').capitalize()} Empty (A)" in data:
            signal += data["I {kwargs.get('key','torque').capitalize()} Empty (A)"].values.flatten()
        maxt.append(signal.max())
        mint.append(signal.min())
    
    # plot max score position per hole
    figs = []
    ww,hh = plt.rcParams['figure.figsize']
    f,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True,figsize=(ww*2,hh*2))
    ax[0,0].plot(smax_pos,'bx')
    ax[0,0].set(xlabel="Hole Number",ylabel="Max Score Position (mm)",title="Hole Number vs Max Score Position")

    # plot min score position per hole
    ax[0,1].plot(smin_pos,'rx')
    ax[0,1].set(xlabel="Hole Number",ylabel="Min Score Position (mm)",title="Hole Number vs Min Score Position")

    # plot the score per hole
    ax[1,0].plot(smax,'bx')
    ax[1,0].set(xlabel="Hole Number",ylabel="Max Score",title="Hole Number vs Max Score")

    ax[1,1].plot(smin,'rx')
    ax[1,1].set(xlabel="Hole Number",ylabel="Min Score",title="Hole Number vs Min Score")
    f.suptitle(f"{kwargs.get('tool_name','Tool')} Scores Check using {kwargs.get('key','torque').capitalize()} per={per:.2f}")
    figs.append(f)

    # plot max torque against score position
    f,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True,figsize=(ww*2,hh*2))
    ax[0,0].plot(maxt,smax_pos,'bx')
    ax[0,0].set(xlabel=f"Max {kwargs.get('key','torque').capitalize()} (A)",ylabel="Max Score Position (mm)",title=f"Max {kwargs.get('key','torque').capitalize()} vs Max Score Pos")

    ax[0,1].plot(maxt,smin_pos,'rx')
    ax[0,1].set(xlabel=f"Max {kwargs.get('key','torque').capitalize()} (A)",ylabel="Min Score Position (mm)",title=f"Max {kwargs.get('key','torque').capitalize()} vs Min Score Pos")

    ax[1,0].plot(mint,smax_pos,'bx')
    ax[1,0].set(xlabel=f"Min {kwargs.get('key','torque').capitalize()} (A)",ylabel="Max Score Position (mm)",title=f"Min {kwargs.get('key','torque').capitalize()} vs Max Score Pos")

    ax[1,1].plot(mint,smin_pos,'rx')
    ax[1,1].set(xlabel=f"Min {kwargs.get('key','torque').capitalize()} (A)",ylabel="Min Score Position (mm)",title=f"Min {kwargs.get('key','torque').capitalize()} vs Min Score Pos")
    f.suptitle(f"{kwargs.get('tool_name','Tool')} Scores Check using {kwargs.get('key','torque').capitalize()} per={per:.2f}")
    figs.append(f)

    # plot max torque against score
    f,ax = plt.subplots(nrows=2,ncols=2,constrained_layout=True,figsize=(ww*2,hh*2))
    ax[0,0].plot(maxt,smax,'bx')
    ax[0,0].set(xlabel=f"Max {kwargs.get('key','torque').capitalize()} (A)",ylabel="Max Score",title=f"Max {kwargs.get('key','torque').capitalize()} vs Max Score")

    ax[0,1].plot(maxt,smin,'rx')
    ax[0,1].set(xlabel=f"Max {kwargs.get('key','torque').capitalize()} (A)",ylabel="Min Score",title=f"Max {kwargs.get('key','torque').capitalize()} vs Min Score")

    ax[1,0].plot(mint,smax,'bx')
    ax[1,0].set(xlabel=f"Min {kwargs.get('key','torque').capitalize()} (A)",ylabel="Max Score",title=f"Min {kwargs.get('key','torque').capitalize()} vs Max Score")

    ax[1,1].plot(mint,smin,'rx')
    ax[1,1].set(xlabel=f"Min {kwargs.get('key','torque').capitalize()} (A)",ylabel="Min Score",title=f"Min {kwargs.get('key','torque').capitalize()} vs Min Score")
    f.suptitle(f"{kwargs.get('tool_name','Tool')} Scores Check using {kwargs.get('key','torque').capitalize()} per={per:.2f}")
    figs.append(f)
    return figs

def tryOptimizeAV(path,mats,tool_dims,mirror=False,scale='mid',key='torque'):
    from scipy.optimize import minimize
    # function for calculating score
    def _score(x,shapelet):
        x.dropna()
        x = x['signal']
        return dtw(shapelet,x)

    data = loadSetitecXls(path,'auto_data')
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
    if scale == 'max':
        scale = search.max() - search[0]
    elif scale == 'min':
        scale = search.min()
    elif (scale == 'mid') or (scale is None):
        scale = (search.max()-search.min())/2

    # convert to a dataframe
    search = pd.DataFrame.from_dict({'pos':np.abs(data['Position (mm)'].values.flatten()),'signal':search})
    mat_thick = math.fsum(mats)
    def estDepth(x,*args):
        per,av = x
        _,spos,shapelet = tool.generate(av,scale,mirror=mirror)
        shapelet += search['signal'][0]
        # group values into groups of size per mm
        gps = search.groupby(search['pos'].apply(lambda x : per*round(x/per)))
        pos = np.abs(list(gps.indices.keys()))+per/2
        score = gps.apply(lambda x : _score(x,shapelet)).values.flatten()
        # find min and max
        ii = np.argmin(score)
        min_pos = pos[ii]
        ii = np.argmax(score)
        max_pos = pos[ii]
        return mat_thick-(max_pos-min_pos)
    # tolerance is set to 0.1mm as we don't need to get much closer
    res = minimize(estDepth,x0=(0.1,0.1),bounds=[(0.1,tool.tlength()),(0.1,5.0)],tol=0.1,method='Nelder-Mead')
    print(res)
    return res.x

def gridSearchPerScale(path,mats,tool_dims,av=3.0,npoints=20,mirror=False,key='torque',plot_res=False):
    # function for calculating score
    def _score(x,shapelet):
        x.dropna()
        x = x['signal']
        return dtw(shapelet,x)

    data = loadSetitecXls(path,'auto_data')
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

    # convert to a dataframe
    search = pd.DataFrame.from_dict({'pos':np.abs(data['Position (mm)'].values.flatten()),'signal':search})

    mat_thick = math.fsum(mats)

    scale_try = np.linspace(search['signal'].min(),search['signal'].max(),npoints)
    per_try = np.linspace(0.1,tool.tlength(),npoints)
    # get first value of signal to offset scaling
    fsg =search['signal'][0]
    def proc(scale,per):
        shapelet = tool.generate(av,scale,mirror=mirror)[-1]+fsg
        # group values into groups of size per mm
        gps = search.groupby(search['pos'].apply(lambda x : per*round(x/per)))
        pos = np.abs(list(gps.indices.keys()))+per/2
        score = gps.apply(lambda x : _score(x,shapelet)).values.flatten()
        # find min and max
        return abs(mat_thick-(pos[np.argmax(score)]-pos[np.argmin(score)]))
    # double map for speed
    res = np.array(list(map(lambda s : list(map(lambda p : proc(s,p),per_try)),scale_try)))
    r,c = np.unravel_index(np.argmin(res),res.shape)
    if not plot_res:
        print(f"finished {path}!",flush=True)
        return scale_try[r],per_try[c]
    f,ax = plt.subplots(constrained_layout=True)
    surf = ax.imshow(res,cmap='hot',interpolation='none')
    plt.colorbar(surf)
    nt = len(ax.get_xticklabels())
    ax.set_xticklabels([f"{x:.2f}" for x in per_try[::int(npoints/nt)]])
    ax.set_yticklabels([f"{x:.2f}" for x in scale_try[::int(npoints/nt)]])
    
    ax.set(xlabel="Period (mm)",ylabel="Scale (A)",title=f"{os.path.splitext(os.path.basename(path))[0]}")
    return f,scale_try[r],per_try[c]

### 4B tool lengths ###
def tool_4B():
    ts1 = np.tan(np.radians(67.5))*(6.324-5.79)
    dims = [1.2,3,ts1,8-3-ts1]
    av=3.0
    return dims
### 8B tool lengths ###
def tool_8B():
    ts1 = np.tan(np.radians(67.5))*(12.68-11.57)
    dims=[np.tan(np.radians(22.5))*(11.57/2),3.0,ts1,8-3-ts1]
    av=3.0
    return dims

import multiprocessing as mp
def proc(f):
    return gridSearchPerScale(f,[26,12],tool_8B(),plot_res=False)

def dtwAlign(path,ref=0,key='torque',plot_res=False):
    paths = glob(path)
    ref_path = paths.pop(ref)
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
        ref = [x[1] for x in dtp]
        if plot_res:
            ax.plot(pos[ref],tq[ref])
        stack.append((pos[ref],tq[ref]))
    if plot_res:
        ax.legend()
    return stack,ref

if __name__ == "__main__":
    #findShapletBrute("4B life test/E00401009F456DD3_18080019_ST_5003_3334.xls",dims,av)
    #convolveShapelet("4B life test/E00401009F456DD3_18080019_ST_5003_3334.xls",dims,av)
    #compareShapeletRolling("4B life test/E00401009F456DD3_18080019_ST_5003_3334.xls",dims,av,ratio=np.arange(0.1,1.1,0.1))
    #compareShapeletRollingPos("4B life test/E00401009F456DD3_18080019_ST_5003_3334.xls",dims,av,per=math.fsum(dims)/2)
##
##    k='torque'
##    m = True
##    for _ in range(2):
##        for tl,tn in zip([tool_4B,tool_8B],['8B using 4B shapelet','8B using 8B shapelet']):
##            dims = tl()
##            for av in [1.0,3.0]:
##                figs = plotCompareShapeletPos("8B life test/*.xls",dims,av,tool_name=tn,key=k,mirror=m,plot_res=True,shapelet=tn.replace(' ','-').lower(),scale='max' if m else 'mid')
##                for f,fname in zip(figs,[f"8B-{tn.replace(' ','-').lower()}-av-{av}-half-tool-{'mirror-' if m else ''}{k}-scores-vs-hole-number",f"8B-{tn.replace(' ','-').lower()}-av-{av}-half-tool-{'mirror-' if m else ''}{k}-vs-score-position",f"8B-{tn.replace(' ','-').lower()}-av-{av}-half-tool-{'mirror-' if m else ''}{k}-scores-vs-{k}"]):
##                    f.savefig(f"dtw_tool/{fname}.png")
##        m= not m
    tn = '8B using 8B shapelet'
    #compareShapeletPosSlider(glob("8B life test/*.xls")[0],tool_8B(),3.0,tool_name=tn,key='torque',mirror=False,shapelet=tn.replace(' ','-').lower(),scale='mid')
    dims = tool_8B()
##    for fn in glob("8B life test/*.xls"):
##        f = compareShapeletPosTryPeriods(fn,dims,3.0,"torque",np.linspace(0.1,math.fsum(dims),1000))
##        f.savefig(f"8B life test/plots/{os.path.splitext(os.path.basename(fn))[0]}-diff-periods-mirror-false-av-3.0.png")
##        f=compareShapeletPosTryPeriods(fn,dims,2.0,"torque",np.linspace(0.1,math.fsum(dims),1000))
##        f.savefig(f"8B life test/plots/{os.path.splitext(os.path.basename(fn))[0]}-diff-periods-mirror-false-av-2.0.png")
##        f=compareShapeletPosTryPeriods(fn,dims,1.0,"torque",np.linspace(0.1,math.fsum(dims),1000))
##        f.savefig(f"8B life test/plots/{os.path.splitext(os.path.basename(fn))[0]}-diff-periods-mirror-false-av-1.0.png")
##        plt.close('all')
    #for f,fname in zip(figs,[f"8B-{tn}-av-{av}-half-tool-{'mirror-' if m else ''}{k}-scores-vs-hole-number",f"8B-{tn}-av-{av}-half-tool-{'mirror-' if m else ''}{k}-vs-score-position",f"8B-{tn}-av-{av}-half-tool-{'mirror-' if m else ''}{k}-scores-vs-{k}"]):
    #    f.savefig(f"dtw_tool/{fname}.png")
    #plt.show()
    #print("trying minimize")
    #best = tryOptimizeAV(glob("8B life test/*.xls")[0],[26,12],dims)
    dtwAlign("4B life test/*.xls",plot_res=True)
