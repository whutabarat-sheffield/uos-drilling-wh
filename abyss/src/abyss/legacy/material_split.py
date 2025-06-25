import abyss.dataparser as dp
import matplotlib.pyplot as plt
import ruptures as rpt
from glob import glob
import numpy as np
import os
from ruptures.base import BaseCost
from abyss.legacy.modelling import rolling_gradient
from scipy.signal import wiener

class ForcedSpacing(BaseCost):
    '''
        Custom class to try and enforce minimum spacing between breakpoints

        The cost is calculated based on the sum of the weighted distance between breakpoints.

        The distance between breakpoints is weighted based on if it falls within a certain range.

        If D(x) is the per-element difference between breakpoints, then the cost is 100 if it's
        outside the range [mind, maxd] and 0 otherwise. The cost of the segment is the sum of these
        values.
    '''
    model = "forcespace"
    min_size=2

    def __init__(self,mind,maxd,dist):
        super().__init__()
        self.mind = mind
        self.maxd = maxd
        # 3 sigma
        self.sigma = maxd-mind
        # peak locataion
        self.pkloc = (maxd-mind)/2
        
        self.dist = dist

    def fit(self, signal):
        self.signal = signal
        return self

    def error(self,start,end):
        ''' Return cost of segment '''
        #sub = self.signal[start:end]
        subdist = self.dist[start:end]
        # find difference between current breakpoints
        diff = np.diff(subdist)
        # those within range have a value of 0 and those outside have a value of 1
        mask = (diff>=self.mind) & (diff<=self.maxd)
        diff[mask] = 0
        diff[~mask] = 100
        return np.sum(diff)

def tryPeltCustom(path,mind=0.55,maxd=5.5,**kwargs):
    ''' Apply PELT using ForcedSpacing custom cost class '''
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)
    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    
    for aa,model,mk in zip(ax.flatten(),["l1","l2","rbf"],['rx','go','k^']):
        algo = rpt.Pelt(custom_cost=ForcedSpacing(mind,maxd,pos)).fit(signal)
        my_bkps = np.asarray(algo.predict(pen=kwargs.get("penalty",0)))
        aa.plot(pos,signal)
        aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,markersize=10)
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")
    f.suptitle(os.path.splitext(path)[0])
    return f

def tryPelt(path,**kwargs):
    '''
        Apply PELT algorithm with fixed parameters to the torque signal and plot the breakpoints

        For PELT see https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/

        Currently tried cost models:
            l1, l2, rbf

        Inputs:
            path : Path to Setitec XLS files
            min_size : Min segment length
            jump : Sumsample
            params : See https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/
            penalty : Penalty value. Affects runtime

        Returns matplotlib figure object
    '''
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)
    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    
    for aa,model,mk in zip(ax.flatten(),["l1","l2","rbf"],['rx','go','k^']):
        algo = rpt.Pelt(model=model, min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",5)).fit(signal)
        my_bkps = np.asarray(algo.predict(pen=kwargs.get("penalty",0)))
        aa.plot(pos,signal)
        aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,markersize=10)
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")
    f.suptitle(os.path.splitext(path)[0])
    return f

def tryPeltCondense(path,**kwargs):
    '''
        Apply PELT algorith with fixed parameters to the torque signal and "condense" them by taking the average of those
        within range of eachother

        For PELT see https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/

        Currently tried cost models:
            l1, l2, rbf

        Inputs:
            path : Path to Setitec XLS files
            min_size : Min segment length
            jump : Sumsample
            params : See https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/
            penalty : Penalty value. Affects runtime

        Returns matplotlib figure object
    '''
    from scipy.spatial import KDTree
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)
    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    
    for aa,model,mk in zip(ax.flatten(),["l1","l2","rbf"],['rX','go','k^']):
        algo = rpt.Pelt(model=model, min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",1)).fit(signal)
        my_bkps = np.asarray(algo.predict(pen=kwargs.get("penalty",0)))
        bppos = pos[my_bkps-1]
        tree = KDTree(bppos.reshape(-1,1))
        new_bkps = []
        new_tq = []
        bc = bppos.tolist().copy()
        while len(bc)>0:
            p = bc.pop()
            # search for points within 1mm of point
            # returns indicies
            n = tree.query_ball_point(p,5.5)
            # if there are neighbours
            # take the average
            new_bkps.append(np.mean(bppos[n]))
            # remove neighbours from search list
            for nn in n:
                if bppos[nn] in bc:
                    bc.remove(bppos[nn])
        # find torque values for condensed breakpoints
        new_tq = [signal[np.argmin(np.abs(p-pos))] for p in new_bkps]
        aa.plot(pos,signal,label="Original")
        aa.plot(bppos,signal[my_bkps-1],mk,label=model,markersize=10)
        aa.plot(new_bkps,new_tq,'m>',label=f"{model} cond.",markersize=10)
        aa.legend()
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps vs {len(new_bkps)} bps")
    f.suptitle(os.path.splitext(path)[0])
    return f

def tryPeltSlider(path,**kwargs):
    ''' https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/ '''
    from matplotlib.widgets import Slider
    models = kwargs.get("models",["l1","l2","rbf"])
    fig,ax = plt.subplots(ncols=len(models),constrained_layout=True)
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    if kwargs.get("no_filter",False):
        signal = wiener(signal,30)

    bps = []
    for aa,model,mk in zip(ax.flatten(),["l1","l2","rbf"],['rx','go','k^']):
        aa.plot(pos,signal,label='Original')
        algo = rpt.Pelt(model=model, min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",5)).fit(signal)
        my_bkps = np.asarray(algo.predict(pen=0.1))
        bps.append(aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)[0])
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")

    axpen = fig.add_axes([0.25, 0.02, 0.65, 0.03])
    pen_slider = Slider(
        ax=axpen,
        label='Penalty',
        valmin=0.1,
        valmax=10,
    )

    def update(val):
        p = pen_slider.val
        print(f"processing with penalty {p}")
        while bps:
            bps.pop().remove()
        for aa,model,mk in zip(ax.flatten(),["l1","l2","rbf"],['rx','go','k^']):
            algo = rpt.Pelt(model=model, min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",5)).fit(signal)
            my_bkps = np.asarray(algo.predict(pen=p))
            bps.append(aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)[0])
            aa.set_title(f"{model.capitalize()} {len(my_bkps)} bps")
    pen_slider.on_changed(update)
    plt.show()

def _calcLenPelt(signal,model,p):
    return len(rpt.Pelt(model=model, min_size=3, jump=5).fit_predict(signal,pen=p))

def _calcLenKernel(signal,model,p):
    try:
        return len(rpt.KernelCPD(kernel=model, min_size=3, jump=5).fit_predict(signal,pen=p))
    except AssertionError:
        return None

def _calcLenBottom(signal,model,p):
    return len(rpt.BottomUp(model=model, jump=5).fit_predict(signal,pen=p))

def penaltyEffect(path,mode="pelt",**kwargs):
    import multiprocessing as mp
    if mode == "pelt":
        models = ["l1","l2","rbf"]
    elif mode == "kernel":
        models = ["linear","rbf","cosine"]
    elif mode == "bottom":
        models = ["l2","l1", "rbf", "normal", "ar"]
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    penalty = np.linspace(0.0,10.0,100)
    pendict = {m : [] for m in models}
    for m in models:
        print(f"processing {m}")
        #algo = rpt.Pelt(model=m, min_size=kwargs.get("min_size",3), jump=kwargs.get("jump",5)).fit(signal)
        #pendict[m] = list(map(lambda x : algo.predict(pen=x),penalty))
        if mode == "pelt":
            pendict[m] = mp.Pool(8).starmap(_calcLenPelt,[(signal,m,p) for p in penalty])
        elif mode == "kernel":
            pendict[m] = mp.Pool(8).starmap(_calcLenKernel,[(signal,m,p) for p in penalty])
        elif mode == "bottom":
            pendict[m] = mp.Pool(8).starmap(_calcLenBottom,[(signal,m,p) for p in penalty])
    fig,ax = plt.subplots(ncols=len(models),constrained_layout=True)
    for aa,(k,v) in zip(ax.flatten(),pendict.items()):
        aa.plot(penalty,v,'x-')
        aa.set(xlabel="Penalty",ylabel="Number of Breakpoints",title=k)
    return fig

def kernelBreakpoints(path,**kwargs):
    ''' https://centre-borelli.github.io/ruptures-docs/user-guide/detection/kernelcpd/ '''
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)
    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    n_bkps = kwargs.get("num_bps",15)
    for aa,model,mk in zip(ax.flatten(),["linear","rbf","cosine"],['rx','go','k^']):
        aa.plot(pos,signal,label='Original')
        algo_c = rpt.KernelCPD(kernel=model, min_size=kwargs.get("min_size",2)).fit(signal)
        my_bkps = np.asarray(algo_c.predict(n_bkps=n_bkps))
        aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")
    f.suptitle(os.path.splitext(path)[0])
    return f

def tryKernelSlider(path,**kwargs):
    ''' https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/ '''
    from matplotlib.widgets import Slider
    models = kwargs.get("models",["linear","rbf","cosine"])
    fig,ax = plt.subplots(ncols=len(models),constrained_layout=True)
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    if kwargs.get("no_filter",False):
        signal = wiener(signal,30)

    bps = []
    for aa,model,mk in zip(ax.flatten(),["linear","rbf","cosine"],['rx','go','k^']):
        aa.plot(pos,signal,label='Original')
        algo_c = rpt.KernelCPD(kernel=model, min_size=kwargs.get("min_size",2)).fit(signal)
        my_bkps = np.asarray(algo_c.predict(pen=0.1))
        bps.append(aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)[0])
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")

    axpen = fig.add_axes([0.25, 0.02, 0.65, 0.03])
    pen_slider = Slider(
        ax=axpen,
        label='Penalty',
        valmin=0.1,
        valmax=10,
    )

    def update(val):
        p = pen_slider.val
        print(f"processing with penalty {p}")
        while bps:
            bps.pop().remove()
        for aa,model,mk in zip(ax.flatten(),["linear","rbf","cosine"],['rx','go','k^']):
            algo_c = rpt.KernelCPD(kernel=model, min_size=kwargs.get("min_size",2)).fit(signal)
            my_bkps = np.asarray(algo_c.predict(pen=p))
            bps.append(aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)[0])
            aa.set_title(f"{model.capitalize()} {len(my_bkps)} bps")
    pen_slider.on_changed(update)
    plt.show()

def kernelBreakpointsAuto(path,**kwargs):
    ''' https://centre-borelli.github.io/ruptures-docs/user-guide/detection/kernelcpd/ '''
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)
    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    for aa,model,mk in zip(ax.flatten(),["linear","rbf","cosine"],['rx','go','k^']):
        aa.plot(pos,signal,label='Original')
        algo_c = rpt.KernelCPD(kernel=model, min_size=kwargs.get("min_size",2)).fit(signal)
        my_bkps = np.asarray(algo_c.predict())
        if len(my_bkps.shape)==0:
            break
        aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")
    f.suptitle(os.path.splitext(path)[0])
    return f

def windowBreakpoints(path,**kwargs):
    ''' https://centre-borelli.github.io/ruptures-docs/user-guide/detection/window/ '''
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    f,ax = plt.subplots(nrows=2,ncols=3,constrained_layout=True)
    n_bkps = kwargs.get("num_bps",15)
    wsz = kwargs.get("win_size",100)
    for aa,model,mk in zip(ax.flatten(),["l2","l1", "rbf", "normal", "ar", "auto"],['rX','go','k^','mP','y*','cp']):
        if model == "auto":
            sigma = np.std(signal)
            my_bkps = np.asarray(algo.predict(pen=np.log(len(signal)) * sigma**2))
        else:
            algo = rpt.Window(width=wsz, model=model).fit(signal)
            my_bkps = np.asarray(algo.predict(n_bkps=n_bkps))
        aa.plot(pos,signal)
        aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")
    f.suptitle(os.path.splitext(path)[0])
    return f

def bottomBreakpoints(path,**kwargs):
    ''' https://centre-borelli.github.io/ruptures-docs/user-guide/detection/bottomup/ '''
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    f,ax = plt.subplots(nrows=2,ncols=3,constrained_layout=True)
    n_bkps = kwargs.get("num_bps",13)
    for aa,model,mk in zip(ax.flatten(),["l2","l1", "rbf", "normal", "ar"],['rX','go','k^','y*','cp']):
        aa.plot(pos,signal,label='Original')
        algo = rpt.BottomUp(model=model, jump=kwargs.get("jump",5)).fit(signal)
        my_bkps = np.asarray(algo.predict(n_bkps=n_bkps))
        aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")
    f.suptitle(f"Bottom-Up {os.path.splitext(path)[0]}")
    return f

def tryBottomSlider(path,**kwargs):
    ''' https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/ '''
    from matplotlib.widgets import Slider
    models = kwargs.get("models",["l2","l1", "rbf", "normal", "ar"])
    fig,ax = plt.subplots(ncols=len(models),constrained_layout=True)
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    if kwargs.get("no_filter",False):
        signal = wiener(signal,30)

    bps = []
    for aa,model,mk in zip(ax.flatten(),models,['rX','go','k^','y*','cp']):
        aa.plot(pos,signal,label='Original')
        algo = rpt.BottomUp(model=model, jump=kwargs.get("jump",5)).fit(signal)
        my_bkps = np.asarray(algo.predict(pen=0.1))
        bps.append(aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)[0])
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")

    axpen = fig.add_axes([0.25, 0.02, 0.65, 0.03])
    pen_slider = Slider(
        ax=axpen,
        label='Penalty',
        valmin=0.1,
        valmax=10,
        valinit=0.1
    )

    def update(val):
        p = pen_slider.val
        print(f"processing with penalty {p}")
        while bps:
            bps.pop().remove()
        for aa,model,mk in zip(ax.flatten(),models,['rX','go','k^','y*','cp']):
            my_bkps = np.asarray(rpt.BottomUp(model=model, jump=kwargs.get("jump",5)).fit_predict(signal,pen=p))
            print(model,len(my_bkps))
            bps.append(aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)[0])
            aa.set_title(f"{model.capitalize()} {len(my_bkps)} bps")
    pen_slider.on_changed(update)
    plt.show()

def bottomBreakpointsVsGrad(path,N=30,**kwargs):
    ''' https://centre-borelli.github.io/ruptures-docs/user-guide/detection/bottomup/ '''
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    #f,ax = plt.subplots(nrows=2,ncols=3,constrained_layout=True)
    f = plt.figure(constrained_layout=True)
    n_bkps = kwargs.get("num_bps",13)
    for i,(model,mk) in enumerate(zip(["l2","l1", "rbf", "normal", "ar"],['rX','go','k^','y*','cp']),start=1):
        aa = f.add_subplot(2,3,i)
        aa.plot(pos,signal,label='Original')
        algo = rpt.BottomUp(model=model, jump=kwargs.get("jump",5)).fit(signal)
        my_bkps = np.asarray(algo.predict(n_bkps=n_bkps))
        aa.plot(pos[my_bkps-1],signal[my_bkps-1],mk,label=model,markersize=10)
        aa.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title=f"{model.capitalize()} {len(my_bkps)} bps")
        # find rolling gradient
        grad = rolling_gradient(signal,N)
        tax = aa.twinx()
        tax.plot(pos[my_bkps-1],grad[my_bkps-1],'mP')
        tax.set_ylabel("Gradient")
        i+=1
    f.suptitle(f"Bottom-Up {os.path.splitext(path)[0]}")
    return f

def naturalBreaks(path,**kwargs):
    ''' https://github.com/mthh/jenkspy '''
    from jenkspy import JenksNaturalBreaks, jenks_breaks
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    breaks = np.asarray(jenks_breaks(signal,n_classes=kwargs.get("num_bps",50)))
    bps = np.asarray([np.where(signal==v)[0][0] for v in breaks])

    f,ax = plt.subplots(constrained_layout=True)
    ax.plot(pos,signal,label="Original")
    ax.plot(pos[bps],signal[bps],'X',markersize=10,label="Natural")
    ax.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title="Natural Breakpoints")
    f.suptitle(f"Natural Breaks {os.path.splitext(path)[0]}")
    return f

def naturalBreaksVsGrad(path,N=10,**kwargs):
    ''' https://github.com/mthh/jenkspy '''
    from jenkspy import JenksNaturalBreaks, jenks_breaks
    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    breaks = np.asarray(jenks_breaks(signal,n_classes=kwargs.get("num_bps",50)))
    bps = np.asarray([np.where(signal==v)[0][0] for v in breaks])

    f,ax = plt.subplots(constrained_layout=True)
    ax.plot(pos,signal,label="Original")
    ax.plot(pos[bps],signal[bps],'X',markersize=10,label="Natural")
    ax.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)",title="Natural Breakpoints")
    # find rolling gradient
    grad = rolling_gradient(signal,N)
    tax = ax.twinx()
    tax.plot(pos[bps],grad[bps],'kP')
    tax.set_ylabel("Gradient")
    f.suptitle(f"Natural Breaks {os.path.splitext(path)[0]}")
    return f

def tryPiecewise(path,**kwargs):
    from modelling import BreakpointFit

    data = dp.loadSetitecXls(path,version="auto_data")
    signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
    pos = np.abs(data['Position (mm)'].values)

    f,ax = plt.subplots(constrained_layout=True)
    ax.plot(pos,signal)
    new_bkps = BreakpointFit().fit(pos,signal)
    tq = [signal[np.argmin(np.abs(p-pos))] for p in new_bkps]
    ax.plot(new_bkps,tq,'X')
    ax.set(xlabel="Position (mm)",ylabel="Torque + Empty (A)")
    f.suptitle(f"PWLF {os.path.splitext(path)[0]}")
    return f

def estimateMaterials(path,**kwargs):
    from modelling import BreakpointFit
    import multiprocessing as mp
    locs = {'pelt' : {'1st':[],'2nd':[],'3rd':[]},'kernel':{'1st':[],'2nd':[],'3rd':[]},'pwlf':{'1st':[],'2nd':[],'3rd':[]},'bus':{'1st':[],'2nd':[],'3rd':[]},'pwlf':{'1st':[],'2nd':[],'3rd':[]}
            ,'kernel_13':{'1st':[],'2nd':[],'3rd':[]},'bus_13':{'1st':[],'2nd':[],'3rd':[]}}
    for fn in sorted(glob(path)):
        data = dp.loadSetitecXls(fn,version="auto_data")
        signal = data['I Torque (A)'].values + data['I Torque Empty (A)'].values
        pos = np.abs(data['Position (mm)'].values)
        # PELT
        my_bkps = np.asarray(rpt.Pelt(model="rbf", min_size=3, jump=5).fit_predict(signal,pen=4.4))-1
        locs['pelt']['1st'].append(pos[my_bkps[0]])
        locs['pelt']['2nd'].append(pos[my_bkps[5]])
        locs['pelt']['3rd'].append(pos[my_bkps[min(13,len(my_bkps))]])
        # KERNEL
        my_bkps = np.asarray(rpt.KernelCPD(kernel="linear", min_size=3, jump=10).fit_predict(signal,pen=0.5))
        if len(my_bkps)>0:
            my_bkps -=1 
            locs['kernel']['1st'].append(pos[my_bkps[1]])
            locs['kernel']['2nd'].append(pos[my_bkps[20]])
            locs['kernel']['3rd'].append(pos[my_bkps[60 if 60<=len(my_bkps) else -1]])
            
        else: 
            locs['kernel']['1st'].append(None)
            locs['kernel']['2nd'].append(None)
            locs['kernel']['3rd'].append(None)

        my_bkps = np.asarray(rpt.KernelCPD(kernel="linear", min_size=3, jump=10).fit_predict(signal,n_bkps=13))
        if len(my_bkps)>0:
            my_bkps -=1 
            locs['kernel_13']['1st'].append(pos[my_bkps[0]])
            locs['kernel_13']['2nd'].append(pos[my_bkps[4]])
            locs['kernel_13']['3rd'].append(pos[my_bkps[11] if 11<len(my_bkps) else -1])
        else:
            locs['kernel_13']['1st'].append(None)
            locs['kernel_13']['2nd'].append(None)
            locs['kernel_13']['3rd'].append(None)

        # PWLF
        new_bkps = BreakpointFit().fit(pos,signal)
        locs['pwlf']['1st'].append(pos[my_bkps[2]])
        locs['pwlf']['2nd'].append(pos[my_bkps[6]])
        locs['pwlf']['3rd'].append(pos[my_bkps[10 if 10<len(my_bkps) else -1]])
        # BUS
        my_bkps = np.asarray(rpt.BottomUp(model="ar", jump=5).fit_predict(signal,pen=0.1))-1
        if len(my_bkps)>0:
            my_bkps -=1 
            locs['bus']['1st'].append(pos[my_bkps[0]])
            locs['bus']['2nd'].append(pos[my_bkps[2]])
            locs['bus']['3rd'].append(pos[my_bkps[4 if 4<len(my_bkps) else -1]])
        else:
            locs['bus']['1st'].append(None)
            locs['bus']['2nd'].append(None)
            locs['bus']['3rd'].append(None)

        my_bkps = np.asarray(rpt.BottomUp(model="ar", jump=5).fit_predict(signal,n_bkps=13))-1
        locs['bus_13']['1st'].append(pos[my_bkps[0]])
        locs['bus_13']['2nd'].append(pos[my_bkps[4]])
        locs['bus_13']['3rd'].append(pos[my_bkps[8 if 8<len(my_bkps) else -1]])

    f,ax = plt.subplots(ncols=3,constrained_layout=True)
    for model,v in locs.items():
        for aa,(track,vals) in zip(ax.flatten(),v.items()):
            aa.plot(vals,'x-',label=model)
            aa.set(xlabel="File Index",ylabel="Position (mm)",title=track)
    for aa in ax.flatten():
        aa.legend()
    return f

def processAll(path,opath='',**kwargs):
    min_sz = kwargs.get("min_size",3)
    jump = kwargs.get("jump",10)
    n_bkps = kwargs.get("num_bps",13)
    penalty = kwargs.get("penalty",0)
    win_sz = kwargs.get("win_size",100)
    for fn in glob(path):
##        f = tryPelt(fn,min_sz=min_sz,jump=jump,penalty=penalty)
##        f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-pelt-models-minsz-{min_sz}-jump-{jump}-penalty-{penalty}.png"))
##        plt.close(f)
##
##        f = kernelBreakpoints(fn,min_sz=min_sz,num_bps=n_bkps)
##        f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-kernel-models-minsz-{min_sz}-nbps-{n_bkps}.png"))
##        plt.close(f)
##
##        f = windowBreakpoints(fn,num_bps=n_bkps,win_size=win_sz)
##        f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-window-models-winsz-{win_sz}-nbps-{n_bkps}.png"))
##        plt.close(f)
##
        f = bottomBreakpoints(fn,num_bps=n_bkps,jump=jump)
        f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-bottom-models-jump-{jump}-nbps-{n_bkps}.png"))
        plt.close(f)
##
##        f = naturalBreaks(fn,num_bps=n_bkps)
##        f.savefig(os.path.join(opath,f"{os.path.splitext(os.path.basename(fn))[0]}-natural-models-nbps-{n_bkps}.png"))
##        plt.close(f)
    plt.close('all')

if __name__ == "__main__":
    estimateMaterials(r"8B life test/*.xls",mode="bottom",opath="breakpoints",min_size=3,jump=5,penalty=0)
    plt.show()
