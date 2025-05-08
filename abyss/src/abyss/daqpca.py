from glob import glob
import nptdms
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt
from scaleogram.wfun import fastcwt
import scaleogram as scg
import abyss.dataparser as dp

from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn import metrics

from joblib import dump

import pandas as pd

def plot_dwt(time,data,coefs,**kwargs):
    # decompose
    coefs = pywt.wavedec(data.values.flatten(),wavelet,level=level)
    # spec
    f = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2,nrows=len(coefs),figure=f)
    ax_data = f.add_subplot(spec[:,0])
    ax_data.plot(time,data)
    ax_data.set(xlabel=kwargs.get("xlabel","Time (s)"),
                ylabel = kwargs.get("ylabel","Magnitude"),
                title = kwargs.get("title","Original Data"))
    for ci,cp in enumerate(coefs):
        ax = f.add_subplot(spec[:,ci])
        ax.set_ylabel("Wavelet Coefficient")
        ax.set_title("Approx. Coeffiecents" if ci==0 else f"Detail Coefficients {ci}")
        ax.plot(cp)
    return f

def plot_pca(time,data,coefs,scales,**kwargs):
    '''
        Plot the data and the PCA coefficients from CWT coeffs

        Creates a grid of plots with the original data occupying the left column and the PCA plots
        on the right hand column

        Inputs:
            time : Vector of time values
            data : Original data signal
            coefs : List of PCA components
            scales : Scales used in CWT to create coefficients. Used in plotting
            xlabel : X-axis label for original data plot
            ylabel : Y-axis label for original data plot
            title : Axis title for the original data plot

        Returns figure object
    '''
    # create figure
    f = plt.figure(tight_layout=True,figsize=(16,14))
    # create grid spec
    spec = gridspec.GridSpec(ncols=2,nrows=len(coefs),figure=f)
    # add subplot for the original data
    # covers left hand column
    ax_data = f.add_subplot(spec[:,0])
    ax_data.plot(time,data)
    # set the labels
    ax_data.set(xlabel=kwargs.get("xlabel","Time (s)"),
                ylabel = kwargs.get("ylabel","Magnitude"),
                title = kwargs.get("title","Original Data"))
    # iterate over the components
    for ci,cc in enumerate(coefs):
        #print("component ",ci," ",len(cc))
        ax = f.add_subplot(spec[ci,1])
        ax.set_ylabel(f"PC{ci}")
        ax.set_xlabel("Scale")
        ax.set_title(f"Component {ci}")
        #ax.plot(scales,cc)
        for ss,aa in zip(scales,cc):
            ax.plot([ss,ss],[0,aa],'b-')
    return f

def plot_pca_clip(time,data,clip,scales=None,**kwargs):
    '''
        Plot the data period and PCA of target time periods

        The following procedure is applied to each time period specified by the user
            - Apply CWT to create a coefficients matrix of size scales x time
            - Transpose to form feature matrix of size time x scales (samples, features)
            - Normalize using sklearn StandardScaler
            - Apply PCA to normed features targeting 95% variance

        The data and PCA components are passed to the plot_pca function.

        Given the potentially large number of figures, the figure, time window and number of components
        is YIELDED.

        If scales is None, scales are set to scales = np.arange(1, min(len(time_clip)/10, 100))

        Inputs:
            time : Time vector of data
            data : Original signal data
            clip : List of 2-element tuples representing time windows to clip to
            scales : Scales used in CWT. If None, scales are calculated based on time vector
            **kwargs : Keyword args for plot_pca
    '''
    # iterate over time periods
    for A,B in clip:
        # create mask for target time period
        mask = (time>=A) & (time<=B)
        # clip data
        time_clip = time[mask]
        data_clip = data.values.flatten()[mask]
        # get CWT coefficients
        scales = np.arange(1, min(len(time_clip)/10, 100))
        coefs,_ = fastcwt(data_clip,scales,'morl',T)
        # transpose to form feature vector
        # the CWT at each scale is a vector
        features = coefs.T
        # normalize to 0-1 with mean at 0
        feat_norm = StandardScaler().fit_transform(features)
        # perform PCA
        pca = PCA(n_components=0.95).fit(feat_norm)
        # clip the data to the specific time period and 
        yield plot_pca(time_clip,data_clip,pca.components_,scales,**kwargs),(A,B),len(pca.components_)

def calculate_score(y_true,y_pred,weighted=True):
    ''' Calculate several metrics for classification score '''
    if isinstance(weighted,bool):
        weighted ='weighted' if weighted else 'binary'

    return metrics.accuracy_score(y_true,y_pred),metrics.f1_score(y_true,y_pred,average=weighted),metrics.precision_score(y_true,y_pred,average=weighted),metrics.recall_score(y_true,y_pred,average=weighted)

def plot_metrics(stats,clips,**kwargs):
    '''
        Plot the SVM classifier metrics

        The user can supply the original data to provide context. If given, the metrics
        are plotting on a twin y axis.

        The metrics are plotted in the middle of the time 

        Inputs:
            metrics: List of tuples containing the metrics from calculate_score at each time period.
            clips : List of 2-element tuples representing time windows
            time : Original time vector. If given, the metrics are placed on a twin y axis
            data : Original data vector
            dxlabel : Data x-label. Defaults to Time (s).
            dylabel : Data y-label. Defaults to Signal Strength
            xlabel : Metrics x-label. Shown if show_win is False. Defaults to Time (s).
            ylabel : Metrics y-label. Defaults to Score
            title : Figure suptitle. Defaults to SVM Metrics Score
            show_win : Plot lines showing time window the metrics apply across
            win_col : Window line color. Default darkorange.

        Returns the figure object
            
    '''
    # create figure and axes
    f,ax = plt.subplots()
    # set the axes used for plotting to the current axes
    ax_met = ax
    # if the user has given the original data
    # create a twin axes for the score
    if "time" in kwargs:
        ax_met = ax.twinx()
        # plot data
        ax.plot(kwargs["time"],kwargs["data"],'-',label="Data")
        ax.set(xlabel=kwargs.get("dxlabel","Time (s)"),
               ylabel=kwargs.get("dylabel","Signal Strength"))
    # time vector as middle value of window
    tclip = [(A/B)/2 for A,B in clips]
    # create separate vectors for metrics
    acc = []
    f1 = []
    prec = []
    recall = []
    # unpack and populate
    for a,f,p,r in stats:
        acc.append(a)
        f1.append(f)
        prec.append(p)
        recall.append(r)
    # convert to dictionary
    met_dict = {'Accuracy' : acc,
                'F1-score' : f1,
                'Precision' : prec,
                'Recall' : recall}
    # iterate over metrics dictionary plotting each
    # can use the same axes as normalize is True meaning they#re on the same scale
    for label,metric in met_dict.items():
        ax_met.plot(tclip,metric,'x-',label)
    show_win = kwargs.get("show_win",False)
    # add boundaries for time period
    if show_win:
        # get y limits
        ymin,ymax = ax.get_ylim()
        # iterate over the time windows
        for A,B in clips:
            # draw dotted line across y axis limits
            ax.plot((A,A),(ymin,ymax),kwargs.get('win_col','darkorange'),linestyle="dashed")
    # set the metrics axis labels
    ax_met.set(xlabel=kwargs.get('xlabel',"Time (s)") if not show_win else '',
           ylabel=kwargs.get('ylabel',"Score"))
    # set figure title
    f.suptitle(kwargs.get("title","SVM Metrics Score"))
    # add legend to plot
    plt.legend()
    # return figure obkect
    return f
    
def train_svm_clips(time,data,clip,stack,scales=None,**kwargs):
    '''
        Train a SVM using PCA as features to identify what material is being drilled.

        Each signal is processed using CWT and PCA to a feature set.

        The class labels are set as follows:
            air : 1
            transition : 2
            material #1 : 3
            material #2 : 4
            ...
            material #N : N+1
            retraction : N+2

        The data is known to start on air and then alternate between transition and material.
        For simplicity, all sections after reaching the final material are set as retraction
        rather than specifying what material it is retracting through. This may change depending
        on need.

        Each period is processed separately and is used to train a single model. The periods are then
        evaluated and the results stored in a list

        Inputs:
            time : Time vector
            data : Data vector
            clip : List of 2-element tuples representing time windows
            scales : Scales used in CWT. If None, scales are calculated based on time vector.
            nc : Number of PCA components to create.

        Returns model, class labels, results 
            
    '''
##    # classes alternate betwen flat and slope
##    # first one is air
##    classes = []
##    # total number of transntions for first half of the curve is number of mats + number of transitions between mats + 1 for
##    # initially starting at air.
##    for ii in range(1,(2*len(stack))+2):
##        #print(ii,ii%2)
##        # if the index is one
##        # then it's air
##        if ii==1:
##            #print("air!")
##            classes.append(1)
##        # if the index is even then it's a transitions
##        elif (ii%2)==0:
##            #print("transition")
##            classes.append(2)
##        # if the index is odd then it's a material
##        # the class label is set as the max class label +1
##        # So the first material is 3, next is 4 etc.
##        elif (ii%2)!=0:
##            #print("material")
##            classes.append(max(classes)+1)
##    clip = clip[:len(classes)]
    classes = [0,1,1,1,1,2,2,2,2,0,0,0,0]
    print(classes)
    # make a pipeline with a StandardScaler as a pre-procesor and the SVM
    clf = make_pipeline(StandardScaler(),svm.SVC(gamma='auto',probability=False,break_ties=True))
    feats = []
    labels = []
    if scales is None:
        scales = np.arange(1, min(len(time_clip)/10, 100))
    print(scales.shape)
    # iterate over time periods
    for (A,B),cl in zip(clip,classes):
        # create mask for target time period
        mask = (time>=A) & (time<=B)
        # clip data
        time_clip = time[mask]
        data_clip = data[mask]
        # get CWT coefficients
        coefs,_ = fastcwt(data_clip,scales,'morl',T)
        # transpose to form feature vector
        # the CWT at each scale is a vector
        coefs = coefs.T
        # normalize to 0-1 with mean at 0
        feat_norm = StandardScaler().fit_transform(coefs)
        print(feat_norm.shape)
        # perform PCA
        pca = PCA(n_components=kwargs.get('nc',8)).fit(feat_norm)
        # re arrange the components into a feature set
        ft = np.hstack([cc.reshape((-1,1)) for cc in pca.components_])
        feats.append(ft)
        #print(ft.shape)
        labels.append(int(cl)*np.ones(ft.shape[0],dtype='uint8'))
        #print("labels ",labels[-1].shape)
    labels = np.hstack(labels)
    feats = np.vstack(feats)
    print("feats ",feats.shape," labels ",labels.shape," unique labels",np.unique(labels))
    # fit model marking each sample with the same class
    clf.fit(feats,labels)
    del feats
    # matrix of results
    results = []
    results_unq = []
    # re-iterate over time periods and evaluate
    for (A,B),cl in zip(clip,classes):
        # create mask for target time period
        mask = (time>=A) & (time<=B)
        # clip data
        time_clip = time[mask]
        data_clip = data[mask]
        # get CWT coefficients
        coefs,_ = fastcwt(data_clip,scales,'morl',T)
        # transpose to form feature vector
        # the CWT at each scale is a vector
        coefs = coefs.T
        # normalize to 0-1 with mean at 0
        feat_norm = StandardScaler().fit_transform(coefs)
        # perform PCA
        pca = PCA(n_components=kwargs.get('nc',8)).fit(feat_norm)
        # re arrange the components into a feature set
        ft = np.hstack([cc.reshape((-1,1)) for cc in pca.components_])
        # append the results matrix
        results.append(clf.predict(ft))
        results_unq.append(np.unique(results[-1]))
        #probs.append(clf.predict_proba(ft))
    results = np.hstack(results)
    # calculate scores
    scores = calculate_score(results,labels,'macro')
    print(scores)
    print(results_unq)
    # return model, list of target class labels and the raw results
    return clf,classes,results_unq,scores


def train_svm_clips_binary(time,data,clip,stack,scales=None,**kwargs):
    '''
        Train a SVM using PCA as features to identify what material is being drilled.

        Each signal is processed using CWT and PCA to a feature set.

        The class labels are set as follows:
            0 : Drilling
            1 : Retraction

        The data is known to start on air and then alternate between transition and material.
        For simplicity, all sections after reaching the final material are set as retraction
        rather than specifying what material it is retracting through. This may change depending
        on need.

        Each period is processed separately and is used to train a single model. The periods are then
        evaluated and the results stored in a list

        Inputs:
            time : Time vector
            data : Data vector
            clip : List of 2-element tuples representing time windows
            scales : Scales used in CWT. If None, scales are calculated based on time vector.
            nc : Number of PCA components to create.

        Returns model, class labels, results 
            
    '''
    # set drilled regions to zero
    classes = [0,]*(2*len(stack)+2)
    # set retraction to 1
    classes.extend([1,]*(len(clip)-len(classes)))
    # make a pipeline with a StandardScaler as a pre-procesor and the SVM
    clf = make_pipeline(StandardScaler(),svm.SVC(gamma='auto'))
    feats = []
    labels = []
    # iterate over time periods
    for (A,B),cl in zip(clip,classes):
        # create mask for target time period
        mask = (time>=A) & (time<=B)
        # clip data
        time_clip = time[mask]
        data_clip = data.values.flatten()[mask]
        # get CWT coefficients
        scales = np.arange(1, min(len(time_clip)/10, 100))
        coefs,_ = fastcwt(data_clip,scales,'morl',T)
        # transpose to form feature vector
        # the CWT at each scale is a vector
        coefs = coefs.T
        # normalize to 0-1 with mean at 0
        feat_norm = StandardScaler().fit_transform(coefs)
        # perform PCA
        pca = PCA(n_components=kwargs.get('nc',8)).fit(feat_norm)
        # re arrange the components into a feature set
        ft = np.hstack([cc.reshape((-1,1)) for cc in pca.components_])
        feats.append(ft)
        #print(ft.shape)
        labels.append(int(cl)*np.ones(ft.shape[0],dtype='uint8'))
        #print("labels ",labels[-1].shape)
    labels = np.hstack(labels)
    feats = np.vstack(feats)
    print("feats ",feats.shape," labels ",labels.shape," unique labels",np.unique(labels))
    # fit model marking each sample with the same class
    clf.fit(feats,labels)
    del feats
    # matrix of results
    results = []
    results_unq = []
    # re-iterate over time periods and evaluate
    for (A,B),cl in zip(clip,classes):
        # create mask for target time period
        mask = (time>=A) & (time<=B)
        # clip data
        time_clip = time[mask]
        data_clip = data.values.flatten()[mask]
        # get CWT coefficients
        scales = np.arange(1, min(len(time_clip)/10, 100))
        coefs,_ = fastcwt(data_clip,scales,'morl',T)
        # transpose to form feature vector
        # the CWT at each scale is a vector
        coefs = coefs.T
        # normalize to 0-1 with mean at 0
        feat_norm = StandardScaler().fit_transform(coefs)
        # perform PCA
        pca = PCA(n_components=kwargs.get('nc',8)).fit(feat_norm)
        # re arrange the components into a feature set
        ft = np.hstack([cc.reshape((-1,1)) for cc in pca.components_])
        # append the results matrix
        results.append(clf.predict(ft))
        results_unq.append(np.unique(results[-1]))
    results = np.hstack(results)
    # calculate scores
    scores = calculate_score(results,labels,False)
    print(scores)
    print(results_unq)
    # return model, list of target class labels and the raw results
    return clf,classes,results_unq,scores


coi = {
        'alpha':0.5,
        'hatch':'/',
    }
# number of levels to decompose under DWT
level =4
# which wavelet to use
wavelet = 'db8'
# time period between sampling
T = 1/1e5
# number of PCA components to use
nc = 5
# time periods to clip to
time_clips = [(0.0,1.3),
              (1.3,1.5),
              (1.5,2.15),
              (2.15,2.27),
              (2.27,3.3),
              (3.3,3.93),
              (3.93,4.4),
              (4.4,5.5),
              (5.5,5.9),
              (5.9,6.3),
              (6.3,6.6),
              (6.6,6.7),
              (6.7,7.0)]

labels = [0,1,1,1,1,2,2,2,2,0,0,0,0]

def multi_label_train():
    from datetime import datetime
    # iterate over TDMS files
    for path in glob("daq/U*/*.tdms"):
        # get filename
        fname = os.path.splitext(os.path.basename(path))[0]
        with open(f"{fname}-svm-train-stats.txt",'w') as file:
            file.write(datetime.today().strftime('%Y-%m-%d'))
            file.write("\n==============================================\n\n")
        # get tool
        tool,_,coupon = fname.split(' ')
        coupon = int(coupon)
        # load target MAT changepoints
        cps_thrust = dp.loadMatChangePoints(f"mat/{tool}_IThrust_A__changepoints_slm.mat")
        cps_torque = dp.loadMatChangePoints(f"mat/{tool}_ITorque_A__changepoints_slm.mat")
        # open file
        with nptdms.TdmsFile(path) as file:
            # iterate over each group        
            for gg in file.groups():
                # format the group name
                gg_name = gg.name.replace('\\','-')
                # get hole
                hole = int(gg_name[1])
                # data stack
                data_stack = None
                # iterate over each channel
                for cc,units in zip(gg.channels(),[cc.properties["unit_string"] for cc in gg.channels()]):
                    # format the channel name
                    cc_name = cc.name.replace('\\','-')
                    # set changepoints used to form labels
                    if 'ai0' in cc_name:
                        cps = cps_thrust
                    elif 'ai1' in cc_name:
                        cps = cps_torque
                    else:
                        cps = cps_torque
                    # get changepoints for hole and coupon
                    cps_gp_cp = cps[hole-1,coupon-1,0,:]
                    # create time period paird
                    time_clips_cp = [(A,B) for A,B in zip(cps_gp_cp,cps_gp_cp[1:])]
                    print(f"found channel {gg_name}, {cc_name} in file {fname}")
                    print(f"hole {hole}, coupon {coupon}")
                    print("loading data")
                    # get data from channel
                    data = cc.as_dataframe(time_index=True).values.flatten()
                    #data = (pd.DataFrame(np.abs(data.values.flatten())**2).rolling(10000).mean())**0.5
                    print("data ",data.shape)
                    # create time vector
                    time = np.arange(0.0,len(data)*T,T,dtype='float16')
                    # decompose
                    #coefs = pywt.wavedec(data,wavelet,level=level)
                    # plot the DWT
                    #f=plot_dwt(time,data,coefs,ylabel=f"{cc.name} ({units})")
                    #f.savefig(f"{fname}-{gg_name}-plot-dwt-{wavelet}-levels-{level}.png")

    ##                # get CWT coefficients
    ##                scales = np.arange(1, min(len(time)/10, 100))
    ##                coefs,_ = fastcwt(data.values.flatten(),scales,'morl',T)
    ##                # transpose to form feature vector
    ##                # the CWT at each scale is a vector
    ##                # matrix is now organised by sample x scale
    ##                features = coefs.T
    ##                # normalize to 0-1 with mean at 0
    ##                feat_norm = StandardScaler().fit_transform(features)
    ##                # perform PCA
    ##                pca = PCA(n_components=0.95).fit(feat_norm)
    ##                print(f"{fname}, nc= {pca.n_components_}")
    ##                f = plot_pca(time,data,pca.components_,scales)
    ##                f.suptitle(f"{fname}, Measurement {cc.name} from group {gg.name}")
    ##                f.savefig(f"{fname}-{gg_name}-{cc_name}-pca-nc-{pca.n_components_}.png")
    ##                plt.close(f)

    ##                # iterate over PCA applied to target time periods
    ##                for f,(A,B),nc in plot_pca_clip(time,data,time_clips):
    ##                    f.savefig(f"{fname}-{gg_name}-{cc_name}-pca-nc-{nc}-time-window-{A}-{B}.png")
    ##                    plt.close(f)

                    # iterate over PCA applied to time periods from MAT changepoints
    ##                for f,(A,B),nc in plot_pca_clip(time,data,time_clips_cp):
    ##                    f.savefig(f"{fname}-{gg_name}-{cc_name}-pca-nc-{nc}-time-window-{A:.2f}-{B:.2f}.png")
    ##                    plt.close(f)
    ##
    ##                for (A,B) in time_clips_cp:
    ##                    # clip data
    ##                    time_clip = time[(time>=A) & (time<=B)]
    ##                    data_clip = data.values.flatten()[(time>=A) & (time<=B)]
    ##                    # set scales
    ##                    scales = np.arange(1, min(len(time_clip)/10, 100))
    ##                    # create axes
    ##                    f,ax = plt.subplots(ncols=2)
    ##                    # plot original data
    ##                    ax[0].plot(time_clip,data_clip)
    ##                    ax[0].set(xlabel="Time (s)",
    ##                              ylabel=f"{cc_name} ({units})",
    ##                              title=f"{cc_name},t=[{A},{B}]")
    ##                    # plot CWT coefs for target period
    ##                    ax[1] = scg.cws(time_clip,data_clip,scales,'morl',coikw=coi,yaxis='scale',xlabel="Time (s)",ylabel="Scale",
    ##                                        title=f"{fname}, {gg_name}, {cc_name}",ax=ax[1])
    ##                    # set figure title
    ##                    f.suptitle(f"{fname}, {gg_name}, {cc_name}")
    ##                    # save 
    ##                    f.savefig(f"{fname}-{gg_name}-{cc_name}-cwt-morl-window-{A:.2f}-{B:.2f}.png")
    ##                    plt.close(f)
    ##                # if the datastack has not been set
    ##                if data_stack is None:
    ##                    data_stack = data.values
    ##                # before being flattened, the data is shape (x,1)
    ##                # so can be stacked to form the feature vector
    ##                else:
    ##                    data_stack = np.hstack((data_stack,data.values))
                    # train svm on signal
                    model,train_labels,results,fit_stats = train_svm_clips(time,data,time_clips_cp,["CFRP","Al"],scales=np.arange(60,100,1))
                    # make model
                    dump(model,f"{fname}-{gg_name}-{cc_name}-pca-svm-svc-nc-8-narrow.joblib")
                    with open(f"{fname}-svm-train-stats-narrow.txt",'a') as file:
                        file.write(f"{gg_name}, {cc_name}\n")
                        acc,f1,prec,recall = fit_stats
                        file.write(f"\taccuracy: {acc}\n")
                        file.write(f"\tF1 (weighted): {f1}\n")
                        file.write(f"\tPrecision (weighted): {prec}\n")
                        file.write(f"\tRecall (weighted): {recall}\n")
                        for rr,(A,B) in zip(results,time_clips):
                            file.write(f"\ttime: {A} {B}: {rr.tolist()}\n")
                        
    ##                # create title for SVM plots
    ##                svm_title = f"SVM Training Results for {fname},{gg_name},{cc_name}"
    ##                # plot metrics
    ##                f = plot_metrics(metrics,time_clips,title=svm_title)
    ##                f.savefig(f"{fname}-{gg_name}-{cc_name}-svm-train-results-nc-8.png")
    ##                plt.close(f)
    ##                # plot metrics with data
    ##                f = plot_metrics(metrics,time_clips,time=time,data=data.values.flatten(),dylabel=f"{cc_name} ({units})",title=svm_title,show_win=True)
    ##                f.savefig(f"{fname}-{gg_name}-{cc_name}-svm-train-results-nc-8-with-data.png")
    ##                plt.close(f)
    ##            print("data stack size ",data_stack.shape)
    ##            # normalize the data
    ##            feat_norm = StandardScaler().fit_transform(data_stack)
    ##            # perform pca
    ##            pca = PCA(n_components=0.95).fit(feat_norm)
    ##            # create figure
    ##            f,axes = plt.subplots(nrows=len(pca.components_),figsize=(14,16),tight_layout=True)
    ##            # iterate over the components
    ##            for (ci,cc),ax in zip(enumerate(pca.components_),axes):
    ##                # set plotting labels
    ##                ax.set_ylabel(f"PC{ci}")
    ##                ax.set_xlabel("Index")
    ##                ax.set_title(f"Component {ci}")
    ##                # iterate over each scale value
    ##                # draw a line from 0 to height
    ##                for ii,aa in enumerate(cc):
    ##                    ax.plot([ii,ii],[0.0,aa],'b-')
    ##            # set figure title
    ##            f.suptitle(f"{fname}\nPCA using all signals in group {gg_name}")
    ##            f.savefig(f"{fname}-{gg_name}-pca-nc-{pca.n_components_}-all-channels.png")
    ##            plt.close(f)

def binary_train():
    from datetime import datetime
    # iterate over TDMS files
    for path in glob("daq/U*/*.tdms"):
        # get filename
        fname = os.path.splitext(os.path.basename(path))[0]
        with open(f"{fname}-svm-train-stats.txt",'w') as file:
            file.write(datetime.today().strftime('%Y-%m-%d'))
            file.write("\n==============================================\n\n")
        # get tool
        tool,_,coupon = fname.split(' ')
        coupon = int(coupon)
        # load target MAT changepoints
        cps = dp.loadMatChangePoints(f"mat/{tool}_IThrust_A__changepoints_slm.mat")
        # open file
        with nptdms.TdmsFile(path) as file:
            # iterate over each group        
            for gg in file.groups():
                # format the group name
                gg_name = gg.name.replace('\\','-')
                # get hole
                hole = int(gg_name[1])
                # get changepoints for hole and coupon
                cps_gp_cp = cps[hole-1,coupon-1,0,:]
                # create time period paird
                time_clips_cp = [(A,B) for A,B in zip(cps_gp_cp,cps_gp_cp[1:])]
                # data stack
                data_stack = None
                # iterate over each channel
                for cc,units in zip(gg.channels(),[cc.properties["unit_string"] for cc in gg.channels()]):
                    # format the channel name
                    cc_name = cc.name.replace('\\','-')
                    print(f"found channel {gg_name}, {cc_name} in file {fname}")
                    print(f"hole {hole}, coupon {coupon}")
                    print("loading data")
                    # get data from channel
                    data = cc.as_dataframe(time_index=True)
                    print("data ",data.shape)
                    # create time vector
                    time = np.arange(0.0,len(data)*T,T,dtype='float16')
                    # train svm on signal
                    model,train_labels,results,fit_stats = train_svm_clips_binary(time,data,time_clips,["CFRP","Al"],scales=None)
                    # make model
                    dump(model,f"{fname}-{gg_name}-{cc_name}-pca-svm-svc-binary-nc-8.joblib")
                    with open(f"{fname}-svm-binary-train-stats.txt",'a') as file:
                        file.write(f"{gg_name}, {cc_name}\n")
                        acc,f1,prec,recall = fit_stats
                        file.write(f"\taccuracy: {acc}\n")
                        file.write(f"\tF1 (weighted): {f1}\n")
                        file.write(f"\tPrecision (weighted): {prec}\n")
                        file.write(f"\tRecall (weighted): {recall}\n")
                        for rr,(A,B) in zip(results,time_clips):
                            file.write(f"\ttime: {A} {B}: {rr.tolist()}\n")

if __name__ == "__main__":
    multi_label_train()
