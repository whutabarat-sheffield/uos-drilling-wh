from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from abyss.eadumatrixprofile_core import *

def process_plot_matrix_profile(full_filename_path, signal_template_file='signal_templates.json', cleanup_template_file='cleanup_templates.json', template=template_default) -> None:
    filename = Path(full_filename_path).name
    # Get the signals from the source file
    position, torque = get_setitec_signals(full_filename_path)
    # Get the templates from the json file
    try:
        with open(signal_template_file, 'r') as fp:
            segments = json.load(fp, cls=NpDecoder)
    except FileNotFoundError:
        segments = get_depth_estimation_templates()
    try:
        with open(cleanup_template_file, 'r') as fp:
            cleanup_segments = json.load(fp, cls=NpDecoder)
    except FileNotFoundError:
        cleanup_segments = get_signal_cleanup_templates()
    # Clean up the signal
    clean_torque = cleanup_signal(torque, cleanup_segments)
    # Calculate the estimated segments
    # idxs, dvals, dparrays, zscores = index_matches(torque, segments)
    idxs, dvals, dparrays, zscores = index_matches(clean_torque, segments)
    estimated_positions, estimated_segments = calculate_estimated_segments(position, clean_torque, idxs, template['template_extract'])
    # Plot the template on the target signal
    fig, ax = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(12, 8))
    color_cycle = ['blue', 'orange', 'green', 'magenta']
    marker_cycle = ['o', '+', 'x', '*']
    ax[0,0].plot(position, torque); ax[0,0].set_ylabel('torque'); ax[0,0].set_xlabel('position')
    ax[0,0].set_prop_cycle(color=color_cycle, marker=marker_cycle)
    ax[0,1].set_ylabel('torque'); ax[0,1].set_xlabel('index')
    ax[0,1].set_prop_cycle(color=color_cycle, marker=marker_cycle)
    for key in segments.keys():
        ax[0,0].plot(segments[key]['segment_position'], segments[key]['segment_torque'], label=f"template {key}")
        ax[0,1].plot(segments[key]['segment_torque'], label=f"template {key}")
    ax[0,0].legend(); ax[0,1].legend()
    ax[0,0].set_title(f'Template segments, on its original positions, superimposed on \n target signal: {filename}')
    ax[0,1].set_title(f'Individual template segments')
    # Plot the estimated segments on the target signal
    ax[1,0].plot(position, clean_torque); ax[1,0].set_ylabel('torque'); ax[1,0].set_xlabel('position')
    ax[1,0].set_prop_cycle(color=color_cycle, marker=reversed(marker_cycle))
    ax[1,1].set_ylabel('torque'); ax[1,1].set_xlabel('index')
    ax[1,1].set_prop_cycle(color=color_cycle, marker=reversed(marker_cycle))
    for key, estimated_position, estimated_segment, dval, zscore in zip(segments.keys(), estimated_positions, estimated_segments, dvals, zscores):
        ax[1,0].plot(estimated_position, estimated_segment, label=f"estimated {key}")  
        ax[1,1].plot(estimated_segment, label=f"estimated {key}")
        rag = convert_ordinal_rag_to_rag(convert_scores_to_ordinal_rag(dval, zscore))
        ax[1,1].annotate(xy=(len(estimated_segment),estimated_segment[-1]), text=f"dval={dval:.1f}\nzscore={zscore:.1f}\nrag={rag}", xytext=(5,0), textcoords='offset points', va='center', backgroundcolor=rag)
    ax[1,0].legend(); ax[1,1].legend()
    ax[1,0].set_title(f'Estimated segments identified on cleaned-up target signal: \n{filename}')
    ax[1,1].set_title(f'Estimated segments with zscore (high zscore = high similarity)')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    return fig, ax


def process_plot_matrix_profile_debug(full_filename_path, signal_template_file='signal_templates.json', cleanup_template_file='cleanup_templates.json', template=template_default, nvalleys=3) -> None:
    filename = Path(full_filename_path).name
    # Get the signals from the source file
    position, torque = get_setitec_signals(full_filename_path)
    # Get the templates from the json file
    try:
        with open(signal_template_file, 'r') as fp:
            segments = json.load(fp, cls=NpDecoder)
    except FileNotFoundError:
        segments = get_depth_estimation_templates()
    try:
        with open(cleanup_template_file, 'r') as fp:
            cleanup_segments = json.load(fp, cls=NpDecoder)
    except FileNotFoundError:
        cleanup_segments = get_signal_cleanup_templates()
    # Clean up the signal
    clean_torque = cleanup_signal(torque, cleanup_segments)
    # Calculate the estimated segments
    idxs, dvals, dparrays, zscores = index_matches(clean_torque, segments, nvalleys=nvalleys)
    estimated_positions, estimated_segments = calculate_estimated_segments(position, clean_torque, idxs, template['template_extract'])
    # Plot the template on the target signal
    fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(12, 8))
    color_cycle = ['blue', 'orange', 'green', 'magenta']
    marker_cycle = ['o', '+', 'x', '*']
    ax[0,0].plot(position, torque); ax[0,0].set_ylabel('torque'); ax[0,0].set_xlabel('position')
    ax[0,0].set_prop_cycle(color=color_cycle, marker=marker_cycle)
    ax[0,1].set_ylabel('torque'); ax[0,1].set_xlabel('index')
    ax[0,1].plot(clean_torque, label='torque', linestyle='--')
    axtwin=ax[0,1].twinx()
    axtwin.set_ylabel('distance')
    axtwin.set_yscale('log')
    axtwin.set_prop_cycle(color=color_cycle)
    for key, dparray in zip(segments.keys(),dparrays):
        ax[0,0].plot(segments[key]['segment_position'], segments[key]['segment_torque'], label=f"template {key}")
        # ax[0,1].plot(segments[key]['segment_torque'], label=f"template {key}")
        axtwin.plot(dparray, label=f"distance_profile {key}")
    
    ax[0,0].legend(); ax[0,1].legend(); axtwin.legend()
    ax[0,0].set_title(f'Template segments, on its original positions, superimposed on \n target signal: {filename}')
    ax[0,1].set_title(f'Individual template segments')
    # Plot the estimated segments on the target signal
    ax[1,0].plot(position, clean_torque); ax[1,0].set_ylabel('torque'); ax[1,0].set_xlabel('position')
    ax[1,0].set_prop_cycle(color=color_cycle, marker=reversed(marker_cycle))
    ax[1,1].set_ylabel('torque'); ax[1,1].set_xlabel('index')
    ax[1,1].set_prop_cycle(color=color_cycle, marker=reversed(marker_cycle))
    for key, estimated_position, estimated_segment, dval, zscore in zip(segments.keys(), estimated_positions, estimated_segments, dvals, zscores):
        ax[1,0].plot(estimated_position, estimated_segment, label=f"estimated {key}")  
        ax[1,1].plot(estimated_segment, label=f"estimated {key}")
        rag = convert_ordinal_rag_to_rag(convert_scores_to_ordinal_rag(dval, zscore))
        # rag = list(map(lambda x: x.replace('amber', 'yellow'), rag))
        c = dval / zscore
        ax[1,1].annotate(xy=(len(estimated_segment),estimated_segment[-1]), text=f"dval={dval:.1f}\nzscore={zscore:.1f}\nc={c:.2f}\nrag={rag}", xytext=(5,0), textcoords='offset points', va='center', backgroundcolor=rag)
    ax[1,0].legend(); ax[1,1].legend()
    ax[1,0].set_title(f'Estimated segments identified on cleaned-up target signal: \n{filename}')
    ax[1,1].set_title(f'Estimated segments with zscore (high zscore = high similarity)')
    [item.legend() for item in fig.axes]
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    return fig, ax


from scipy.stats import pearsonr 
def corrfunc(x, y, hue=None, ax=None, **kws):
    '''Plot the correlation coefficient in the bottom left hand corner of a plot.'''
    if hue is not None:
        hue_order = pd.unique(g.hue_vals)
        color_dict = dict(zip(hue_order, sns.color_palette('tab10', hue_order.shape[0]) ))
        groups = x.groupby(g.hue_vals)
        r_values = []
        for name, group in groups:
            mask = (~group.isnull()) & (~y[group.index].isnull())
            if mask.sum() > 0:
                r, _ = pearsonr(group[mask], y[group.index][mask])
                r_values.append((name, r))
        text = '\n'.join([f'{name}: ρ = {r:.2f}' for name, r in r_values])
        fontcolors = [color_dict[name] for name in hue_order]
        
    else:
        mask = (~x.isnull()) & (~y.isnull())
        if mask.sum() > 0:
            r, _ = pearsonr(x[mask], y[mask])
            text = f'ρ = {r:.2f}'
            fontcolors = 'grey'
            # print(fontcolors)
        else:
            text = ''
            fontcolors = 'grey'
        
    ax = ax or plt.gca()
    if hue is not None:
        for i, name in enumerate(hue_order):
            text_i = [f'{name}: ρ = {r:.2f}' for n, r in r_values if n==name][0]
            # print(text_i)
            color_i = fontcolors[i]
            ax.annotate(text_i, xy=(.02, .98-i*.05), xycoords='axes fraction', ha='left', va='top',
                        color=color_i, fontsize=10)
    else:
        ax.annotate(text, xy=(.02, .98), xycoords='axes fraction', ha='left', va='top',
                    color=fontcolors, fontsize=10)
        

# from https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)