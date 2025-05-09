import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(data_points, h, label='Errors', hline=False):
    """
    Zee
    Given a list, this func draw a scatter figure
    :param data_points: a list contains the points to plot as scatter
    :param h: if a horizontal line is needed, h is the y-axis coordinate
    :param label: str, the label of scattering plot
    :param hline: bool, whether you need a horizontal line
    :return: show a figure, close to continue
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data_points)), data_points, label=label)
    plt.title('Error Scatter Plot with Shaded Region')
    plt.xlabel('Index')
    plt.ylabel('y')
    plt.legend()
    if hline:
        plt.axhline(y=h, color='b', linestyle='--')
    plt.show()


def output_heatmap_plot(true_exit_index, estimated_exit_index, original_data,
                        heatmap_all, full_label_plot=False, exit_end=None, save_figure=False,
                        hole_id=None, error=None):
    """
    Zee
    Given the depth label, original data from drilling, and model's output, this func visualise the model's estimation
    :param true_exit_index: exit point label
    :param estimated_exit_index: model's estimation convert from the max value of heatmap
    :param original_data:
    :param heatmap_all: the heat_map output by model
    :param full_label_plot: if true, exit_end will be plotted as well
    :param exit_end: the index of exit end point.
    :param save_figure: whether save the figure (for developer).
    :param hole_id: hole_id
    :param error: prediction error
    :return:
    """
    plt.figure(figsize=(15, 10))
    plt.axvline(x=true_exit_index, color='red', linestyle='-', linewidth=1, label='Measured depth')
    plt.axvline(x=estimated_exit_index, color='blue', linestyle='-', linewidth=1, label='Estimated depth')
    if full_label_plot:
        plt.axvline(x=exit_end, color='red', linestyle='-', linewidth=1, label='Exit end')

    x_axis = np.arange(original_data.shape[0])
    for col in range(original_data.shape[1]):
        plt.plot(x_axis, original_data[:, col], label=f'Col {col}')

    extent = [x_axis.min(), x_axis.max(), -3, 3]  # Control the x and y axis limits of the figure
    plt.imshow(heatmap_all.T, cmap='hot', aspect='auto', alpha=0.5, extent=extent)
    plt.colorbar(label='Heatmap Intensity')
    plt.legend()
    if save_figure:
        path = '/home/zeeai/PycharmProjects/TsDrilling/Test/Figures/lab_data_results/'
        plt.savefig(f'{path}_{hole_id}_error{error}.png', dpi=100)
        plt.close()
        print('figure saved')
    else:
        plt.show()