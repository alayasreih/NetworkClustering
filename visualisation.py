import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# from osgeo import gdal
# import contextily as ctx
import scienceplots

plt.style.use(['science', 'no-latex'])

colors_base = ["#F8766D", "#CD9600", "#7CAE00", "#00BE67", "#00BFC4", "#00A9FF", "#C77CFF", "#FF61CC"]
colors_extended = colors_base + ["#FF8C42", "#B79F00", "#5C9EAD", "#FF4B00", "#7F00FF", "#FF6D6A", "#00BFFF",
                                 "#FF007F", "#8E44AD", "#3498DB", "#2ECC71", "#E74C3C", "#F1C40F", "#1ABC9C",
                                 "#9B59B6", "#2980B9", "#27AE60", "#E67E22", "#BDC3C7", "#34495E", "#16A085"]

color_dict = {i: color for i, color in enumerate(colors_extended)}


def plot_clusters(standard_input, labels, plot_path):
    gdf = gpd.GeoDataFrame(
        pd.merge(standard_input[['edge_id', 'geometry']].drop_duplicates(), labels, how='inner', on='edge_id'),
        geometry='geometry', crs='EPSG:4326'
    ).to_crs('EPSG:25832')

    # plot
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    gdf['color'] = gdf['label'].apply(lambda x: color_dict[x])
    gdf.plot(ax=ax, color=gdf['color'], linewidth=0.1)

    # create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', linestyle='none', color=color_dict[i],
                              markerfacecolor=color_dict[i], markersize=4, label=f'{i}')
                       for i in sorted(gdf['label'].unique())]
    ax.legend(handles=legend_elements, loc='upper right', title='Cluster', frameon=False, title_fontsize=3,
              fontsize=3, markerscale=0.2, labelspacing=0.4, handletextpad=0.4)

    # format
    ax.set_axis_off()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # save plot
    plt.savefig(plot_path, bbox_inches='tight', dpi=1200)
    plt.close()
    return


def plot_mfd(mfd, plot_path):
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    axs = axs.flatten()

    for (label, temp), ax in zip(mfd.groupby('label'), axs):
        ax.scatter(temp['density'], temp['flow'], s=0.5, color=color_dict[label], edgecolors='none')
        ax.set_xlabel('Density [veh/km-ln]', fontsize=2, labelpad=1)
        ax.set_ylabel('Flow [veh/hr-ln]', fontsize=2, labelpad=1)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.tick_params(axis='both', which='both', labelsize=2, pad=1, top=False, right=False)
        ax.minorticks_off()
        ax.spines['left'].set_linewidth(0.2)
        ax.spines['bottom'].set_linewidth(0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', width=0.2, length=1)

    for ax in axs[len(mfd['label'].unique()):]:
        ax.axis('off')

    plt.savefig(plot_path, bbox_inches='tight', dpi=600)
    plt.close(fig)

    return


def plot_density(stats, plot_path):
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, temp in stats.groupby('label'):
        ax.plot(temp['timestep'], temp['density_mean'], linewidth=0.6, color=color_dict[label])
        ax.plot(temp['timestep'], temp['density_std'], linestyle='--', linewidth=0.6, color=color_dict[label])

    ax.set_xlabel('Time [s]', fontsize=6, labelpad=4)
    ax.set_ylabel('Density [veh/km-ln]', fontsize=6, labelpad=4)
    ax.set_xlim(0, stats['timestep'].max())
    ax.set_ylim(0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', labelsize=6, pad=4, top=False, right=False)
    ax.minorticks_off()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.savefig(plot_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    return


def plot_mfd_aval(eval_results, plot_path):
    fig, axs = plt.subplots(2, 6, figsize=(8, 2))
    axs = axs.flatten()

    for (label, temp), ax in zip(eval_results.groupby('label'), axs):
        points_normalized = temp.iloc[0]['points_normalized']
        alpha_shape = temp.iloc[0]['alpha_shape']

        # Plot the points
        ax.scatter(points_normalized[:, 0], points_normalized[:, 1], s=1, label=f'Points {label}',
                   color=color_dict[label], alpha=0.6)

        # Plot the alpha shape
        if not alpha_shape.is_empty:
            x, y = alpha_shape.exterior.xy
            ax.plot(x, y, label=f'Alpha Shape {label}', color='black', linewidth=0.5)

        ax.set_xlabel('Density', fontsize=2, labelpad=1)
        ax.set_ylabel('Flow', fontsize=2, labelpad=1)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.tick_params(axis='both', which='both', labelsize=2, pad=1, top=False, right=False)
        ax.minorticks_off()
        ax.legend(fontsize=2)

    # Hide any unused subplots
    for ax in axs[len(eval_results['label'].unique()):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight', dpi=600)
    plt.close(fig)

    return


def plot_traveltime_eval(eval_results, plot_path):
    eval_results = eval_results.loc[
        (eval_results.percentage_error > eval_results.percentage_error.quantile(0.01)) &
        (eval_results.percentage_error < eval_results.percentage_error.quantile(0.99))
        ]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # cumulative density distribution of  percent error
    p_error = eval_results.percentage_error

    # Freedman-Diaconis rule again to determine bins width and count
    bin_width = (2 * (np.quantile(p_error, 0.75) - np.quantile(p_error, 0.25))) / (len(p_error) ** (1 / 3))
    bin_count = int(np.ceil((np.max(p_error) - np.min(p_error)) / bin_width))

    # plot
    n, bins, patches = axs[0].hist(p_error, bin_count, linewidth=0.5, density=True, histtype='step',
                                   cumulative=True, alpha=1.0, color='blue')
    axs[0].fill_between(x=bins[:-1], y1=n, where=(0.1 <= n) & (n <= 0.9), color='lightblue', alpha=0.2)

    # Add a line showing the expected distribution
    sigma = np.std(p_error)
    mu = np.mean(p_error)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    y = y.cumsum()
    y /= y[-1]
    axs[0].plot(bins, y, '--', linewidth=0.5, color='orange', label='Expected Normal Distribution')

    # format
    axs[0].set_xlabel(r'Percent Error [%]', fontsize=5, labelpad=2)
    axs[0].set_ylabel('Likelihood of Occurrence', fontsize=5, labelpad=2)
    axs[0].set_ylim(0, 1)
    axs[0].legend(fontsize=4, markerscale=0.4, scatteryoffsets=[0.6], loc='upper left', frameon=False)
    axs[0].tick_params(axis='both', which='both', labelsize=4, pad=2, length=1, width=0.4, top=False, right=False)
    axs[0].set_aspect(0.88 / axs[0].get_data_ratio(), adjustable='box')

    # 2D histogram of actual vs estimated travel time & 45-degree plot
    x = eval_results.travel_time
    y = eval_results.estimated_travel_time

    # Freedman-Diaconis rule to determine bins width and count
    bin_width = (2 * (np.quantile(x, 0.75) - np.quantile(x, 0.25))) / (len(x) ** (1 / 3))
    bin_count = int(np.ceil((np.max(x) - np.min(x)) / bin_width))

    # plot
    histo_2d = axs[1].hist2d(x, y, bins=bin_count, cmin=1, cmax=max(x), cmap='Blues')

    # format cbar
    cbar = fig.colorbar(histo_2d[3], ax=axs[1], location='right', shrink=0.75)
    cbar.set_label(label=r'Count [$\#$]', size=5)
    cbar.ax.tick_params(which='both', labelsize=4, pad=2, length=1, width=0.4)
    cbar.ax.minorticks_off()

    # arbitrary values to plot regression line
    arb_x = np.arange(0, max(x) + 5)

    # arbitrary values to plot estimation error
    arb_y1 = [float(i - 0.3 * i) for i in arb_x]
    arb_y2 = [float(i - 0.2 * i) for i in arb_x]
    arb_y3 = [float(i - 0.1 * i) for i in arb_x]
    arb_y = arb_x
    arb_y4 = [float(i + 0.1 * i) for i in arb_x]
    arb_y5 = [float(i + 0.2 * i) for i in arb_x]
    arb_y6 = [float(i + 0.3 * i) for i in arb_x]

    # plot
    axs[1].plot(arb_x, arb_y1, '--', linewidth=0.6, c='red')
    axs[1].plot(arb_x, arb_y2, '--', linewidth=0.6, c='orange')
    axs[1].plot(arb_x, arb_y3, '--', linewidth=0.6, c='yellow')
    axs[1].plot(arb_x, arb_y, linewidth=0.6, c='green', label=r'$\ 0\%$')
    axs[1].plot(arb_x, arb_y4, '--', linewidth=0.6, c='yellow', label=r'$\pm\ 10\%$')
    axs[1].plot(arb_x, arb_y5, '--', linewidth=0.6, c='orange', label=r'$\pm\ 20\%$')
    axs[1].plot(arb_x, arb_y6, '--', linewidth=0.6, c='red', label=r'$\pm\ 30\%$')

    # format
    axs[1].set_xlabel('Actual Travel Time [sec]', fontsize=5, labelpad=2)
    axs[1].set_ylabel('Estimated Travel Time [sec]', fontsize=5, labelpad=2)
    axs[1].legend(fontsize=4, markerscale=0.4, scatteryoffsets=[0.6], loc='upper left', frameon=False)
    axs[1].tick_params(axis='both', which='both', labelsize=4, pad=2, length=1, width=0.4, top=False, right=False)

    max_lim = max([max(histo_2d[1]), max(histo_2d[2])])
    axs[1].set_xlim(0, max_lim)
    axs[1].set_ylim(0, max_lim)
    axs[1].set_aspect('equal', adjustable='box')

    plt.savefig(plot_path, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close(fig)

    return
