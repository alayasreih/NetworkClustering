import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
from aux_functions import load_pickle
import numpy as np
# from osgeo import gdal
# import contextily as ctx
import scienceplots

plt.style.use(['science', 'no-latex'])

colors_base = ["#F8766D", "#CD9600", "#7CAE00", "#00BE67", "#00BFC4", "#00A9FF", "#C77CFF", "#FF61CC"]
colors_extended = colors_base + ["#FF8C42", "#B79F00", "#5C9EAD", "#FF4B00", "#7F00FF", "#FF6D6A", "#00BFFF",
                                 "#FF007F", "#8E44AD", "#3498DB", "#2ECC71", "#E74C3C", "#F1C40F", "#1ABC9C",
                                 "#9B59B6", "#2980B9", "#27AE60", "#E67E22", "#BDC3C7", "#34495E", "#16A085"]

colors_base_light = ['#fcbbb6', '#e6cb80', '#bed680', '#80deb3', '#80dfe2', '#80d4ff', '#e3beff', '#ffb0e6']
colors_extended_light = colors_base_light + ['#ffc6a0', '#dbcf80', '#aeced6', '#ffa580', '#bf80ff', '#ffb6b4',
                                             '#80dfff', '#ff80bf', '#c6a2d6', '#9acced', '#96e6b8', '#f3a59e',
                                             '#f8e287', '#8cdece', '#cdacdb', '#94c0dc', '#93d6b0', '#f2be90',
                                             '#dee1e3', '#9aa4ae', '#8ad0c2']

color_dict = {i: color for i, color in enumerate(colors_extended_light)}


def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def get_voronoi_polygons(centroids, radius, output_path):
    # extract the centroid coordinates
    centroid_points = centroids.apply(lambda row: [row['geometry'].x, row['geometry'].y], axis=1).tolist()

    # compute the infinite Voronoi regions
    vor = Voronoi(centroid_points)

    # get finite Voronoi regions and vertices
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=radius)

    # create polygons for the Voronoi regions
    polygons = []
    for region in regions:
        polygon = vertices[region]
        polygons.append(Polygon(polygon))

    gdf = []
    for label, centroid in zip(centroids['label'], centroids['geometry']):
        for polygon in polygons:
            if polygon.contains(centroid):
                gdf.append((label, polygon))
                break
    gdf = gpd.GeoDataFrame(gdf, columns=['label', 'geometry'], crs='EPSG:4326')

    # save the GeoDataFrame to a shapefile
    gdf.to_file(output_path)

    return gdf


def plot_voronoi_polygons(detectors, network, voronoi_polygons, bbox, plot_path):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    voronoi_polygons['color'] = voronoi_polygons['label'].apply(lambda x: color_dict[x])
    detectors.to_crs('EPSG:25832')
    voronoi_polygons.to_crs('EPSG:25832')
    network.to_crs('EPSG:25832')

    detectors.plot(ax=ax, color='white', edgecolor='none', markersize=0.2, zorder=3)
    voronoi_polygons.plot(ax=ax, color=voronoi_polygons['color'], linewidth=0.1, zorder=1)
    network.plot(ax=ax, color='black', linewidth=0.1, zorder=2)

    # create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', linestyle='none', color=color_dict[i],
                              markerfacecolor=color_dict[i], markersize=4, label=f'{i}')
                       for i in sorted(detectors['label'].unique())]
    legend = ax.legend(handles=legend_elements, loc='lower left', title='Cluster', frameon=True, title_fontsize=3,
                       fontsize=3, markerscale=0.2, labelspacing=0.4, handletextpad=0.4)

    # set the legend background to white and remove the frame
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('none')

    # set the x and y limits
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])

    # format
    ax.set_axis_off()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # save plot
    plt.savefig(plot_path, bbox_inches='tight', dpi=1200)
    plt.close()

    return


if __name__ == '__main__':
    scr = 'Ingolstadt'
    result_dir = 'results/{}/ambuehl_method_min_2_max_25'.format(scr)

    # load data
    labels = pd.read_csv(os.path.join(result_dir, 'ambuehl_labels.csv'))
    standard_input = load_pickle(r'data/{}/input/standard_input.pkl'.format(scr))
    detectors_standard_input = load_pickle(r'data/{}/input/detectors_standard_input.pkl'.format(scr))

    # calculate the radius from total bound of the network
    detectors = gpd.GeoDataFrame(
        pd.merge(
            detectors_standard_input[['detector_id', 'geometry']].drop_duplicates(), labels, how='inner',
            on='detector_id'),
        geometry='geometry', crs='EPSG:4326'
    )
    network = gpd.GeoDataFrame(
        standard_input[['edge_id', 'geometry']].drop_duplicates(), geometry='geometry', crs='EPSG:4326'
    )
    bbox = network.total_bounds
    radius = abs(max(bbox[2] - bbox[0], bbox[3] - bbox[1], key=abs))

    # calculate the centroids of each group
    centroids = detectors.groupby('label')['geometry'].apply(
        lambda x: unary_union(x).centroid).reset_index(name='geometry'
                                                       )
    # get and plot the Voronoi polygons
    voronoi_polygons = get_voronoi_polygons(centroids, radius, os.path.join(result_dir, 'voronoi_polygons.shp'))
    plot_voronoi_polygons(detectors, network, voronoi_polygons, bbox, os.path.join(result_dir, 'voronoi_polygons.png'))

