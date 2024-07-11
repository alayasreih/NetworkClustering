import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from input import DataProcessor
from aux_functions import load_pickle, save_pickle
from abc import abstractmethod
import scienceplots

plt.style.use(['science', 'no-latex'])


class MLOptimiser:
    def __init__(self, input_data, standard_input, network_graph, timestep_eval, dir):
        self.input_data = input_data
        self.standard_input = standard_input
        self.network_graph = network_graph
        self.timestep_eval = timestep_eval
        self.dir = dir

    def save_pickle(self, obj, file_name):
        file_path = os.path.join(self.dir, file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        return

    def ml_metrics(self, labels):
        silhouette = silhouette_score(
            self.input_data.to_numpy(), labels
        )

        davies_bouldin = davies_bouldin_score(
            self.input_data.to_numpy(), labels
        )

        calinski_harabasz = calinski_harabasz_score(
            self.input_data.to_numpy(), labels
        )
        return silhouette, davies_bouldin, calinski_harabasz

    def ave_density_sd(self, labels):
        labelled_input = self.standard_input.merge(
            pd.DataFrame({'edge_id': self.input_data.index, 'label': labels}
                         ), on='edge_id', how='inner'
        )
        labelled_input = labelled_input[labelled_input.timestep == self.timestep_eval]

        densities_sd = []
        densities_cv = []
        for _, temp in labelled_input.groupby('label'):
            stats = DescrStatsW(temp.density, weights=temp.length)
            density_sd = stats.std
            density_cv = stats.std / stats.mean
            densities_sd.append(density_sd)
            densities_cv.append(density_cv)

        return np.mean(densities_sd), np.mean(densities_cv)

    def ave_connectivity(self, labels):
        labelled_input = self.standard_input.merge(
            pd.DataFrame({'edge_id': self.input_data.index, 'label': labels}
                         ), on='edge_id', how='inner'
        )
        labelled_input = labelled_input[['edge_id', 'lane_id', 'label', 'length']].drop_duplicates(ignore_index=True)

        connectivities = []
        for _, temp in labelled_input.groupby('label'):
            cluster_length = temp.length.sum()
            cluster_subgraph = self.network_graph.subgraph(temp.edge_id.unique())
            cluster_largest_cc = max(nx.connected_components(cluster_subgraph), key=len)
            cluster_largest_cc_length = temp[temp.edge_id.isin(cluster_largest_cc)].length.sum()
            connectivity = cluster_largest_cc_length / cluster_length
            connectivities.append(connectivity)

        return np.mean(connectivities)

    @abstractmethod
    def optimise(self):
        pass


class DBSCANOptimiser(MLOptimiser):
    def __init__(self, input_data, standard_input, network_graph, timestep_eval, dir, max_min_samples):
        super().__init__(input_data, standard_input, network_graph, timestep_eval, dir)
        self.max_min_samples = max_min_samples

    def optimise(self):
        opt_results = []
        for min_samples in range(2, self.max_min_samples + 1):
            # determine optimal epsilon
            neighbors = NearestNeighbors(n_neighbors=min_samples).fit(self.input_data.to_numpy())
            distances, _ = neighbors.kneighbors(self.input_data.to_numpy())
            distances_sorted = np.sort(distances[:, min_samples - 1], axis=0)

            opt_eps = KneeLocator(
                np.arange(len(distances_sorted)), distances_sorted,
                S=1, curve='convex', direction='increasing', interp_method='polynomial'
            ).knee_y

            # perform DBSCAN
            dbscan = DBSCAN(
                eps=opt_eps, min_samples=min_samples
            ).fit(
                self.input_data.to_numpy()
            )

            labels = dbscan.labels_

            # calculate ML metrics
            no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            no_noise_points = list(labels).count(-1)
            silhouette, davies_bouldin, calinski_harabasz = self.ml_metrics(labels)

            # calculate traffic metrics
            ave_density_sd, ave_density_cv = self.ave_density_sd(labels)
            ave_connectivity = self.ave_connectivity(labels)

            opt_results.append(
                (min_samples, opt_eps, no_clusters, no_noise_points, silhouette, davies_bouldin, calinski_harabasz,
                 ave_density_sd, ave_density_cv, ave_connectivity)
            )
        opt_results = pd.DataFrame(
            opt_results,
            columns=[
                'min_samples', 'opt_eps', 'no_clusters', 'no_noise_points', 'silhouette',
                'davies_bouldin', 'calinski_harabasz', 'ave_density_sd', 'ave_density_cv', 'ave_connectivity'
            ]
        )

        # calculate rate of change for each metric
        for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'ave_density_sd', 'ave_density_cv',
                       'ave_connectivity']:
            opt_results[f'{metric}_roc'] = opt_results[metric].diff() / opt_results[metric].shift(1)

        # determine optimal points
        optimal_points = {
            'opt_no_noise_points': opt_results.loc[opt_results.no_noise_points.idxmin(), 'min_samples'],
            'opt_silhouette': opt_results.loc[opt_results.silhouette.idxmax(), 'min_samples'],
            'opt_davies_bouldin': opt_results.loc[opt_results.davies_bouldin.idxmin(), 'min_samples'],
            'opt_calinski_harabasz': opt_results.loc[opt_results.calinski_harabasz.idxmax(), 'min_samples'],
            'opt_ave_density_sd': opt_results.loc[opt_results.ave_density_sd.idxmin(), 'min_samples'],
            'opt_ave_density_cv': opt_results.loc[opt_results.ave_density_cv.idxmin(), 'min_samples'],
            'opt_ave_connectivity': opt_results.loc[opt_results.ave_connectivity.idxmax(), 'min_samples']
        }
        self.plot_optimisation(opt_results, optimal_points)

        self.save_pickle(opt_results, 'dbscan_opt_results.pkl')
        self.save_pickle(optimal_points, 'dbscan_optimal_points.pkl')

    def plot_optimisation(self, opt_results, optimal_points):
        fig, axs = plt.subplots(nrows=7, ncols=3, figsize=(20, 28), dpi=600)

        plt.yticks(fontsize=7)
        plt.xticks(fontsize=7)

        min_samples = opt_results.min_samples
        plots = [
            (0, 0, min_samples, opt_results.opt_eps, r'Optimum $\epsilon$', 'Minimum Samples', 'opt_eps'),
            (0, 1, min_samples, opt_results.no_clusters, 'No. of Clusters', 'Minimum Samples', 'no_clusters'),
            (
            0, 2, min_samples, opt_results.no_noise_points, 'No. of Noise Points', 'Minimum Samples', 'no_noise_points',
            optimal_points['opt_no_noise_points']),
            (1, 0, min_samples, opt_results.silhouette, 'Silhouette Score', 'Minimum Samples', 'silhouette',
             optimal_points['opt_silhouette']),
            (1, 1, min_samples, opt_results.silhouette_roc, 'Silhouette ROC', 'Minimum Samples', 'silhouette_roc'),
            (2, 0, min_samples, opt_results.davies_bouldin, 'Davies-Bouldin Score', 'Minimum Samples', 'davies_bouldin',
             optimal_points['opt_davies_bouldin']),
            (2, 1, min_samples, opt_results.davies_bouldin_roc, 'Davies-Bouldin ROC', 'Minimum Samples',
             'davies_bouldin_roc'),
            (3, 0, min_samples, opt_results.calinski_harabasz, 'Calinski-Harabasz Score', 'Minimum Samples',
             'calinski_harabasz', optimal_points['opt_calinski_harabasz']),
            (3, 1, min_samples, opt_results.calinski_harabasz_roc, 'Calinski-Harabasz ROC', 'Minimum Samples',
             'calinski_harabasz_roc'),
            (4, 0, min_samples, opt_results.ave_density_sd, 'Ave. Density SD', 'Minimum Samples', 'ave_density_sd',
             optimal_points['opt_ave_density_sd']),
            (4, 1, min_samples, opt_results.ave_density_sd_roc, 'Ave. Density SD ROC', 'Minimum Samples',
             'ave_density_sd_roc'),
            (5, 0, min_samples, opt_results.ave_density_cv, 'Ave. Density CV', 'Minimum Samples', 'ave_density_cv',
             optimal_points['opt_ave_density_cv']),
            (5, 1, min_samples, opt_results.ave_density_cv_roc, 'Ave. Density CV ROC', 'Minimum Samples',
             'ave_density_cv_roc'),
            (
            6, 0, min_samples, opt_results.ave_connectivity, 'Ave. Connectivity', 'Minimum Samples', 'ave_connectivity',
            optimal_points['opt_ave_connectivity']),
            (6, 1, min_samples, opt_results.ave_connectivity_roc, 'Ave. Connectivity ROC', 'Minimum Samples',
             'ave_connectivity_roc')
        ]

        for plot in plots:
            row, col, x_data, y_data, y_label, x_label, metric, *opt_line = plot
            axs[row, col].plot(x_data, y_data, color='#005293')
            axs[row, col].set_xlabel(x_label, fontsize=8, labelpad=2)
            axs[row, col].set_ylabel(y_label, fontsize=8, labelpad=2)
            axs[row, col].tick_params(axis='both', which='both', labelsize=7, pad=2, top=False, right=False)
            axs[row, col].set_xlim(left=2)
            if opt_line:
                axs[row, col].axvline(x=opt_line[0], color='#a2ad00', linestyle='--')

        # hide the unused axes
        for row in range(7):
            for col in range(3):
                if (row, col) not in [(plot[0], plot[1]) for plot in plots]:
                    axs[row, col].set_visible(False)

        plt.savefig(os.path.join(self.dir, 'dbscan_optimisation_plot.png'), bbox_inches='tight')
        plt.close('all')


class GMMOptimiser(MLOptimiser):
    def __init__(self, input_data, standard_input, network_graph, timestep_eval, dir, max_no_clusters):
        super().__init__(input_data, standard_input, network_graph, timestep_eval, dir)
        self.max_no_clusters = max_no_clusters

    def optimise(self):

        opt_results = []
        for n_components in range(2, self.max_no_clusters + 1):
            gmm = GaussianMixture(
                n_components=n_components, covariance_type='full', tol=0.001, max_iter=100, n_init=1,
                init_params='k-means++'
            )

            labels = gmm.fit_predict(
                self.input_data.to_numpy()
            )

            silhouette, davies_bouldin, calinski_harabasz = self.ml_metrics(labels)
            bic_score = gmm.bic(self.input_data.to_numpy())
            aic_score = gmm.aic(self.input_data.to_numpy())
            ave_density_sd, ave_density_cv = self.ave_density_sd(labels)
            ave_connectivity = self.ave_connectivity(labels)

            opt_results.append(
                (n_components, bic_score, aic_score, silhouette,
                 davies_bouldin, calinski_harabasz, ave_density_sd, ave_density_cv, ave_connectivity)
            )

        opt_results = pd.DataFrame(
            opt_results,
            columns=['n_components', 'bic_score', 'aic_score', 'silhouette', 'davies_bouldin', 'calinski_harabasz',
                     'ave_density_sd', 'ave_density_cv', 'ave_connectivity'
                     ]
        )

        # calculate rate of change for each metric
        for metric in ['bic_score', 'aic_score', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'ave_density_sd',
                       'ave_density_cv', 'ave_connectivity']:
            opt_results[f'{metric}_roc'] = opt_results[metric].diff() / opt_results[metric].shift(1)

        # determine optimal points
        optimal_points = {
            'opt_bic': KneeLocator(opt_results.n_components, opt_results.bic_score, curve='convex',
                                   direction='decreasing', interp_method='polynomial').elbow,
            'opt_aic': KneeLocator(opt_results.n_components, opt_results.aic_score, curve='convex',
                                   direction='decreasing', interp_method='polynomial').elbow,
            'opt_silhouette': opt_results.loc[opt_results.silhouette.idxmax(), 'n_components'],
            'opt_davies_bouldin': opt_results.loc[opt_results.davies_bouldin.idxmin(), 'n_components'],
            'opt_calinski_harabasz': opt_results.loc[opt_results.calinski_harabasz.idxmax(), 'n_components'],
            'opt_ave_density_sd': opt_results.loc[opt_results.ave_density_sd.idxmin(), 'n_components'],
            'opt_ave_density_cv': opt_results.loc[opt_results.ave_density_cv.idxmin(), 'n_components'],
            'opt_ave_connectivity': opt_results.loc[opt_results.ave_connectivity.idxmax(), 'n_components']
        }
        self.plot_optimisation(opt_results, optimal_points)

        self.save_pickle(opt_results, 'gmm_opt_results.pkl')
        self.save_pickle(optimal_points, 'gmm_optimal_points.pkl')

    def plot_optimisation(self, opt_results, optimal_points):
        fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(20, 32), dpi=600)

        plt.yticks(fontsize=7)
        plt.xticks(fontsize=7)

        n_components = opt_results.n_components
        plots = [
            (0, 0, n_components, opt_results.bic_score, 'BIC Score', 'Number of Components', 'bic_score',
             optimal_points['opt_bic']),
            (0, 1, n_components, opt_results.bic_score_roc, 'BIC ROC', 'Number of Components', 'bic_score_roc'),
            (1, 0, n_components, opt_results.aic_score, 'AIC Score', 'Number of Components', 'aic_score',
             optimal_points['opt_aic']),
            (1, 1, n_components, opt_results.aic_score_roc, 'AIC ROC', 'Number of Components', 'aic_score_roc'),
            (2, 0, n_components, opt_results.silhouette, 'Silhouette Score', 'Number of Components', 'silhouette',
             optimal_points['opt_silhouette']),
            (
            2, 1, n_components, opt_results.silhouette_roc, 'Silhouette ROC', 'Number of Components', 'silhouette_roc'),
            (3, 0, n_components, opt_results.davies_bouldin, 'Davies-Bouldin Score', 'Number of Components',
             'davies_bouldin', optimal_points['opt_davies_bouldin']),
            (3, 1, n_components, opt_results.davies_bouldin_roc, 'Davies-Bouldin ROC', 'Number of Components',
             'davies_bouldin_roc'),
            (4, 0, n_components, opt_results.calinski_harabasz, 'Calinski-Harabasz Score', 'Number of Components',
             'calinski_harabasz', optimal_points['opt_calinski_harabasz']),
            (4, 1, n_components, opt_results.calinski_harabasz_roc, 'Calinski-Harabasz ROC', 'Number of Components',
             'calinski_harabasz_roc'),
            (
            5, 0, n_components, opt_results.ave_density_sd, 'Ave. Density SD', 'Number of Components', 'ave_density_sd',
            optimal_points['opt_ave_density_sd']),
            (5, 1, n_components, opt_results.ave_density_sd_roc, 'Ave. Density SD ROC', 'Number of Components',
             'ave_density_sd_roc'),
            (
            6, 0, n_components, opt_results.ave_density_cv, 'Ave. Density CV', 'Number of Components', 'ave_density_cv',
            optimal_points['opt_ave_density_cv']),
            (6, 1, n_components, opt_results.ave_density_cv_roc, 'Ave. Density CV ROC', 'Number of Components',
             'ave_density_cv_roc'),
            (7, 0, n_components, opt_results.ave_connectivity, 'Ave. Connectivity', 'Number of Components',
             'ave_connectivity', optimal_points['opt_ave_connectivity']),
            (7, 1, n_components, opt_results.ave_connectivity_roc, 'Ave. Connectivity ROC', 'Number of Components',
             'ave_connectivity_roc')
        ]

        for plot in plots:
            row, col, x_data, y_data, y_label, x_label, metric, *opt_line = plot
            axs[row, col].plot(x_data, y_data, color='#005293')
            axs[row, col].set_xlabel(x_label, fontsize=8, labelpad=2)
            axs[row, col].set_ylabel(y_label, fontsize=8, labelpad=2)
            axs[row, col].tick_params(axis='both', which='both', labelsize=7, pad=2, top=False, right=False)
            axs[row, col].set_xlim(left=2)
            if opt_line:
                axs[row, col].axvline(x=opt_line[0], color='#a2ad00', linestyle='--')

        # hide the unused axes
        for row in range(8):
            for col in range(2):
                if (row, col) not in [(plot[0], plot[1]) for plot in plots]:
                    axs[row, col].set_visible(False)

        plt.savefig(os.path.join(self.dir, 'gmm_optimisation_plot.png'), bbox_inches='tight')
        plt.close('all')


class KMeansOptimiser(MLOptimiser):
    def __init__(self, input_data, standard_input, network_graph, timestep_eval, dir, max_no_clusters):
        super().__init__(input_data, standard_input, network_graph, timestep_eval, dir)
        self.max_no_clusters = max_no_clusters

    def optimise(self):
        opt_results = []
        for n_clusters in range(2, self.max_no_clusters + 1):
            kmeans = KMeans(
                n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.00001, algorithm='lloyd'
            )

            labels = kmeans.fit_predict(
                self.input_data.to_numpy()
            )

            wcsse = kmeans.inertia_
            silhouette, davies_bouldin, calinski_harabasz = self.ml_metrics(labels)
            ave_density_sd, ave_density_cv = self.ave_density_sd(labels)
            ave_connectivity = self.ave_connectivity(labels)

            opt_results.append(
                (n_clusters, wcsse, silhouette, davies_bouldin, calinski_harabasz, ave_density_sd, ave_density_cv,
                 ave_connectivity)
            )

        opt_results = pd.DataFrame(
            opt_results,
            columns=[
                'n_clusters', 'wcsse', 'silhouette', 'davies_bouldin',
                'calinski_harabasz', 'ave_density_sd', 'ave_density_cv', 'ave_connectivity'
            ]
        )

        # calculate rate of change for each metric
        for metric in ['wcsse', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'ave_density_sd', 'ave_density_cv',
                       'ave_connectivity']:
            opt_results[f'{metric}_roc'] = opt_results[metric].diff() / opt_results[metric].shift(1)

        # determine optimal points
        optimal_points = {
            'opt_wcsse': KneeLocator(opt_results.n_clusters, opt_results.wcsse, curve='convex', direction='decreasing',
                                     interp_method='polynomial').elbow,
            'opt_silhouette': opt_results.loc[opt_results.silhouette.idxmax(), 'n_clusters'],
            'opt_davies_bouldin': opt_results.loc[opt_results.davies_bouldin.idxmin(), 'n_clusters'],
            'opt_calinski_harabasz': opt_results.loc[opt_results.calinski_harabasz.idxmax(), 'n_clusters'],
            'opt_ave_density_sd': opt_results.loc[opt_results.ave_density_sd.idxmin(), 'n_clusters'],
            'opt_ave_density_cv': opt_results.loc[opt_results.ave_density_cv.idxmin(), 'n_clusters'],
            'opt_ave_connectivity': opt_results.loc[opt_results.ave_connectivity.idxmax(), 'n_clusters']
        }
        self.plot_optimisation(opt_results, optimal_points)

        self.save_pickle(opt_results, 'kmeans_opt_results.pkl')
        self.save_pickle(optimal_points, 'kmeans_optimal_points.pkl')

    def plot_optimisation(self, opt_results, optimal_points):
        fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(20, 28), dpi=600)

        plt.yticks(fontsize=7)
        plt.xticks(fontsize=7)

        n_clusters = opt_results.n_clusters
        plots = [
            (0, 0, n_clusters, opt_results.wcsse, 'WCSSE', 'Number of Clusters', 'wcsse', optimal_points['opt_wcsse']),
            (0, 1, n_clusters, opt_results.wcsse_roc, 'WCSSE ROC', 'Number of Clusters', 'wcsse_roc'),
            (1, 0, n_clusters, opt_results.silhouette, 'Silhouette Score', 'Number of Clusters', 'silhouette',
             optimal_points['opt_silhouette']),
            (1, 1, n_clusters, opt_results.silhouette_roc, 'Silhouette ROC', 'Number of Clusters', 'silhouette_roc'),
            (2, 0, n_clusters, opt_results.davies_bouldin, 'Davies-Bouldin Score', 'Number of Clusters',
             'davies_bouldin', optimal_points['opt_davies_bouldin']),
            (2, 1, n_clusters, opt_results.davies_bouldin_roc, 'Davies-Bouldin ROC', 'Number of Clusters',
             'davies_bouldin_roc'),
            (3, 0, n_clusters, opt_results.calinski_harabasz, 'Calinski-Harabasz Score', 'Number of Clusters',
             'calinski_harabasz', optimal_points['opt_calinski_harabasz']),
            (3, 1, n_clusters, opt_results.calinski_harabasz_roc, 'Calinski-Harabasz ROC', 'Number of Clusters',
             'calinski_harabasz_roc'),
            (4, 0, n_clusters, opt_results.ave_density_sd, 'Ave. Density SD', 'Number of Clusters', 'ave_density_sd',
             optimal_points['opt_ave_density_sd']),
            (4, 1, n_clusters, opt_results.ave_density_sd_roc, 'Ave. Density SD ROC', 'Number of Clusters',
             'ave_density_sd_roc'),
            (5, 0, n_clusters, opt_results.ave_density_cv, 'Ave. Density CV', 'Number of Clusters', 'ave_density_cv',
             optimal_points['opt_ave_density_cv']),
            (5, 1, n_clusters, opt_results.ave_density_cv_roc, 'Ave. Density CV ROC', 'Number of Clusters',
             'ave_density_cv_roc'),
            (6, 0, n_clusters, opt_results.ave_connectivity, 'Ave. Connectivity', 'Number of Clusters',
             'ave_connectivity', optimal_points['opt_ave_connectivity']),
            (6, 1, n_clusters, opt_results.ave_connectivity_roc, 'Ave. Connectivity ROC', 'Number of Clusters',
             'ave_connectivity_roc')
        ]

        for plot in plots:
            row, col, x_data, y_data, y_label, x_label, metric, *opt_line = plot
            axs[row, col].plot(x_data, y_data, color='#005293')
            axs[row, col].set_xlabel(x_label, fontsize=8, labelpad=2)
            axs[row, col].set_ylabel(y_label, fontsize=8, labelpad=2)
            axs[row, col].tick_params(axis='both', which='both', labelsize=7, pad=2, top=False, right=False)
            axs[row, col].set_xlim(left=2)
            if opt_line:
                axs[row, col].axvline(x=opt_line[0], color='#a2ad00', linestyle='--')

        # hide the unused axes
        for row in range(7):
            for col in range(2):
                if (row, col) not in [(plot[0], plot[1]) for plot in plots]:
                    axs[row, col].set_visible(False)

        plt.savefig(os.path.join(self.dir, 'kmeans_optimisation_plot.png'), bbox_inches='tight')
        plt.close('all')


class AgglomerativeOptimiser(MLOptimiser):

    def __init__(self, input_data, standard_input, network_graph, timestep_eval, dir, max_no_clusters):
        super().__init__(input_data, standard_input, network_graph, timestep_eval, dir)
        self.max_no_clusters = max_no_clusters

    def optimise(self):
        opt_results = []
        for n_clusters in range(2, self.max_no_clusters + 1):
            connectivity = nx.adjacency_matrix(self.network_graph, nodelist=self.input_data.index)

            agg = AgglomerativeClustering(
                n_clusters=n_clusters, connectivity=connectivity
            )

            labels = agg.fit_predict(
                self.input_data.to_numpy()
            )

            silhouette, davies_bouldin, calinski_harabasz = self.ml_metrics(labels)
            ave_density_sd, ave_density_cv = self.ave_density_sd(labels)
            ave_connectivity = self.ave_connectivity(labels)

            opt_results.append(
                (n_clusters, silhouette, davies_bouldin, calinski_harabasz, ave_density_sd, ave_density_cv,
                 ave_connectivity)
            )

        opt_results = pd.DataFrame(
            opt_results,
            columns=[
                'n_clusters', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'ave_density_sd', 'ave_density_cv',
                'ave_connectivity'
            ]
        )

        # calculate rate of change for each metric
        for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'ave_density_sd', 'ave_density_cv',
                       'ave_connectivity']:
            opt_results[f'{metric}_roc'] = opt_results[metric].diff() / opt_results[metric].shift(1)

        # determine optimal points
        optimal_points = {
            'opt_silhouette': opt_results.loc[opt_results.silhouette.idxmax(), 'n_clusters'],
            'opt_davies_bouldin': opt_results.loc[opt_results.davies_bouldin.idxmin(), 'n_clusters'],
            'opt_calinski_harabasz': opt_results.loc[opt_results.calinski_harabasz.idxmax(), 'n_clusters'],
            'opt_ave_density_sd': opt_results.loc[opt_results.ave_density_sd.idxmin(), 'n_clusters'],
            'opt_ave_density_cv': opt_results.loc[opt_results.ave_density_cv.idxmin(), 'n_clusters'],
            'opt_ave_connectivity': opt_results.loc[opt_results.ave_connectivity.idxmax(), 'n_clusters']
        }
        self.plot_optimisation(opt_results, optimal_points)

        self.save_pickle(opt_results, 'agglomerative_opt_results.pkl')
        self.save_pickle(optimal_points, 'agglomerative_optimal_points.pkl')

    def plot_optimisation(self, opt_results, optimal_points):
        fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(20, 24), dpi=600)

        plt.yticks(fontsize=6)
        plt.xticks(fontsize=6)

        n_clusters = opt_results.n_clusters
        plots = [
            (0, 0, n_clusters, opt_results.silhouette, 'Silhouette Score', 'Number of Clusters', 'silhouette',
             optimal_points['opt_silhouette']),
            (0, 1, n_clusters, opt_results.silhouette_roc, 'Silhouette ROC', 'Number of Clusters', 'silhouette_roc'),
            (1, 0, n_clusters, opt_results.davies_bouldin, 'Davies-Bouldin Score', 'Number of Clusters',
             'davies_bouldin', optimal_points['opt_davies_bouldin']),
            (1, 1, n_clusters, opt_results.davies_bouldin_roc, 'Davies-Bouldin ROC', 'Number of Clusters',
             'davies_bouldin_roc'),
            (2, 0, n_clusters, opt_results.calinski_harabasz, 'Calinski-Harabasz Score', 'Number of Clusters',
             'calinski_harabasz', optimal_points['opt_calinski_harabasz']),
            (2, 1, n_clusters, opt_results.calinski_harabasz_roc, 'Calinski-Harabasz ROC', 'Number of Clusters',
             'calinski_harabasz_roc'),
            (3, 0, n_clusters, opt_results.ave_density_sd, 'Ave. Density SD', 'Number of Clusters', 'ave_density_sd',
             optimal_points['opt_ave_density_sd']),
            (3, 1, n_clusters, opt_results.ave_density_sd_roc, 'Ave. Density SD ROC', 'Number of Clusters',
             'ave_density_sd_roc'),
            (4, 0, n_clusters, opt_results.ave_density_cv, 'Ave. Density CV', 'Number of Clusters', 'ave_density_cv',
             optimal_points['opt_ave_density_cv']),
            (4, 1, n_clusters, opt_results.ave_density_cv_roc, 'Ave. Density CV ROC', 'Number of Clusters',
             'ave_density_cv_roc'),
            (5, 0, n_clusters, opt_results.ave_connectivity, 'Ave. Connectivity', 'Number of Clusters',
             'ave_connectivity', optimal_points['opt_ave_connectivity']),
            (5, 1, n_clusters, opt_results.ave_connectivity_roc, 'Ave. Connectivity ROC', 'Number of Clusters',
             'ave_connectivity_roc')
        ]

        for plot in plots:
            row, col, x_data, y_data, y_label, x_label, metric, *opt_line = plot
            axs[row, col].plot(x_data, y_data, color='#005293')
            axs[row, col].set_xlabel(x_label, fontsize=8, labelpad=2)
            axs[row, col].set_ylabel(y_label, fontsize=8, labelpad=2)
            axs[row, col].tick_params(axis='both', which='both', labelsize=7, pad=2, top=False, right=False)
            axs[row, col].set_xlim(left=2)
            if opt_line:
                axs[row, col].axvline(x=opt_line[0], color='#a2ad00', linestyle='--')

        # hide the unused axes
        for row in range(6):
            for col in range(2):
                if (row, col) not in [(plot[0], plot[1]) for plot in plots]:
                    axs[row, col].set_visible(False)

        plt.savefig(os.path.join(self.dir, 'agglomerative_optimisation_plot.png'), bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':

    # read the Excel file for params
    df = pd.read_excel(f'FineTuning.xlsx')
    params_list = [
        (
            row['id'],
            row['scr'],
            row['feature'],
            row['agg_interval'],
            row['timestep'],
            row['timestep_eval'],
            row['geo_weight'],
            row['traffic_weight']
        )
        for index, row in df.iterrows()
    ]

    # execute rows sequentially
    for params in params_list:
        id, scr, feature, agg_interval, timestep, timestep_eval, geo_weight, traffic_weight = params

        # load necessary files
        network_graph = load_pickle(r'data\{}\network\network_graph.pkl'.format(scr))
        standard_input = load_pickle(r'data\{}\input\standard_input.pkl'.format(scr))

        # define result directory
        result_dir = r'results\{}\fine_tuning_{}_{}_{}_{}_{}_{}'.format(
                                                              scr,
                                                                    feature,
                                                                    agg_interval,
                                                                    timestep,
                                                                    timestep_eval,
                                                                    geo_weight,
                                                                    traffic_weight
        )
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        # generate input
        ml_input_path = r'data\{}\input\ml_input_{}_{}_{}_{}.pkl'.format(
                                                            scr, agg_interval, feature, traffic_weight, geo_weight
                                                        )
        if not os.path.exists(ml_input_path):
            data_processor = DataProcessor(r'data\{}\input'.format(scr), standard_input)
            ml_input = data_processor.generate_ML_input(agg_interval=agg_interval,
                                                        feature=feature,
                                                        traffic_feature_weight=traffic_weight,
                                                        geo_feature_weight=geo_weight)
            save_pickle(ml_input, os.path.dirname(ml_input_path), os.path.basename(ml_input_path))
        else:
            ml_input = load_pickle(ml_input_path)

        if timestep != 'all':
            ml_input = ml_input.loc[:, ['x', 'y', timestep]]

        # optimise
        optimisers = [
            DBSCANOptimiser(ml_input, standard_input, network_graph, timestep_eval, result_dir, max_min_samples=100),
            GMMOptimiser(ml_input, standard_input, network_graph, timestep_eval, result_dir, max_no_clusters=100),
            KMeansOptimiser(ml_input, standard_input, network_graph, timestep_eval, result_dir, max_no_clusters=100),
            AgglomerativeOptimiser(ml_input, standard_input, network_graph, timestep_eval, result_dir,
                                   max_no_clusters=100),
        ]
        for optimiser in optimisers:
            optimiser.optimise()