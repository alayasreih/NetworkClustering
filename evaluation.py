import random as rd
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import MultiPoint
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.utils import random
import alphashape
from sklearn.preprocessing import MinMaxScaler


class Evaluation:
    def __init__(self, labels, standard_input):
        self.labels = labels
        self.standard_input = standard_input
        self.labelled_input = standard_input.merge(labels, on='edge_id', how='inner')

    @staticmethod
    def percent_error(total_actual_val, total_estimated_val):
        return ((total_estimated_val - total_actual_val) / total_actual_val) * 100 if not total_actual_val == 0 else 0

    @staticmethod
    def rmsne(actual_val: np.ndarray, estimated_val: np.ndarray) -> float:
        """Root Mean Squared Normalized Error"""
        diff = np.subtract(estimated_val, actual_val)
        percent_error = np.divide(diff, actual_val)
        squared_percent_error = np.square(percent_error)
        msne = squared_percent_error.mean()
        return np.sqrt(msne)

    @staticmethod
    def mane(actual_val: np.ndarray, estimated_val: np.ndarray) -> float:
        """Mean Absolute Normalized Error"""
        diff = np.subtract(estimated_val, actual_val)
        abs_diff = np.abs(diff)
        abs_percent_error = np.divide(abs_diff, actual_val)
        return abs_percent_error.mean()


class Stats(Evaluation):
    def get_stats(self):
        stats = []
        for (timestep, label), temp in self.labelled_input.groupby(['timestep', 'label']):
            speed = DescrStatsW(temp.speed, weights=temp.sampled_seconds)
            density = DescrStatsW(temp.density, weights=temp.length)
            flow = DescrStatsW(temp.flow, weights=temp.length)

            stats.append((timestep, label,
                          speed.mean, speed.std, speed.var,
                          density.mean, density.std, density.var,
                          flow.mean, flow.var, flow.std))

        return pd.DataFrame(stats, columns=['timestep', 'label',
                                            'speed_mean', 'speed_std', 'speed_var',
                                            'density_mean', 'density_std', 'density_var',
                                            'flow_mean', 'flow_var', 'flow_std'])


class MFD(Evaluation):
    def mfd_estimation(self):
        mfd = []
        for (timestep, label), temp in self.labelled_input.groupby(['timestep', 'label']):
            speed = np.average(temp.speed, weights=temp.sampled_seconds) if not temp.sampled_seconds.all() == 0 else 0
            density = np.average(temp.density, weights=temp.length)
            flow = np.average(temp.flow, weights=temp.length)
            mfd.append((timestep, label, speed, density, flow))
        mfd = pd.DataFrame(mfd, columns=['timestep', 'label', 'speed', 'density', 'flow'])
        return mfd

    def mfd_resampling(self, p_sample, n_combinations):
        resampled_mfd = []
        for label, cluster_meas in self.labelled_input.groupby('label'):
            n_population = cluster_meas.edge_id.nunique()
            n_samples = int(n_population * (p_sample / 100))

            population = cluster_meas.edge_id.unique().tolist()
            population_subsets = []
            seen_subsets = set()
            while len(population_subsets) < n_combinations:
                subsets_indices = tuple(sorted(random.sample_without_replacement(n_population, n_samples)))
                if subsets_indices not in seen_subsets:
                    subset = [population[n] for n in subsets_indices]
                    population_subsets.append(subset)
                    seen_subsets.add(subsets_indices)
                else:
                    continue

            for subset in population_subsets:
                subset_meas = cluster_meas.loc[cluster_meas.edge_id.isin(subset)]
                for timestep, temp in subset_meas.groupby('timestep'):
                    speed = np.average(temp.speed, weights=temp.sampled_seconds) if not temp.sampled_seconds.all() == 0 else 0
                    density = np.average(temp.density, weights=temp.length)
                    flow = np.average(temp.flow, weights=temp.length)
                    resampled_mfd.append((label, timestep, flow, density, speed))
        resampled_mfd = pd.DataFrame(resampled_mfd, columns=['label', 'timestep', 'flow', 'density', 'speed'])

        return resampled_mfd

    def evaluate_mfd(mfd):
        alpha = 0.95  # parameter to adjust the alpha shape
        eval_results = []

        for label, temp in mfd.groupby('label'):
            points = temp[['density', 'flow']].values
            points_normalized = MinMaxScaler().fit_transform(points)  # normalize the points between 0 and 1
            alpha_shape = alphashape.alphashape(points_normalized, alpha)
            alpha_shape_area = alpha_shape.area
            eval_results.append((label, points_normalized, alpha_shape, alpha_shape_area))
        eval_results = pd.DataFrame(eval_results,
                                    columns=['label', 'points_normalized', 'alpha_shape', 'alpha_shape_area'])

        return eval_results


class Topology(Evaluation):
    def __init__(self, standard_input, labels, network_graph):
        super().__init__(standard_input, labels)
        self.network_graph = network_graph

    def get_intra_connectivity(self):

        labelled_attr = self.labelled_input[['edge_id', 'lane_id', 'label', 'length']].drop_duplicates(ignore_index=True)

        connectivities = []
        for _, temp in labelled_attr.groupby('label'):
            cluster_length = temp.length.sum()
            cluster_subgraph = self.network_graph.subgraph(temp.edge_id.unique())
            cluster_largest_cc = max(nx.connected_components(cluster_subgraph), key=len)
            cluster_largest_cc_length = temp[temp.edge_id.isin(cluster_largest_cc)].length.sum()
            connectivity = cluster_largest_cc_length / cluster_length
            connectivities.append(connectivity)

        return

    def get_inter_connectivity(self):
        pass

    def get_compactness(self):
        compactness_data = []
        for label, temp in self.labelled_input.groupby('label'):
            convex_hull_polygon = MultiPoint(temp.geometry.unique()).convex_hull
            hull_area = convex_hull_polygon.area
            hull_perimeter = convex_hull_polygon.length
            isoperimetric_quotient = (4 * np.pi * hull_area) / (hull_perimeter ** 2) if hull_area > 0 else 0
            compactness_measure = (hull_perimeter ** 2) / hull_area if hull_area > 0 else 0
            boundary_complexity = hull_perimeter / np.sqrt(hull_area) if hull_area > 0 else 0
            compactness_data.append(
                (label, hull_area, hull_perimeter, isoperimetric_quotient, compactness_measure, boundary_complexity))

        return pd.DataFrame(compactness_data, columns=['label', 'hull_area', 'hull_perimeter', 'isoperimetric_quotient',
                                                       'compactness_measure', 'boundary_complexity'])


class TravelTime(Evaluation):
    def __init__(self, standard_input, labels, vehroute):
        super().__init__(standard_input, labels)
        self.vehroute = vehroute
        self.traveltimes = self.get_traveltimes()

    def get_traveltimes(self):
        travel_times = []
        for _, temp in self.labelled_input.groupby(['timestep', 'label']):
            # temp['travel_time'] = temp.length / temp.speed
            temp['estimated_speed'] = DescrStatsW(temp.speed, weights=temp.sampled_seconds).mean
            temp['estimated_travel_time'] = temp.length / temp.estimated_speed
            travel_times.append(temp[['timestep', 'label', 'edge_id', 'lane_id', 'speed', 'estimated_speed',
                                      'estimated_travel_time']])

        return pd.concat(travel_times)

    def vehroute_traveltime(self, sampling_percentage):
        no_samples = round((sampling_percentage / 100) * self.vehroute.veh_id.nunique())
        sample = rd.sample(list(self.vehroute.veh_id.unique()), no_samples)

        sample_vehroute_error = self.vehroute.loc[self.vehroute.veh_id.isin(sample)]
        sample_vehroute_error = sample_vehroute_error.loc[~(sample_vehroute_error == -1).any(axis=1)]
        sample_vehroute_error['timestep'] = (sample_vehroute_error['entry_time'] // 300) * 300
        sample_vehroute_error = sample_vehroute_error.merge(self.traveltimes, on=['timestep', 'edge_id'], how='inner')
        sample_vehroute_error['percentage_error'] = sample_vehroute_error.apply(
            lambda x: self.percent_error(x['travel_time'], x['estimated_travel_time']), axis=1)

        total_traveltimes = sample_vehroute_error.groupby('veh_id')['travel_time'].sum()
        total_estimated_traveltimes = sample_vehroute_error.groupby('veh_id')['estimated_travel_time'].sum()

        rmsne_val = self.rmsne(total_traveltimes.values, total_estimated_traveltimes.values)
        mane_val = self.mane(total_traveltimes.values, total_estimated_traveltimes.values)

        return sample_vehroute_error, rmsne_val, mane_val
