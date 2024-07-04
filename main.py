import os
import pandas as pd
from input import DataProcessor
from clustering import ML, Snake
from evaluation import Stats, MFD, TravelTime
from aux_functions import load_pickle, save_pickle
from visualisation import plot_clusters, plot_mfd, plot_density, plot_mfd_aval, plot_traveltime_eval
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # read the Excel file for params
    df = pd.read_excel(f'Clustering.xlsx')
    params_list = [
        (
            row['id'],
            row['scr'],
            row['feature'],
            row['agg_interval'],
            row['timestep'],
            row['geo_weight'],
            row['traffic_weight'],
            row['no_clusters'],
            row['eps'],
            row['min_samples']
        )
        for index, row in df.iterrows()
    ]

    # execute rows sequentially
    for params in params_list:
        id, scr, feature, agg_interval, timestep, geo_weight, traffic_weight, no_clusters, eps, min_samples = params

        # load necessary files
        network_graph = load_pickle(r'data\{}\network\network_graph.pkl'.format(scr))
        standard_input = load_pickle(r'data\{}\input\standard_input.pkl'.format(scr))
        vehroute = load_pickle(r'data\{}\output\output_vehroute.pkl'.format(scr))

        # define result directory
        result_dir = r'results\{}\run_{}_{}_{}_{}_{}_{}_{}'.format(
                                                             scr,
                                                                   feature,
                                                                   agg_interval,
                                                                   timestep,
                                                                   geo_weight,
                                                                   traffic_weight,
                                                                   no_clusters,
                                                                   eps,
                                                                   min_samples
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
        else:
            ml_input = load_pickle(ml_input_path)

        if not timestep == 'all':
            ml_input = ml_input.loc[:, ['x', 'y', timestep]]

        # get results
        ml_clustering = ML(network_graph, ml_input)

        results = {
            'gmm': ml_clustering.gmm(n_components=no_clusters),
            'dbscan': ml_clustering.dbscan(eps=eps, min_samples=min_samples),
            'kmeans': ml_clustering.kmeans(n_clusters=no_clusters),
            'agglomerative': ml_clustering.agglomerative(n_clusters=no_clusters)
        }

        # evaluate and save results
        for method, (result, runtime) in results.items():
            stats = Stats(result, standard_input).get_stats()
            mfd = MFD(result, standard_input).mfd_estimation()
            mfd_eval = MFD.evaluate_mfd(mfd)
            mfd_resampled = MFD(result, standard_input).mfd_resampling(n_combinations=500, p_sample=60)
            mfd_resampled_eval = MFD.evaluate_mfd(mfd_resampled)
            sample_vehroute_error, rmsne_val, mane_val = TravelTime(standard_input, result,
                                                                    vehroute).vehroute_traveltime(
                sampling_percentage=25)

            plot_clusters(standard_input, result, os.path.join(result_dir, f'{method}_clusters.png'))
            plot_density(stats, os.path.join(result_dir, f'{method}_density_timeseries.png'))
            plot_mfd(mfd, os.path.join(result_dir, f'{method}_mfd.png'))
            plot_mfd_aval(mfd_eval, os.path.join(result_dir, f'{method}_mfd_eval.png'))
            plot_mfd(mfd_resampled, os.path.join(result_dir, f'{method}_mfd_resampled.png'))
            plot_mfd_aval(mfd_resampled_eval, os.path.join(result_dir, f'{method}_mfd_resampled_eval.png'))
            plot_traveltime_eval(sample_vehroute_error, os.path.join(result_dir, f'{method}_traveltime_eval.png'))

            save_pickle(result, result_dir, f'{method}_labels.pkl')
            save_pickle(stats, result_dir, f'{method}_stats.pkl')
            save_pickle(mfd, result_dir, f'{method}_mfd.pkl')
            save_pickle(mfd_resampled, result_dir, f'{method}_mfd_resampled.pkl')
            save_pickle(mfd_eval, result_dir, f'{method}_mfd_eval.pkl')
            save_pickle(mfd_resampled_eval, result_dir, f'{method}_mfd_resampled_eval.pkl')
            save_pickle(sample_vehroute_error, result_dir, f'{method}_traveltime_eval.pkl')

    # %% snake clustering
    #
    # # define directory
    # result_dir = r'results\{}\run_snake_{}_{}_{}_{}'.format(scr,
    #                                                           agg_interval,
    #                                                           feature,
    #                                                           timestep,
    #                                                           no_clusters)
    # if not os.path.isdir(result_dir):
    #     os.makedirs(result_dir)
    #
    # # load files
    # network_graph = load_pickle(r'data\{}\network\network_graph.pkl'.format(scr))
    # standard_input = load_pickle(r'data\{}\input\standard_input.pkl'.format(scr))
    #
    # # generate input
    # snake_input = data_processor.generate_snake_input(agg_interval=agg_interval, feature=feature)[timestep]
    #
    # # get results
    # snake_clustering = Snake(network_graph, snake_input, dir=result_dir)
    # snake_result, snake_runtime = snake_clustering.snake(snake_input, no_clusters=no_clusters)
    # dump_pickle(snake_result, result_dir, f'snake_result_{timestep}_{no_clusters}.pkl')
    #
    # # verbose
    # print('Snake runtime: ', snake_runtime)
