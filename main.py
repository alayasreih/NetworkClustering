import os
import ast
import pandas as pd
from input import DataProcessor
from clustering import ML, Snake
from evaluation import Stats, MFD, TravelTime
from aux_functions import load_pickle, save_pickle
from visualisation import plot_clusters, plot_mfd, plot_density, plot_mfd_aval, plot_traveltime_eval
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # %% ML clustering

    # read the Excel file for params
    df = pd.read_excel('Clustering.xlsx').assign(
        hyper_parameters=lambda df: df['hyper_parameters'].apply(lambda x: tuple(ast.literal_eval(x)))
    )

    params_list = [
        (
            row['id'],
            row['scr'],
            row['feature'],
            row['agg_interval'],
            row['timestep'],
            row['geo_weight'],
            row['traffic_weight'],
            row['hyper_parameters'],
        )
        for index, row in df.iterrows()
    ]

    # execute rows sequentially
    for params in params_list:
        id, scr, feature, agg_interval, timestep, geo_weight, traffic_weight, hyper_parameters = params
        n_components_gmm, n_clusters_kmeans, n_clusters_agglomerative, min_samples, eps = hyper_parameters
        print(id, scr, feature, agg_interval, timestep, geo_weight, traffic_weight, hyper_parameters)

        # load necessary files
        network_graph = load_pickle(r'data\{}\network\network_graph.pkl'.format(scr))
        standard_input = load_pickle(r'data\{}\input\standard_input.pkl'.format(scr))
        vehroute = load_pickle(r'data\{}\output\output_vehroute.pkl'.format(scr))

        # define result directory
        result_dir = r'results\{}\run_{}_{}_{}_{}_{}_{}'.format(
            scr,
            feature,
            agg_interval,
            timestep,
            geo_weight,
            traffic_weight,
            hyper_parameters
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
            'gmm': ml_clustering.gmm(n_components=n_components_gmm),
            'dbscan': ml_clustering.dbscan(eps=eps, min_samples=min_samples),
            'kmeans': ml_clustering.kmeans(n_clusters=n_clusters_kmeans),
            'agglomerative': ml_clustering.agglomerative(n_clusters=n_clusters_agglomerative)
        }

        # evaluate and save results
        for method, (result, runtime) in results.items():
            print(f'{method} runtime: {runtime}')
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
            save_pickle(mfd_eval, result_dir, f'{method}_mfd_eval.pkl')
            save_pickle(mfd_resampled, result_dir, f'{method}_mfd_resampled.pkl')
            save_pickle(mfd_resampled_eval, result_dir, f'{method}_mfd_resampled_eval.pkl')
            save_pickle(sample_vehroute_error, result_dir, f'{method}_traveltime_eval.pkl')

    #%% snake clustering
    method = 'snake'

    scr = 'Zurich'
    agg_interval = 300
    feature = 'density'
    timestep = 63300
    n_clusters = 6

    # define directory
    result_dir = r'results\{}\run_snake_{}_{}_{}'.format(scr,
                                                         agg_interval,
                                                         feature,
                                                         timestep
                                                         )
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # load files
    network_graph = load_pickle(r'data\{}\network\network_graph.pkl'.format(scr))
    standard_input = load_pickle(r'data\{}\input\standard_input.pkl'.format(scr))
    vehroute = load_pickle(r'data\{}\output\output_vehroute.pkl'.format(scr))

    # generate input
    data_processor = DataProcessor(r'data\{}\input'.format(scr), standard_input)
    snake_input = data_processor.generate_snake_input(agg_interval=agg_interval, feature=feature)[timestep]

    # get results
    snake_clustering = Snake(network_graph, snake_input, dir=result_dir)
    result, runtime = snake_clustering.snake(n_clusters=n_clusters)
    print(f'{method} runtime: {runtime}')

    stats = Stats(result, standard_input).get_stats()
    mfd = MFD(result, standard_input).mfd_estimation()
    mfd_eval = MFD.evaluate_mfd(mfd)
    mfd_resampled = MFD(result, standard_input).mfd_resampling(n_combinations=500, p_sample=60)
    mfd_resampled_eval = MFD.evaluate_mfd(mfd_resampled)
    sample_vehroute_error, rmsne_val, mane_val = TravelTime(standard_input, result,
                                                            vehroute).vehroute_traveltime(
        sampling_percentage=25)

    plot_clusters(standard_input, result, os.path.join(result_dir, f'{method}_clusters_{n_clusters}.png'))
    plot_density(stats, os.path.join(result_dir, f'{method}_density_timeseries_{n_clusters}.png'))
    plot_mfd(mfd, os.path.join(result_dir, f'{method}_mfd_{n_clusters}.png'))
    plot_mfd_aval(mfd_eval, os.path.join(result_dir, f'{method}_mfd_eval_{n_clusters}.png'))
    plot_mfd(mfd_resampled, os.path.join(result_dir, f'{method}_mfd_resampled_{n_clusters}.png'))
    plot_mfd_aval(mfd_resampled_eval, os.path.join(result_dir, f'{method}_mfd_resampled_eval_{n_clusters}.png'))
    plot_traveltime_eval(sample_vehroute_error, os.path.join(result_dir, f'{method}_traveltime_eval_{n_clusters}.png'))

    save_pickle(result, result_dir, f'{method}_labels_{n_clusters}.pkl')
    save_pickle(stats, result_dir, f'{method}_stats_{n_clusters}.pkl')
    save_pickle(mfd, result_dir, f'{method}_mfd_{n_clusters}.pkl')
    save_pickle(mfd_eval, result_dir, f'{method}_mfd_eval_{n_clusters}.pkl')
    save_pickle(mfd_resampled, result_dir, f'{method}_mfd_resampled_{n_clusters}.pkl')
    save_pickle(mfd_resampled_eval, result_dir, f'{method}_mfd_resampled_eval_{n_clusters}.pkl')
    save_pickle(sample_vehroute_error, result_dir, f'{method}_traveltime_eval_{n_clusters}.pkl')



