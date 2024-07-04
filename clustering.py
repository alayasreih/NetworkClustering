import time
import pandas as pd
import networkx as nx
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from snake_method import parallel_run_snake, parallel_estimate_similarity, symnmf


class ClusteringMethods:
    def __init__(self, network_graph):
        self.network_graph = network_graph

    @staticmethod
    def format_labels(labels, index):
        label_counts = pd.Series(labels).value_counts()
        label_mapping = {label: new_label for new_label, label in enumerate(label_counts.index)}
        new_labels = [label_mapping[label] for label in labels]

        return pd.DataFrame({'edge_id': index, 'label': new_labels})


class ML(ClusteringMethods):
    def __init__(self, network_graph, input_data):
        super().__init__(network_graph)
        self.input_data = input_data

    def kmeans(self, n_clusters=5):
        tb = time.perf_counter()

        kmeans = KMeans(
            n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.00001, algorithm='lloyd'
        ).fit_predict(
            self.input_data.to_numpy()
        )

        te = time.perf_counter()
        run_time = round(te - tb)

        return self.format_labels(kmeans, self.input_data.index), run_time

    def dbscan(self, eps=0.06, min_samples=50):
        tb = time.perf_counter()

        dbscan = DBSCAN(
            eps=eps, min_samples=min_samples
        ).fit(
            self.input_data.to_numpy()
        )

        te = time.perf_counter()
        run_time = round(te - tb)

        return self.format_labels(dbscan.labels_, self.input_data.index), run_time

    def gmm(self, n_components=5):
        tb = time.perf_counter()

        gmm = GaussianMixture(
            n_components=n_components, covariance_type='full', tol=0.001, max_iter=100, n_init=1,
            init_params='k-means++'
        )

        gmm_model = gmm.fit_predict(
            self.input_data.to_numpy()
        )

        te = time.perf_counter()
        run_time = round(te - tb)

        return self.format_labels(gmm_model, self.input_data.index), run_time

    def agglomerative(self, n_clusters=5):
        tb = time.perf_counter()

        connectivity = nx.adjacency_matrix(self.network_graph, nodelist=self.input_data.index)

        agg = AgglomerativeClustering(
            n_clusters=n_clusters, connectivity=connectivity
        ).fit_predict(
            self.input_data.to_numpy()
        )

        te = time.perf_counter()
        run_time = round(te - tb)

        return self.format_labels(agg, self.input_data.index), run_time


class Snake(ClusteringMethods):
    def __init__(self, network_graph, input_data, dir):
        super().__init__(network_graph)
        self.input_data = input_data
        self.dir = dir

    def snake(self, n_clusters=6):
        tb = time.perf_counter()

        adjacency = nx.to_dict_of_lists(self.network_graph, nodelist=self.input_data.keys())

        snakes_lists = parallel_run_snake(
            adjacency, self.input_data, self.dir
        )

        similarity_matrix_path, similarity_matrix_index = parallel_estimate_similarity(
            snakes_lists, self.dir
        )

        labels, no_iterations, objective_value = symnmf(
            similarity_matrix_path, n_clusters
        )

        te = time.perf_counter()
        run_time = round(te - tb)

        return self.format_labels(labels, similarity_matrix_index), run_time


class Method3:
    """Method from Lukas Ambühl: https://doi.org/10.1177/0361198119843264"""
    pass


class Method4:
    """Method from Mehdi Keyvan-Ekbatani: https://doi.org/10.1111/mice.12895"""
    pass
