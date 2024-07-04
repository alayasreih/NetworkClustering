import os
import time
import shelve
import operator
import functools
import itertools
import numpy as np
import pandas as pd
import multiprocessing


# ---------------------------------------------------------------------------------------------------------------------

def all_slices(seq):
    slices = itertools.starmap(slice, itertools.combinations(range(len(seq) + 1), 2))
    return map(operator.getitem, itertools.repeat(seq, len(seq)), slices)


def len_intersection(seq1, seq2):
    return len(set(seq1).intersection(seq2))


def similarity(slices_list_1, slices_list_2):
    return sum(map(lambda slice_1, slice_2: len_intersection(slice_1, slice_2), slices_list_1, slices_list_2))


# ---------------------------------------------------------------------------------------------------------------------

def generate_edge_snake(edge, network_graph, input):
    print(f'start snake for {edge}', flush=True)
    tb = time.perf_counter()

    # fixed arg
    edges = [*input]
    edges_count = len(edges)

    # initiate all lists
    snake = []
    snake_mean = []
    snake_variance = []

    # start snake for the edge
    snake.append(edge)
    snake_mean.append(operator.itemgetter(edge)(input))
    snake_variance.append(0)

    while len(snake) < round(edges_count / 4):
        snake_adjacent = operator.itemgetter(*snake)(network_graph)

        snake_adjacent_flat = set(functools.reduce(operator.iconcat, snake_adjacent, [])) if not len(snake) == 1 \
            else set(snake_adjacent)

        candidates = snake_adjacent_flat.intersection(edges).difference(snake)

        if len(candidates) == 0:
            print(f'something happened: this edge is not connected any more {edge}', flush=True)
            break

        else:
            record = []
            for candidate in candidates:
                candidate_density = operator.itemgetter(edge)(input)

                # Welford's method for updating the variance and running mean for single pass
                possible_snake_mean = ((len(snake) * snake_mean[-1]) + candidate_density) / (len(snake) + 1)

                possible_snake_variance = ((len(snake) * snake_variance[-1]) + (
                        candidate_density - possible_snake_mean) * (candidate_density - snake_mean[-1])) / (
                                                  len(snake) + 1)

                # criterion: the change in variance to the change in mean
                possible_mean_change = possible_snake_mean - snake_mean[-1]
                possible_variance_change = possible_snake_variance - snake_variance[-1]
                criterion = 0 if possible_mean_change == 0 else (possible_variance_change / possible_mean_change)

                record.append({'candidate': candidate, 'possible_snake_mean': possible_snake_mean,
                               'possible_snake_variance': possible_snake_variance, 'criterion': criterion})

            # best candidate
            best_candidate = min(record, key=lambda x: x['criterion'])

            snake.append(best_candidate['candidate'])
            snake_mean.append(best_candidate['possible_snake_mean'])
            snake_variance.append(best_candidate['possible_snake_variance'])

    te = time.perf_counter()
    print(f'No. of edges in edge {edge} snake is {len(snake)}, run time is {round(te - tb)}', flush=True)

    return edge, snake, snake_mean, snake_variance


def parallel_run_snake(network_graph, input, dir):
    edges = [*input]
    edges_count = len(edges)

    snakes_path = os.path.join(dir, 'snakes.db').replace(os.sep, '/')
    snakes_mean_path = os.path.join(dir, 'snakes_mean.db').replace(os.sep, '/')
    snakes_variance_path = os.path.join(dir, 'snakes_variance.db').replace(os.sep, '/')

    t_start = time.perf_counter()

    with multiprocessing.Pool(32) as pool:
        for edge, snake, snake_mean, snake_variance in pool.imap(
                functools.partial(generate_edge_snake, network_graph=network_graph, input=input), edges):
            snakes = shelve.open(snakes_path)
            snakes[edge] = snake
            edges_done = len(snakes)
            snakes.close()

            snakes_mean = shelve.open(snakes_mean_path)
            snakes_mean[edge] = snake_mean
            snakes_mean.close()

            snakes_variance = shelve.open(snakes_variance_path)
            snakes_variance[edge] = snake_variance
            snakes_variance.close()

            print(f'{edges_done} out of {edges_count} snakes done so far')

        pool.close()
        pool.join()

    t_stop = time.perf_counter()
    print(f'total run time {round(t_stop - t_start)}')

    return shelve.open(snakes_path)


def estimate_edges_similarity(i, snakes):
    tb = time.perf_counter()
    print(f'start similarity estimation for {i[0]}', flush=True)

    similarity_list = []

    edge_i = i[0]
    snake_i = i[1]

    for j in snakes.items():
        edge_j = j[0]
        snake_j = j[1]

        common_edges = set(snake_i).intersection(snake_j)
        similarity_value = sum(
            [(min(len(snake_i), len(snake_j)) - max(snake_i.index(h) + 1, snake_j.index(h) + 1) + 1) for h in
             common_edges])

        similarity_list.append({'i': edge_i, 'j': edge_j, 'similarity': similarity_value})

    te = time.perf_counter()
    print(f'similarity estimation for {i[0]} is done, run time is {round(te - tb)}', flush=True)

    return edge_i, similarity_list


def parallel_estimate_similarity(snakes, dir):
    edges = [*snakes]
    edges_count = len(edges)

    similarity_lists_path = os.path.join(dir, 'similarity_lists.db').replace(os.sep, '/')

    print(f'Similarity estimation starts')
    t_start = time.perf_counter()

    with multiprocessing.Pool(32) as pool:
        for edge, similarity_list in pool.imap(functools.partial(estimate_edges_similarity, snakes=snakes),
                                               snakes.items()
                                               ):

            similarity_lists = shelve.open(similarity_lists_path)
            similarity_lists[edge] = similarity_list
            edges_done = len(similarity_lists)

            similarity_lists.close()

            print(f'{edges_done} out of {edges_count} edges done so far')

        pool.close()
        pool.join()

    t_stop = time.perf_counter()
    print(f'Similarity estimation total run time {round(t_stop - t_start)}')

    # reformat the list
    similarity_lists = shelve.open(similarity_lists_path)
    similarity_lists_flat = [l for _, v in similarity_lists.items() for l in v]
    similarity_lists.close()

    # save similarity values as df in long and wide format
    matrix_long = pd.DataFrame(similarity_lists_flat)
    matrix_wide = matrix_long.pivot(index='i', columns='j', values='similarity').astype(float)
    matrix_wide_index = matrix_wide.index.to_series(index=None, name='edge_id').reset_index(drop=True)

    # save to directory
    matrix_wide_path = os.path.join(dir, 'similarity_matrix.csv').replace(os.sep, '/')
    matrix_wide.to_csv(matrix_wide_path, na_rep='0', header=False, index=False)

    index_path = os.path.join(dir, 'similarity_matrix_index.csv').replace(os.sep, '/')
    matrix_wide_index.to_csv(index_path, header=True, index=False)

    return matrix_wide_path, matrix_wide_index


def symnmf(matrix_wide_path, n_clusters):
    """
    Call symnmf_anls function from Matlab; where the input A is an N-by-N symmetric matrix containing pairwise
    similarity values, and the output H is an N-by-K non-negative matrix indicating clustering assignment.The output 'H'
    is a clustering indicator matrix, and clustering assignments are indicated by the largest entry in each row of 'H'.

    GitHub: https://github.com/dakuang/symnmf

    To call built-in or scripted Matlab functions in python; one need to have a copy of MATLAB installed in the system,
    and to install MatLab engin API for python.

    MathWorks Documentation: https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    """

    matlab_subdir = os.path.join(os.getcwd(), 'SymNMF')

    import matlab.engine

    eng = matlab.engine.start_matlab()
    eng.addpath(matlab_subdir);

    matlab_similarity_matrix = eng.readmatrix(matrix_wide_path);
    matlab_similarity_matrix = eng.double(matlab_similarity_matrix);
    [H, iter, obj] = eng.symnmf_anls(matlab_similarity_matrix, float(n_clusters), nargout=3);

    eng.exit()

    return np.argmax(np.array(H), axis=1), iter, obj
