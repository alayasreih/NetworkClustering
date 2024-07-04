import os
import pickle


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def save_pickle(obj, dir, file_name):
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
    return