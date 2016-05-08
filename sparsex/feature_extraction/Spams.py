import spams
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class Spams(object):
    
    def __init__(self, model_filename=None, **kwargs):
        raise NotImplementedError

    def save_model(self, filename):
        raise NotImplementedError

    def load_model(self, filename):
        raise NotImplementedError

    def learn_dictionary(self, whitened_patches):
        raise NotImplementedError

    def get_dictionary(self):        
        raise NotImplementedError

    def get_sparse_features(self, whitened_patches):
        raise NotImplementedError

    def get_sign_split_features(self, sparse_features):
        raise NotImplementedError

    def get_pooled_features(self, input_feature_map, filter_size):
        raise NotImplementedError

    def get_pooled_features_from_whitened_patches(self, whitened_patches, filter_size):
        raise NotImplementedError
