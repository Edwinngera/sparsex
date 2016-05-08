from SklearnDL import SklearnDL
from ..tests.preprocessing_test import test_whitening
from ..customutils import customutils
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class SparseCoding(object):
    
    SKLEARN_DL = "SklearnDL"
    
    def __init__(self, library_name=SKLEARN_DL, model_filename=None, **kwargs):
        if library_name == SparseCoding.SKLEARN_DL:
            self.library = SklearnDL(model_filename, **kwargs)
        else:
            self.library = SklearnDL(model_filename, **kwargs)

    def save_model(self, filename):
        return self.library.save_model(filename)

    def load_model(self, filename):
        return self.library.load_model(filename)

    def learn_dictionary(self, whitened_patches):
        return self.library.learn_dictionary(whitened_patches)

    def get_dictionary(self):
        return self.library.get_dictionary()

    def get_sparse_features(self, whitened_patches):
        return self.library.get_sparse_features(whitened_patches)

    def get_sign_split_features(self, sparse_features):
        return self.library.get_sign_split_features(sparse_features)

    def get_pooled_features(self, input_feature_map, filter_size=(19,19)):
        # need to determine if there must be a default value for filter_size
        return self.library.get_pooled_features(input_feature_map, filter_size)

    def get_pooled_features_from_whitened_patches(self, whitened_patches, filter_size=(19,19)):
        # need to determine if there must be a default value for filter_size
        return self.library.get_pooled_features_from_whitened_patches(whitened_patches, filter_size)



if __name__ == "__main__":
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    
    # get whitened patches
    whitened_patches = test_whitening(image_filename, False, False)

    # create sparse coding object
    sparse_coding = SparseCoding()

    # learn dictionary on whitened patches
    sparse_coding.learn_dictionary(whitened_patches[:100]) # making sure the model has a trained dictionary
    
    print "dictionary Shape :"
    print sparse_coding.get_dictionary().shape

    # get sparse code
    sparse_features = sparse_coding.get_sparse_features(whitened_patches)

    # get feature sign split
    sign_split_features = sparse_coding.get_sign_split_features(sparse_features)

    # get pooled features
    pooled_features = sparse_coding.get_pooled_features(input_feature_map=sign_split_features)

    print "dictionary Shape :"
    print sparse_coding.get_dictionary().shape

    print "sparse features shape :"
    print sparse_features.shape

    print "sign split features shape :"
    print sign_split_features.shape

    print "pooled features shape :"
    print pooled_features.shape

    # get pooled features directly from whitened patches using feature extraction pipeline
    pooled_features_from_whitened_patches = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches)

    print "pooled features from whitened patches shape (combined pipeline):"
    print pooled_features_from_whitened_patches.shape

    print "saving model"
    model_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/feature_extraction_model.pkl"))
    sparse_coding.save_model(model_filename)

    print "reloading model"
    model_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/feature_extraction_model.pkl"))
    sparse_coding = SparseCoding(model_filename=model_filename)

    print "dictionary Shape :"
    print sparse_coding.get_dictionary().shape

    sparse_features = sparse_coding.get_sparse_features(whitened_patches)
    print "sparse features shape from loaded model :"
    print sparse_features.shape
