from SklearnDL import SklearnDL
from Spams import Spams
from ..tests.preprocessing_test import test_whitening
from ..customutils import customutils
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class SparseCoding(object):
    
    SKLEARN_DL = "SklearnDL"
    SPAMS = "Spams"
    
    def __init__(self, library_name=SPAMS, model_filename=None, **kwargs):
        if library_name == SparseCoding.SKLEARN_DL:
            self.library = SklearnDL(model_filename, **kwargs)
        elif library_name == SparseCoding.SPAMS:
            self.library = Spams(model_filename, **kwargs)
        else:
            raise AttributeError("Invalid library_name : \"{0}\" for SparseCoding class in" \
            + "feature_extraction".format(library_name))

    def save_model(self, filename):
        return self.library.save_model(filename)

    def load_model(self, filename):
        return self.library.load_model(filename)

    def learn_dictionary(self, whitened_patches):
        """Returns None from (n,p,p) or (n,p**2) whitened_patches."""
        return self.library.learn_dictionary(whitened_patches)

    def get_dictionary(self):
        """Returns (k,p**2) dictionary"""
        return self.library.get_dictionary()

    def get_sparse_features(self, whitened_patches):
        """Returns (n,k) encoding from (n,p,p) or (n,p**2) whitened_patches and (k,p**2) internal dictionary."""
        return self.library.get_sparse_features(whitened_patches)

    def get_sign_split_features(self, sparse_features):
        """Returns (n,2*f) feature_vector from (n,f) feature_vector."""
        return self.library.get_sign_split_features(sparse_features)

    def get_pooled_features(self, input_feature_map, filter_size=(19,19)):
        """Returns ((n**2/s**2),k) feature_map from (n**2, k) feature_map and (s,s) filter_size."""
        # need to determine if there must be a default value for filter_size
        return self.library.get_pooled_features(input_feature_map, filter_size)

    def get_pooled_features_from_whitened_patches(self, whitened_patches, filter_size=(19,19)):
        """Returns ((n**2/s**2),k) feature_map from (n**2,p**2) or (n**2,p,p) patches, (s,s) filter_size, (k,p**2) internal dictionary."""
        # need to determine if there must be a default value for filter_size
        return self.library.get_pooled_features_from_whitened_patches(whitened_patches, filter_size)
        


if __name__ == "__main__":
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    library_name = SparseCoding.SPAMS
    model_filename = Spams.DEFAULT_MODEL_FILENAME
    
    for message, library_name, model_filename in zip(["\n\n### Spams feature extraction",
                                                      "\n\n### SklearnDL feature extraction"],
                                                     [SparseCoding.SPAMS,
                                                      SparseCoding.SKLEARN_DL],
                                                     [Spams.DEFAULT_MODEL_FILENAME,
                                                      SklearnDL.DEFAULT_MODEL_FILENAME]):
        print message
        
        # get whitened patches
        whitened_patches = test_whitening(image_filename, False, False)

        # create sparse coding object
        sparse_coding = SparseCoding(library_name=library_name)

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
        sparse_coding.save_model(model_filename)

        print "reloading model"
        sparse_coding = SparseCoding(library_name=library_name, model_filename=model_filename)

        print "dictionary Shape :"
        print sparse_coding.get_dictionary().shape

        sparse_features = sparse_coding.get_sparse_features(whitened_patches)
        print "sparse features shape from loaded model :"
        print sparse_features.shape
