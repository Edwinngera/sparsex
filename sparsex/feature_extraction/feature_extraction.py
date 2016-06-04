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

    def learn_dictionary(self, patches, multiple_images=False):
        """Returns None from (n,p**2) patches for single image."""
        return self.library.learn_dictionary(patches, multiple_images)

    def get_dictionary(self):
        """Returns (k,p**2) dictionary"""
        return self.library.get_dictionary()

    def get_sparse_features(self, patches, multiple_images=False):
        """Returns (n,k) encoding from (n,p**2) patches and (k,p**2) internal dictionary for single image."""
        return self.library.get_sparse_features(patches, multiple_images)

    def get_sign_split_features(self, sparse_features, multiple_images=False):
        """Returns (n,2*k) feature_vector from (n,k) feature_vector."""
        return self.library.get_sign_split_features(sparse_features, multiple_images)

    def pipeline(self, patches, sign_split=True, pooling=True, pooling_size=(3,3), multiple_images=False):
        """Returns (n/s**2,2k) feature map from (n,p**2), sign_split, pooling, (s,s) pooling_size, (k,p**2) internal dictionary for single image."""
        # need to determine if there must be a default value for pooling_size
        return self.library.pipeline(patches, sign_split, pooling, pooling_size, multiple_images)
        


if __name__ == "__main__":
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))

    # loop through all libraries
    for message, library_name, model_filename in zip(["\n### Spams feature extraction",
                                                      "\n### Spams trained feature extraction",
                                                      "\n### SklearnDL feature extraction",
                                                      "\n### SklearnDL trained feature extraction"],
                                                     [SparseCoding.SPAMS,
                                                      SparseCoding.SPAMS,
                                                      SparseCoding.SKLEARN_DL,
                                                      SparseCoding.SKLEARN_DL],
                                                     [Spams.DEFAULT_MODEL_FILENAME,
                                                      Spams.DEFAULT_TRAINED_MODEL_FILENAME,
                                                      SklearnDL.DEFAULT_MODEL_FILENAME,
                                                      SklearnDL.DEFAULT_TRAINED_MODEL_FILENAME]):
        print message
        print library_name
        print model_filename
        
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
