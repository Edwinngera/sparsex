from sklearn.externals import joblib
from ..feature_extraction.feature_extraction import SparseCoding, Spams, SklearnDL
import os
import numpy as np
from ..tests.preprocessing_test import test_whitening

from sklearn.utils.validation import NotFittedError

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class SVMLight(object):
    
    DEFAULT_MODEL_FILENAME = os.path.realpath(os.path.join(THIS_FILE_PATH,
                             "../tests/data/classification_model_svmlight.pkl"))
    DEFAULT_TRAINED_MODEL_FILENAME = os.path.realpath(os.path.join(THIS_FILE_PATH,
                             "../tests/data/trained_classification_model_svmlight.pkl"))
    
    def __init__(self, model_filename=None):
        raise NotImplementedError


    def save_model(self, filename):
        raise NotImplementedError


    def load_model(self, filename):
        raise NotImplementedError


    def train(self, X, Y):
        raise NotImplementedError


    def get_predictions(self, X):
        raise NotImplementedError



if __name__ == "__main__":
    image_filename_1 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    image_filename_2 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB02_P00A-005E-10_64x64.pgm"))
    for message, feature_extraction_library_name, trained_feature_extraction_model_filename, classification_model_filename in zip(
        ["\n### classification using spams feature extraction",
         "\n### classification using spams feature extraction",
         "\n### classification using spams feature extraction",
         "\n### classification using spams feature extraction"],
        [SparseCoding.SKLEARN_DL,
         SparseCoding.SKLEARN_DL,
         SparseCoding.SPAMS,
         SparseCoding.SPAMS],
        [SklearnDL.DEFAULT_TRAINED_MODEL_FILENAME,
         SklearnDL.DEFAULT_TRAINED_MODEL_FILENAME,
         Spams.DEFAULT_TRAINED_MODEL_FILENAME,
         Spams.DEFAULT_TRAINED_MODEL_FILENAME],
        [SVMLight.DEFAULT_MODEL_FILENAME,
         SVMLight.DEFAULT_TRAINED_MODEL_FILENAME,
         SVMLight.DEFAULT_MODEL_FILENAME,
         SVMLight.DEFAULT_TRAINED_MODEL_FILENAME]):
        
        print message
        print feature_extraction_library_name
        print trained_feature_extraction_model_filename
        print classification_model_filename

        # get whitened patches
        whitened_patches_1 = test_whitening(image_filename_1, False, False)
        whitened_patches_2 = test_whitening(image_filename_2, False, False)

        # create sparse coding object
        print "loading trained feature extraction model from file :\n", trained_feature_extraction_model_filename
        sparse_coding = SparseCoding(feature_extraction_library_name,
                                     trained_feature_extraction_model_filename)

        # get pooled features directly from whitened patches using feature extraction pipeline
        pooled_features_1 = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches_1)
        pooled_features_2 = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches_2)

        # input X, flattened features, ndim = 2, [n_samples, n_features]. Here there is only one image, so n_samples = 1
        X_input_1 = pooled_features_1.ravel().reshape((1,-1)) # will be removed when pipeline has standardized shapes.
        X_input_2 = pooled_features_2.ravel().reshape((1,-1)) # will be removed when pipeline has standardized shapes.
        X_input = np.vstack((X_input_1,X_input_2))

        # generate an fake class Y for X, ndim = 1, [n_samples]
        Y_input = np.arange(X_input.shape[0])

        # create SVMLight object
        svm_light = SVMLight()

        # training the SVMLight on X and Y. This is just to train once for the SVMLight to be able to classify.
        svm_light.train(X_input, Y_input)

        # save the model
        print "saving classification model to file :\n", classification_model_filename
        svm_light.save_model(classification_model_filename)

        # re-load the classification model
        print "re-loading classification model from file :\n", classification_model_filename
        svm_light = SVMLight(model_filename=classification_model_filename)

        # predict the class of X
        Y_predict = svm_light.get_predictions(X_input)

        print "pooled features 1 shape :\n", pooled_features_1.shape
        print "pooled features 2 shape :\n", pooled_features_2.shape

        print "X_input shape :\n", X_input.shape

        print "Y_input shape :\n", Y_input.shape
        print "Y_input :\n", Y_input

        print "Y_predict shape :\n", Y_predict.shape
        print "Y_predict :\n", Y_predict

