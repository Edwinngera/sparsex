from SklearnSVC import SklearnSVC
from ..feature_extraction.feature_extraction import SparseCoding, Spams, SklearnDL
import os
import numpy as np
from ..tests.preprocessing_test import test_whitening

from sklearn.utils.validation import NotFittedError

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class Classifier(object):
    
    SKLEARN_SVC = "SklearnSVC"
    
    def __init__(self, library_name=SKLEARN_SVC, model_filename=None):
        if library_name == Classifier.SKLEARN_SVC:
            self.library = SklearnSVC(model_filename)
        else:
            raise AttributeError("Invalid library_name : \"{0}\" for Classifier class in" \
            + "feature_extraction".format(library_name))


    def save_model(self, filename):
        return self.library.save_model(filename)


    def load_model(self, filename):
        return self.library.load_model(filename)


    def train(self, X, Y):
        return self.library.train(X, Y)


    def get_predictions(self, X):
        return self.library.get_predictions(X)



if __name__ == "__main__":
    image_filename_1 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    image_filename_2 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB02_P00A-005E-10_64x64.pgm"))
    for message, feature_extraction_library_name, trained_feature_extraction_model_filename, classification_library_name, classification_model_filename in zip(
        ["\n### sklearnsvc classification using sklearndl feature extraction",
         "\n### sklearnsvc classification using sklearndl feature extraction",
         "\n### sklearnsvc classification using spams feature extraction",
         "\n### sklearnsvc classification using spams feature extraction"],
        [SparseCoding.SKLEARN_DL,
         SparseCoding.SKLEARN_DL,
         SparseCoding.SPAMS,
         SparseCoding.SPAMS],
        [SklearnDL.DEFAULT_TRAINED_MODEL_FILENAME,
         SklearnDL.DEFAULT_TRAINED_MODEL_FILENAME,
         Spams.DEFAULT_TRAINED_MODEL_FILENAME,
         Spams.DEFAULT_TRAINED_MODEL_FILENAME],
        [Classifier.SKLEARN_SVC,
         Classifier.SKLEARN_SVC,
         Classifier.SKLEARN_SVC,
         Classifier.SKLEARN_SVC],
        [SklearnSVC.DEFAULT_MODEL_FILENAME,
         SklearnSVC.DEFAULT_TRAINED_MODEL_FILENAME,
         SklearnSVC.DEFAULT_MODEL_FILENAME,
         SklearnSVC.DEFAULT_TRAINED_MODEL_FILENAME]):
        
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

        # create classifier object
        classifier = Classifier(classification_library_name)

        # training the classifier on X and Y. This is just to train once for the classifier to be able to classify.
        classifier.train(X_input, Y_input)

        # save the model
        print "saving classification model to file :\n", classification_model_filename
        classifier.save_model(classification_model_filename)

        # re-load the classification model
        print "re-loading classification model from file :\n", classification_model_filename
        classifier = Classifier(classification_library_name, classification_model_filename)

        # predict the class of X
        Y_predict = classifier.get_predictions(X_input)

        print "pooled features 1 shape :\n", pooled_features_1.shape
        print "pooled features 2 shape :\n", pooled_features_2.shape

        print "X_input shape :\n", X_input.shape

        print "Y_input shape :\n", Y_input.shape
        print "Y_input :\n", Y_input

        print "Y_predict shape :\n", Y_predict.shape
        print "Y_predict :\n", Y_predict

