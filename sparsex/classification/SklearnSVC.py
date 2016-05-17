from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from ..feature_extraction.feature_extraction import SparseCoding, Spams, SklearnDL
import os
import numpy as np
from ..tests.preprocessing_test import test_whitening

from sklearn.utils.validation import NotFittedError

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class SklearnSVC(object):
    def __init__(self, model_filename=None):
        if model_filename is not None:
            self.load_model(model_filename)
        else:
            self.linSVC_obj = LinearSVC()


    def save_model(self, filename):
        # save linSVC object to file, compress is also to prevent multiple model files.
        joblib.dump(self.linSVC_obj, filename, compress=3)


    def load_model(self, filename):
        # load linSVC Object from file
        self.linSVC_obj = joblib.load(filename)


    def train(self, X, Y):
        assert X.ndim == 2, "Classifier training data X.ndim is %d instead of 2" %X.ndim
        assert Y.ndim == 1, "Classifier training data Y.ndim is %d instead of 1" %Y.ndim

        # train the model
        self.linSVC_obj.fit(X,Y)


    def get_predictions(self, X):
        assert X.ndim == 2, "Classifier prediction data X.ndim is %d instead of 2" %X.ndim

        # get classes
        try:
            return self.linSVC_obj.predict(X)
        except NotFittedError:
            raise NotFittedError("Classification model cannot preidct without being trained first. " \
                                 + "Train the classification model at least once to prevent this error.")



if __name__ == "__main__":
    image_filename_1 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    image_filename_2 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB02_P00A-005E-10_64x64.pgm"))
    classification_model_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/classification_model.pkl"))
    for message, feature_extraction_library_name, trained_feature_extraction_model_filename, classification_model_filename in zip(
        ["\n### classification using spams feature extraction",
         "\n### classification using spams feature extraction"],
        [SparseCoding.SPAMS,
         SparseCoding.SKLEARN_DL],
        [Spams.DEFAULT_TRAINED_MODEL_FILENAME,
         SklearnDL.DEFAULT_TRAINED_MODEL_FILENAME],
        [classification_model_filename,
         classification_model_filename]):
        
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

        # create SklearnSVC object
        sklearn_svc = SklearnSVC()

        # training the SklearnSVC on X and Y. This is just to train once for the SklearnSVC to be able to classify.
        sklearn_svc.train(X_input, Y_input)

        # save the model
        print "saving classification model to file :\n", classification_model_filename
        sklearn_svc.save_model(classification_model_filename)

        # re-load the classification model
        print "re-loading classification model from file :\n", classification_model_filename
        sklearn_svc = SklearnSVC(model_filename=classification_model_filename)

        # predict the class of X
        Y_predict = sklearn_svc.get_predictions(X_input)

        print "pooled features 1 shape :\n", pooled_features_1.shape
        print "pooled features 2 shape :\n", pooled_features_2.shape

        print "X_input shape :\n", X_input.shape

        print "Y_input shape :\n", Y_input.shape
        print "Y_input :\n", Y_input

        print "Y_predict shape :\n", Y_predict.shape
        print "Y_predict :\n", Y_predict

