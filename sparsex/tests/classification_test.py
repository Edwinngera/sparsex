from ..classification.classification import Classifier
from ..feature_extraction.feature_extraction import SparseCoding, Spams, SklearnDL
from ..tests.preprocessing_test import test_whitening
import numpy as np
import os

from sklearn.utils.validation import NotFittedError


THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def test_save_model(feature_extraction_library_name, trained_feature_extraction_model_filename,
                    classification_model_filename, train_model=False):
    print "##### Testing Save Classification Model, train_model={0}".format(train_model)
    # train the LinSVC object such that it does not throw NotFittedError, or save untrained LinSVC object
    if train_model:
        # force the LinSVC object to be trained and then save the model
        image_filename_1 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
        image_filename_2 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB02_P00A-005E-10_64x64.pgm"))
        whitened_patches_1 = test_whitening(image_filename_1, False, False)
        whitened_patches_2 = test_whitening(image_filename_2, False, False)
        print "loading trained feature extraction model from file :\n", trained_feature_extraction_model_filename
        sparse_coding = SparseCoding(library_name=feature_extraction_library_name,
                                     model_filename=trained_feature_extraction_model_filename)
        pooled_features_1 = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches_1)
        pooled_features_2 = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches_2)
        X_input_1 = pooled_features_1.ravel().reshape((1,-1)) # will be removed when pipeline has standardized shapes.
        X_input_2 = pooled_features_2.ravel().reshape((1,-1)) # will be removed when pipeline has standardized shapes.
        X_input = np.vstack((X_input_1,X_input_2))
        Y_input = np.arange(X_input.shape[0])
        classifier = Classifier()
        classifier.train(X_input, Y_input)
        # save the model
        print "saving trained classification model to file :\n", classification_model_filename
        classifier.save_model(classification_model_filename)
    else:
        classifier = Classifier()
        print "saving untrained classification model to file :\n", classification_model_filename
        classifier.save_model(classification_model_filename)


def test_load_model(feature_extraction_library_name, trained_feature_extraction_model_filename,
                    classification_model_filename):
    print "##### Testing Load Classification Model"
    classifier = Classifier(model_filename=classification_model_filename)

    # test if model is trained or not
    try:
        image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
        whitened_patches = test_whitening(image_filename, False, False)
        print "loading trained feature extraction model from file :\n", trained_feature_extraction_model_filename
        sparse_coding = SparseCoding(library_name=feature_extraction_library_name,
                                     model_filename=trained_feature_extraction_model_filename)
        pooled_features = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches)
        X_input = pooled_features.ravel().reshape((1,-1)) # will be removed when pipeline has standardized shapes.
        print "check if model can classify, Y_predict shape :\n", classifier.get_predictions(X_input)
        print "classifier was able to classify, trained model"
    except NotFittedError:
        print "classifier was unable to classify, untrained model"


def test_get_predictions(feature_extraction_library_name, trained_feature_extraction_model_filename,
                         classification_model_filename):
    print "##### Testing Classification Get Predictions"
    # load trained feature extraction model and trained classification model and test whether the model generates predictions
    image_filename_1 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    image_filename_2 = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB02_P00A-005E-10_64x64.pgm"))
    whitened_patches_1 = test_whitening(image_filename_1, False, False)
    whitened_patches_2 = test_whitening(image_filename_2, False, False)
    print "loading trained feature extraction model from file :\n", trained_feature_extraction_model_filename
    sparse_coding = SparseCoding(library_name=feature_extraction_library_name,
                                 model_filename=trained_feature_extraction_model_filename)
    pooled_features_1 = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches_1)
    pooled_features_2 = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches_2)
    X_input_1 = pooled_features_1.ravel().reshape((1,-1)) # will be removed when pipeline has standardized shapes.
    X_input_2 = pooled_features_2.ravel().reshape((1,-1)) # will be removed when pipeline has standardized shapes.
    X_input = np.vstack((X_input_1,X_input_2))
    print "loading trained classification model from file :\n", classification_model_filename
    classifier = Classifier(model_filename=classification_model_filename)
    Y_predict = classifier.get_predictions(X_input)

    print "X_input shape :\n", X_input.shape
    print "Y_predict shape :\n", Y_predict.shape
    print "Y_predict :\n", Y_predict



if __name__ == "__main__":
    classification_model_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/classification_model.pkl"))
    
    for message, feature_extraction_library_name, trained_feature_extraction_model_filename, classification_model_filename in zip(
        ["\n### Default classification with spams feature_extraction",
         "\n### Default classification with sklearn_dl feature_extraction",
         "\n### Model classification with spams feature_extraction",
         "\n### Model classification with sklearn_dl feature_extraction"],
        [SparseCoding.SPAMS,
         SparseCoding.SKLEARN_DL,
         SparseCoding.SPAMS,
         SparseCoding.SKLEARN_DL],
        [Spams.DEFAULT_TRAINED_MODEL_FILENAME,
         SklearnDL.DEFAULT_TRAINED_MODEL_FILENAME,
         Spams.DEFAULT_TRAINED_MODEL_FILENAME,
         SklearnDL.DEFAULT_TRAINED_MODEL_FILENAME],
        [classification_model_filename,
         classification_model_filename,
         classification_model_filename,
         classification_model_filename]):

        print message
        print feature_extraction_library_name
        print trained_feature_extraction_model_filename
        print classification_model_filename
        
        # test save untrained model
        test_save_model(feature_extraction_library_name, trained_feature_extraction_model_filename,
                        classification_model_filename, train_model=False)

        # test load untrained model
        test_load_model(feature_extraction_library_name, trained_feature_extraction_model_filename,
                        classification_model_filename)

        # test save trained model
        test_save_model(feature_extraction_library_name, trained_feature_extraction_model_filename,
                        classification_model_filename, train_model=True)

        # test load trained model
        test_load_model(feature_extraction_library_name, trained_feature_extraction_model_filename,
                        classification_model_filename)

        # test get predictions
        test_get_predictions(feature_extraction_library_name, trained_feature_extraction_model_filename,
                             classification_model_filename)

