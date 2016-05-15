from ..feature_extraction.feature_extraction import SparseCoding, Spams, SklearnDL
from preprocessing_test import test_whitening, test_preprocessing_combined_pipeline
import numpy as np
import os, sys, traceback

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def test_save_model(library_name=None, model_filename=None, train_model=False):
    print "##### Testing Save Feature Extraction Model, train_model={0}".format(train_model)
    # train the DL object such that it does not throw NotFittedError, or save untrained DL object
    if train_model:
        # force the DL object to be trained and then save the model
        image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
        whitened_patches = test_whitening(image_filename, False, False)
        sparse_coding = SparseCoding(library_name, model_filename)
        sparse_coding.learn_dictionary(whitened_patches[:100]) # making sure the model has a trained dictionary
        sparse_features = sparse_coding.get_sparse_features(whitened_patches[100:101])
        print "check if dictionary is computed, dictionary shape :\n", sparse_coding.get_dictionary().shape
        # save weights
        print "saving feature extraction model to file :\n", model_filename
        sparse_coding.save_model(model_filename)
    else:
        sparse_coding = SparseCoding(library_name, model_filename)
        print "default instance, dictionary not computed"
        print "saving feature extraction model to file :\n", model_filename
        sparse_coding.save_model(model_filename)


def test_load_model(library_name=None, model_filename=None):
    print "##### Testing Load Feature Extraction Model"
    sparse_coding = SparseCoding(library_name, model_filename)

    # test if model is trained or not
    try:
        print "check if dictionary is computed, dictionary shape :\n", sparse_coding.get_dictionary().shape
        print "dictionary computed, trained model"
    except AttributeError, KeyError:
        print "dictionary not computed, untrained model"
    

def test_get_sparse_features(library_name=None, model_filename=None):
    print "##### Testing Get Sparse Features"
    # create sparse coding object
    sparse_coding = SparseCoding(library_name, model_filename)

    # testing pipeline on sparse coding object
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    if model_filename == None:
        sparse_coding.learn_dictionary(whitened_patches[:100]) # making sure the model has a trained dictionary
    sparse_features = sparse_coding.get_sparse_features(whitened_patches)
    print "sparse features shape :\n", sparse_features.shape
    print "sparse features :\n", sparse_features[0]
    print "sparse features norm :", np.linalg.norm(sparse_features)
    
    return sparse_features


def test_get_sign_split_features(library_name=None, model_filename=None):
    print "##### Testing Get Sign Split Features"
    # create sparse coding object
    sparse_coding = SparseCoding(library_name, model_filename)

    # testing pipeline on sparse coding object
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    if model_filename == None:
        sparse_coding.learn_dictionary(whitened_patches[:100]) # making sure the model has a trained dictionary
    sparse_features = sparse_coding.get_sparse_features(whitened_patches)
    sign_split_features = sparse_coding.get_sign_split_features(sparse_features)

    print "sparse features shape :\n", sparse_features.shape
    print "sparse features :\n", sparse_features[0]
    print "sparse features norm :", np.linalg.norm(sparse_features)

    print "sign split features shape :\n", sign_split_features.shape 
    print "sign split features :\n", sign_split_features[0]
    print "sign split features norm :\n", np.linalg.norm(sign_split_features)
    
    return sign_split_features


def test_get_pooled_features(library_name=None, model_filename=None):
    print "##### Testing Get Pooled Features"
    # create sparse coding object
    sparse_coding = SparseCoding(library_name, model_filename)

    # testing pipeline on sparse coding object
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    if model_filename == None:
        sparse_coding.learn_dictionary(whitened_patches[:100]) # making sure the model has a trained dictionary
    sparse_features = sparse_coding.get_sparse_features(whitened_patches)
    sign_split_features = sparse_coding.get_sign_split_features(sparse_features)
    pooled_features = sparse_coding.get_pooled_features(sign_split_features)

    print "sparse features shape :\n", sparse_features.shape
    print "sparse features :\n", sparse_features[0]
    print "sparse features norm :", np.linalg.norm(sparse_features)

    print "sign split features shape :\n", sign_split_features.shape 
    print "sign split features :\n", sign_split_features[0]
    print "sign split features norm :\n", np.linalg.norm(sign_split_features)

    print "pooled features shape :\n", pooled_features.shape
    
    return pooled_features


def test_get_pooled_features_from_whitened_patches(library_name=None, model_filename=None):
    print "##### Testing Get Pooled Features From Whitened Patches (Combined Pipeline)"
    # create sparse coding object
    sparse_coding = SparseCoding(library_name, model_filename)

    # get whitened patches from preprocessing combined pipeline
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_preprocessing_combined_pipeline(image_filename, False, False)
    if model_filename == None:
        sparse_coding.learn_dictionary(whitened_patches[:100]) # making sure the model has a trained dictionary
    pooled_features = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches)
        
    print "pooled features shape :\n", pooled_features.shape

    return pooled_features


if __name__ == "__main__":
    # loop through all libraries
    for message, library_name, model_filename in zip(["\n### Default feature extraction",
                                                      "\n### Spams feature extraction",
                                                      "\n### SklearnDL feature extraction"],
                                                     [SparseCoding.SPAMS,
                                                      SparseCoding.SPAMS,
                                                      SparseCoding.SKLEARN_DL],
                                                     [None,
                                                      Spams.DEFAULT_MODEL_FILENAME,
                                                      SklearnDL.DEFAULT_MODEL_FILENAME]):
        print message
        print library_name
        print model_filename

        # test save and load model, only when a model_filename is given
        if model_filename != None:
            # test saving model, don't train model
            test_save_model(library_name, model_filename, train_model=False)

            # test loading untrained model
            test_load_model(library_name, model_filename)

            # test saving model, train model
            test_save_model(library_name, model_filename, train_model=True)

            # test loading trained model
            test_load_model(library_name, model_filename)

        # test get sparse features
        test_get_sparse_features(library_name, model_filename)

        # test get sign split features
        test_get_sign_split_features(library_name, model_filename)

        # test get pooled features
        test_get_pooled_features(library_name, model_filename)

        # test feature extraction combined pipeline
        test_get_pooled_features_from_whitened_patches(library_name, model_filename)
