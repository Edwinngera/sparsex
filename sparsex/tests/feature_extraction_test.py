from ..feature_extraction.feature_extraction import SparseCoding
from preprocessing_test import test_whitening, test_preprocessing_combined_pipeline
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def test_save_model(filename, train_model=False):
    print "##### Testing Save Feature Extraction Model, train_model={0}".format(train_model)
    # train the DL object such that it does not throw NotFittedError, or save untrained DL object
    if train_model:
        # force the DL object to be trained and then save the model
        image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
        whitened_patches = test_whitening(image_filename, False, False)
        sparse_coding = SparseCoding()
        sparse_coding.learn_dictionary(whitened_patches[:10])
        sparse_features = sparse_coding.get_sparse_features(whitened_patches[100:101])
        print "check if dictionary is computed, dictionary shape :\n", sparse_coding.get_dictionary().shape
        # save weights
        print "saving feature extraction model to file :\n", filename
        sparse_coding.save_model(filename)
    else:
        sparse_coding = SparseCoding()
        print "default instance, dictionary not computed"
        print "saving feature extraction model to file :\n", filename
        sparse_coding.save_model(filename)


def test_load_model(filename):
    print "##### Testing Load Feature Extraction Model"
    sparse_coding = SparseCoding(model_filename=filename)

    # test if model is trained or not
    try:
        print "check if dictionary is computed, dictionary shape :\n", sparse_coding.get_dictionary().shape
        print "dictionary computed, trained model"
    except AttributeError:
        print "dictionary not computed, untrained model"
    

def test_get_sparse_features():
    print "##### Testing Get Sparse Features"
    # create sparse coding object
    sparse_coding = SparseCoding()

    # testing pipeline on sparse coding object
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    # sparse_coding.learn_dictionary(whitened_patches[:100]) # not learning dictionary - only testing getting sparse code
    sparse_features = sparse_coding.get_sparse_features(whitened_patches)
    print "sparse features shape :\n", sparse_features.shape
    print "sparse features :\n", sparse_features[0]
    print "sparse features norm :", np.linalg.norm(sparse_features)


def test_get_sign_split_features():
    print "##### Testing Get Sign Split Features"
    # create sparse coding object
    sparse_coding = SparseCoding()

    # testing pipeline on sparse coding object
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    # sparse_coding.learn_dictionary(whitened_patches[:100]) # not learning dictionary - only testing getting sparse code
    sparse_features = sparse_coding.get_sparse_features(whitened_patches)
    sign_split_features = sparse_coding.get_sign_split_features(sparse_features)

    print "sparse features shape :\n", sparse_features.shape
    print "sparse features :\n", sparse_features[0]
    print "sparse features norm :", np.linalg.norm(sparse_features)

    print "sign split features shape :\n", sign_split_features.shape 
    print "sign split features :\n", sign_split_features[0]
    print "sign split features norm :\n", np.linalg.norm(sign_split_features)


def test_get_pooled_features():
    print "##### Testing Get Pooled Features"
    # create sparse coding object
    sparse_coding = SparseCoding()

    # testing pipeline on sparse coding object
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    # sparse_coding.learn_dictionary(whitened_patches[:100]) # not learning dictionary - only testing getting sparse code
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


def test_get_pooled_features_from_whitened_patches():
    print "##### Testing Get Pooled Features From Whitened Patches (Combined Pipeline)"
    # create sparse coding object
    sparse_coding = SparseCoding()

    # get whitened patches from preprocessing combined pipeline
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_preprocessing_combined_pipeline(image_filename, False, False)
    pooled_features = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches)

    print "pooled features shape :\n", pooled_features.shape

    return pooled_features


if __name__ == "__main__":
    model_params_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "./data/model_params_test.json"))
    model_weights_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "./data/model_weights_test.h5"))
    model_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/feature_extraction_model.pkl"))

    # test saving model, don't train model
    test_save_model(model_filename, train_model=False)

    # test loading untrained model
    test_load_model(model_filename)

    # test saving model, train model
    test_save_model(model_filename, train_model=True)

    # test loading trained model
    test_load_model(model_filename)

    # test get sparse features
    test_get_sparse_features()

    # test get sign split features
    test_get_sign_split_features()

    # test get pooled features
    test_get_pooled_features()

    # test feature extraction combined pipeline
    test_get_pooled_features_from_whitened_patches()
