from ..feature_extraction.feature_extraction import SparseCoding
from preprocessing_test import test_whitening, test_preprocessing_combined_pipeline
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def test_save_model_params(filename=None):
    sparse_coding = SparseCoding()
    sparse_coding.save_model_params(filename)


def test_save_model_weights(filename=None, default_state=True):
    sparse_coding = SparseCoding()
    if default_state:
        sparse_coding.save_model_weights(filename)
    else:
        # testing pipeline on sparse coding object which has loaded weights
        image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
        whitened_patches = test_whitening(image_filename, False, False)
        sparse_coding.learn_dictionary(whitened_patches[:10])
        sparse_features = sparse_coding.get_sparse_features(whitened_patches[100:101])
        print "Dictionary Shape :\n", sparse_coding.DL_obj.components_.shape
        print "sparse features shape :\n", sparse_features.shape
        # SAVE WEIGHTS of non default
        sparse_coding.save_model_weights(filename)


def test_load_model_params(filename):
    print "\n\nLoading Sparse Code Params Test"
    sparse_coding = SparseCoding(model_params_file=filename)

    # testing pipeline on sparse coding object which has loaded weights
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    sparse_coding.learn_dictionary(whitened_patches[:10])
    sparse_features = sparse_coding.get_sparse_features(whitened_patches[100:101])
    print "Dictionary Shape :\n", sparse_coding.DL_obj.components_.shape
    print "sparse features shape :\n", sparse_features.shape


def test_load_model_weights(filename):
    print "\n\nLoading Sparse Code Weights Test"
    sparse_coding = SparseCoding(model_weights_file=filename)

    # testing pipeline on sparse coding object which has loaded weights
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    sparse_coding.learn_dictionary(whitened_patches[:10])
    sparse_features = sparse_coding.get_sparse_features(whitened_patches[100:101])
    print "Dictionary Shape :\n", sparse_coding.DL_obj.components_.shape
    print "sparse features shape :\n", sparse_features.shape


def test_load_params_and_weights(model_params_file, model_weights_file):
    print "\n\nLoading Sparse Code Params and Weights Test"
    sparse_coding = SparseCoding(model_params_file=model_params_file, model_weights_file=model_weights_file)

    # testing pipeline on sparse coding object which has loaded weights
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    sparse_coding.learn_dictionary(whitened_patches[:10])
    sparse_features = sparse_coding.get_sparse_features(whitened_patches[100:101])
    print "Dictionary Shape :\n", sparse_coding.DL_obj.components_.shape
    print "sparse features shape :\n", sparse_features.shape


def test_get_sparse_features():
    print "\n\nGet Sparse Features Test"
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
    print "\n\nGet Sign Split Features Test"
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
    print "\n\nGet Pooled Features Test"
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
    print "pooled features :\n", pooled_features


def test_get_pooled_features_from_whitened_patches():
    print "\n\nGet Pooled Features From Whitened Patches (Combined Pipeline) Test"
    # create sparse coding object
    sparse_coding = SparseCoding()

    # get whitened patches from preprocessing combined pipeline
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_preprocessing_combined_pipeline(image_filename, False, False)
    pooled_features = sparse_coding.get_pooled_features_from_whitened_patches(whitened_patches)

    print "pooled features shape :\n", pooled_features.shape 
    print "pooled features :\n", pooled_features

    return pooled_features


if __name__ == "__main__":
    model_params_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "./data/model_params_test.json"))
    model_weights_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "./data/model_weights_test.h5"))

    # test saving model params
    # test_save_model_params(model_params_file)

    # test saving model weights
    # test_save_model_weights(model_weights_file, default_state=False)

    # test loading model params
    # test_load_model_params(model_params_file)

    # test loading model weights
    # test_load_model_weights(model_weights_file)

    # test loading model params & weights
    # test_load_params_and_weights(model_params_file, model_weights_file)

    # test get sparse features
    # test_get_sparse_features()

    # test get sign split features
    # test_get_sign_split_features()

    # test get pooled features
    # test_get_pooled_features()

    # test feature extraction combined pipeline
    test_get_pooled_features_from_whitened_patches()
