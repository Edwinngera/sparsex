from ..feature_extraction.feature_extraction import SparseCoding
from preprocessing_test import test_whitening
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
        whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))
        sparse_coding.learn_dictionary(whitened_patches[:10])
        sparse_code = sparse_coding.get_sparse_code(whitened_patches[100:101])
        print "Dictionary Shape :"
        print sparse_coding.DL_obj.components_.shape
        print "Sparse Code Shape :"
        print sparse_code.shape
        # SAVE WEIGHTS of non default
        sparse_coding.save_model_weights(filename)




def test_load_model_params(filename):
    print "\n\nLoading Sparse Code Params Test"
    sparse_coding = SparseCoding(model_params_file=filename)

    # testing pipeline on sparse coding object which has loaded weights
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))
    sparse_coding.learn_dictionary(whitened_patches[:10])
    sparse_code = sparse_coding.get_sparse_code(whitened_patches[100:101])
    print "Dictionary Shape :"
    print sparse_coding.DL_obj.components_.shape
    print "Sparse Code Shape :"
    print sparse_code.shape


def test_load_model_weights(filename):
    print "\n\nLoading Sparse Code Weights Test"
    sparse_coding = SparseCoding(model_weights_file=filename)

    # testing pipeline on sparse coding object which has loaded weights
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))
    sparse_coding.learn_dictionary(whitened_patches[:10])
    sparse_code = sparse_coding.get_sparse_code(whitened_patches[100:101])
    print "Dictionary Shape :"
    print sparse_coding.DL_obj.components_.shape
    print "Sparse Code Shape :"
    print sparse_code.shape

def test_load_params_and_weights(model_params_file, model_weights_file):
    print "\n\nLoading Sparse Code Params and Weights Test"
    sparse_coding = SparseCoding(model_params_file=model_params_file, model_weights_file=model_weights_file)

    # testing pipeline on sparse coding object which has loaded weights
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "../tests/data/yaleB01_P00A-005E-10_64x64.pgm"))
    whitened_patches = test_whitening(image_filename, False, False)
    whitened_patches = whitened_patches.reshape((whitened_patches.shape[0], -1))
    sparse_coding.learn_dictionary(whitened_patches[:10])
    sparse_code = sparse_coding.get_sparse_code(whitened_patches[100:101])
    print "Dictionary Shape :"
    print sparse_coding.DL_obj.components_.shape
    print "Sparse Code Shape :"
    print sparse_code.shape


if __name__ == "__main__":
    model_params_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "./data/model_params_test.json"))
    model_weights_file = os.path.realpath(os.path.join(THIS_FILE_PATH, "./data/model_weights_test.h5"))

    # test saving model params
    test_save_model_params(model_params_file)

    # test saving model weights
    test_save_model_weights(model_weights_file, default_state=False)

    # test loading model params
    test_load_model_params(model_params_file)

    # test loading model weights
    test_load_model_weights(model_weights_file)

    # test loading model params & weights
    test_load_params_and_weights(model_params_file, model_weights_file)
    