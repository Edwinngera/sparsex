from ..customutils.customutils import get_image_from_file
from ..feature_extraction.Spams import Spams
from ..preprocessing.preprocessing import Preprocessing
import numpy as np
import os, sys, logging

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def test_learn_dictionary(image_array, library_name=None, model_filename=None):
    preprocessing = Preprocessing()
    patches = preprocessing.pipeline(image_array, image_size=(64,64), patch_size=(8,8), normalize=True, whiten=True, multiple_images=True)
    spams = Spams()
    spams.learn_dictionary(patches, multiple_images=True)
    spams_dictionary = spams.get_dictionary()
    
    # (k, p**2)
    assert spams_dictionary.ndim == 2, "test_learn_dictionary: Failed.\nspams_dictionary.ndim is {0} instead of 2".format(spams_dictionary.ndim)
    logging.debug("spams_dictionary.shape: {0}".format(spams_dictionary.shape))
    logging.info("test_learn_dictionary: Success")


def test_get_sparse_encoding(image_array, library_name=None, model_filename=None):
    preprocessing = Preprocessing()
    patches = preprocessing.pipeline(image_array, image_size=(64,64), patch_size=(8,8), normalize=True, whiten=True, multiple_images=True)
    spams = Spams()
    spams.learn_dictionary(patches, multiple_images=True)
    sparse_encoding = spams.get_sparse_features(patches, multiple_images=True)
    
    # (number_images, number_patches, k)
    assert sparse_encoding.ndim == 3, "test_get_sparse_encoding: Failed.\nsparse_encoding.ndim is {0} instead of 3".format(sparse_encoding.ndim)
    logging.debug("sparse_encoding.shape: {0}".format(sparse_encoding.shape))
    logging.info("test_get_sparse_encoding: Success")
    
    
def test_get_sign_split_features(image_array, library_name=None, model_filename=None):
    preprocessing = Preprocessing()
    patches = preprocessing.pipeline(image_array, image_size=(64,64), patch_size=(8,8), normalize=True, whiten=True, multiple_images=True)
    spams = Spams()
    spams.learn_dictionary(patches, multiple_images=True)
    sparse_encoding = spams.get_sparse_features(patches, multiple_images=True)
    sparse_sign_split_encoding = spams.get_sign_split_features(sparse_encoding, multiple_images=True)
    
    # (number_images, number_patches, 2k)
    assert sparse_sign_split_encoding.ndim == 3, "test_get_sign_split_features: Failed.\nsparse_sign_split_encoding.ndim is {0} instead of 3".format(sparse_sign_split_encoding.ndim)
    logging.debug("sparse_sign_split_encoding.shape: {0}".format(sparse_sign_split_encoding.shape))
    logging.info("test_get_sign_split_features: Success")
    

def test_get_pooled_features(library_name=None, model_filename=None):
    preprocessing = Preprocessing()
    patches = preprocessing.pipeline(image_array, image_size=(64,64), patch_size=(8,8), normalize=True, whiten=True, multiple_images=True)
    spams = Spams()
    spams.learn_dictionary(patches, multiple_images=True)
    sparse_encoding = spams.get_sparse_features(patches, multiple_images=True)
    sparse_sign_split_encoding = spams.get_sign_split_features(sparse_encoding, multiple_images=True)
    pooled_features = spams.get_pooled_features(sparse_sign_split_encoding, filter_size=(19,19), multiple_images=True)
    
    # (number_images, number_patches/filter_side**2, 2k)
    assert pooled_features.ndim == 3, "test_get_pooled_features: Failed.\npooled_features.ndim is {0} instead of 3".format(pooled_features.ndim)
    logging.debug("pooled_features.shape: {0}".format(pooled_features.shape))
    logging.info("test_get_pooled_features: Success")


def test_pipeline(library_name=None, model_filename=None):
    preprocessing = Preprocessing()
    patches = preprocessing.pipeline(image_array, image_size=(64,64), patch_size=(8,8), normalize=True, whiten=True, multiple_images=True)
    spams = Spams()
    spams.learn_dictionary(patches, multiple_images=True)
    features = spams.pipeline(patches, sign_split=False, pooling=True, pooling_size=(19,19), multiple_images=True)

    # (number_images, number_patches/filter_side**2, 2k)
    assert features.ndim == 3, "test_pipeline: Failed.\nfeatures.ndim is {0} instead of 3".format(features.ndim)
    logging.debug("features.shape: {0}".format(features.shape))
    logging.info("test_pipeline: Success")



if __name__ == "__main__":
    logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
                        level=logging.DEBUG,
                        stream=sys.stdout)

    # image file/files
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "./data/yaleB01_P00A-005E-10.pgm"))
    image_array = get_image_from_file(image_filename)

    # convert to 3d array, (number_images, image_shape[0], image_shape[1])
    image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1])

    logging.debug("image_array.shape : {0}".format(image_array.shape))
    
    # test learn dictionary
    test_learn_dictionary(image_array)
    
    # test get sparse encoding
    test_get_sparse_encoding(image_array)
    
    # test get sign split features
    test_get_sign_split_features(image_array)
    
    # test get pooled features
    test_get_pooled_features(image_array)
    
    # test pipeline
    test_pipeline(image_array)
    
    