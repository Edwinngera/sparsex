from ..customutils.customutils import save_image, get_image_from_file, get_giant_patch_image, resize_image_to_64x64
from ..preprocessing.preprocessing import Preprocessing
from PIL import Image
import numpy as np
import os, logging, sys

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def test_image_resize(image_array):
    preprocessing = Preprocessing()
    resized_image_array = preprocessing.get_resized_image(image_array, image_size=(64,64), multiple_images=True)

    # (number_images, image_side, image_side)
    assert resized_image_array.ndim == 3, "test_image_resize: Failed.\nresized_image_array.ndim is {0} instead of 3".format(resized_image_array.ndim)
    logging.debug("resized_image_array.shape: {0}".format(resized_image_array.shape))
    logging.info("test_image_resize: Success")


def test_extract_patches(image_array):
    preprocessing = Preprocessing()
    resized_image_array = preprocessing.get_resized_image(image_array, image_size=(64,64), multiple_images=True)
    patches = preprocessing.extract_patches(resized_image_array, patch_size=(8,8), multiple_images=True)

    # (number_images, number_patches, patch_side, patch_side)
    assert patches.ndim == 4, "test_extract_patches: Failed.\patches.ndim is {0} instead of 4".format(patches.ndim)
    logging.debug("patches.shape: {0}".format(patches.shape))
    logging.info("test_extract_patches: Success")


def test_contrast_normalization(image_array):
    preprocessing = Preprocessing()
    resized_image_array = preprocessing.get_resized_image(image_array, image_size=(64,64), multiple_images=True)
    patches = preprocessing.extract_patches(resized_image_array, patch_size=(8,8), multiple_images=True)
    normalized_patches = preprocessing.get_contrast_normalized_patches(patches, multiple_images=True)

    # (number_images, number_patches, patch_side, patch_side)
    assert normalized_patches.ndim == 4, "test_contrast_normalization: Failed.\normalized_patches.ndim is {0} instead of 4".format(normalized_patches.ndim)
    logging.debug("normalized_patches.shape: {0}".format(normalized_patches.shape))
    logging.debug("normalized_patch.mean(): {0}".format(normalized_patches[0][0].mean()))
    logging.debug("normalized_patch.var(): {0}".format(normalized_patches[0][0].var()))
    logging.info("test_contrast_normalization: Success")


def test_whitening(image_array):
    preprocessing = Preprocessing()
    resized_image_array = preprocessing.get_resized_image(image_array, image_size=(64,64), multiple_images=True)
    patches = preprocessing.extract_patches(resized_image_array, patch_size=(8,8), multiple_images=True)
    normalized_patches = preprocessing.get_contrast_normalized_patches(patches, multiple_images=True)
    whitened_patches = preprocessing.get_whitened_patches(normalized_patches, multiple_images=True)

    # (number_images, number_patches, patch_side, patch_side)
    assert whitened_patches.ndim == 4, "test_whitening: Failed.\whitened_patches.ndim is {0} instead of 4".format(whitened_patches.ndim)
    logging.debug("whitened_patches.shape: {0}".format(whitened_patches.shape))
    logging.info("test_whitening: Success")


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

    # test image resize
    test_image_resize(image_array)

    # test extract patches
    test_extract_patches(image_array)

    # test contrast normalization
    test_contrast_normalization(image_array)

    # test whitening
    test_whitening(image_array)
