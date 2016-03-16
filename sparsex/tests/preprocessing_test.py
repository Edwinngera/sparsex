from ..customutils.customutils import save_image, get_image_from_file, get_giant_patch_image, resize_image_to_64x64
from ..preprocessing.preprocessing import Preprocessing
from PIL import Image
from skimage.util.montage import montage2d
from scipy.misc import imshow, imsave
import numpy as np
import os

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))

def show_and_save_montage_of_patches(patches, is_show, is_save, save_filename=None):
    if is_show or is_save:
        if patches.ndim == 3:
            patches_montage = montage2d(patches)
        else:
            patches_montage = patches

        if is_show:
            imshow(patches_montage)
        if is_save and (save_filename is not None):
            save_filename = os.path.join(THIS_FILE_PATH, save_filename)
            imsave(save_filename, patches_montage)

def test_patch_extraction(image_filename, show_montage=True, save_montage=True):
    # get instance of Preprocessing
    preprocessing = Preprocessing()

    # get original image array
    print "Original Image"
    image_array = get_image_from_file(image_filename)
    print image_array.shape
    print image_array.dtype
    
    # get patches
    print "Extracting patches"
    patches = preprocessing.extract_patches(image_array, patch_size=(8,8))
    print patches.shape

    show_and_save_montage_of_patches(image_array, show_montage, save_montage,
                                     "./data/01_original_image.jpg")
    show_and_save_montage_of_patches(patches, show_montage, save_montage,
                                     "./data/02_patch_extraction_montage.jpg")

    return patches


def test_contrast_normalization(image_filename, show_montage=True, save_montage=True):
    # get instance of Preprocessing
    preprocessing = Preprocessing()

    # get patches
    patches = test_patch_extraction(image_filename, show_montage, save_montage)

    # get normalized patches
    print "Normalizing"
    normalized_patches = preprocessing.get_contrast_normalized_patches(patches)
    print normalized_patches.shape

    # get variance of normalized patches (each patch is reshaped along axis=1)
    print "Normalized patches variance"
    print normalized_patches.reshape((normalized_patches.shape[0], -1)).var(axis=1)[:10]

    show_and_save_montage_of_patches(normalized_patches, show_montage, save_montage,
                                     "./data/03_normalized_patches_montage.jpg")

    return normalized_patches


def test_whitening(image_filename, show_montage=True, save_montage=True):
    # get instance of Preprocessing
    preprocessing = Preprocessing()

    # get normalized patches
    normalized_patches = test_contrast_normalization(image_filename, show_montage, save_montage)

    # get whitened patches
    print "Whitening"
    whitened_patches = preprocessing.get_whitened_patches(normalized_patches)
    print whitened_patches.shape

    show_and_save_montage_of_patches(whitened_patches, show_montage, save_montage,
                                     "./data/04_whitened_patches_montage.jpg")

    return whitened_patches
    

if __name__ == "__main__":
    image_filename = os.path.realpath(os.path.join(THIS_FILE_PATH, "./data/yaleB01_P00A-005E-10_64x64.pgm"))
    
    # test patch extraction
    # test_patch_extraction(image_filename, show_montage=True, save_montage=True)

    # test contrast normalization
    # test_contrast_normalization(image_filename, show_montage=True, save_montage=True)

    # test whitening
    test_whitening(image_filename, show_montage=True, save_montage=True)
    