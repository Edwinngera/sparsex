from ..customutils.customutils import save_image, get_image_from_file, get_giant_patch_image, resize_image_to_64x64
from ..preprocessing.preprocessing import Preprocessing
from PIL import Image
from skimage.util.montage import montage2d
from scipy.misc import imshow
import numpy as np
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))

def test_patch_extraction(image_filename):
    # get instance of Preprocessing
    preprocessing = Preprocessing()

    # get original image array
    image_array = get_image_from_file(image_filename)
    # print image_array.shape
    # print image_array.dtype

    # save original image
    destination_filename = os.path.join(this_file_path, "./data/01_original_image.jpg")
    save_image(image_array.astype('uint8'), destination_filename=destination_filename)
    
    # get patches
    print "Extracting patches"
    patches = preprocessing.extract_patches(image_array, patch_size=(8,8))
    print patches.shape

    # get montage
    patches_montage = montage2d(patches)

    # show image
    imshow(patches_montage)

    # get giant patch image
    # giant_patch_image_array = get_giant_patch_image(patches, dtype='uint8', scale=False)
    # print giant_patch_image_array.shape

    # save giant image
    # destination_filename = os.path.join(this_file_path, "./data/02_patch_extraction_giant_image.jpg")
    # save_image(giant_patch_image_array, destination_filename=destination_filename)

    return patches


def test_contrast_normalization(image_filename):
    # get instance of Preprocessing
    preprocessing = Preprocessing()

    # get patches
    patches = test_patch_extraction(image_filename)

    # get normalized patches
    print "Normalizing"
    normalized_patches = preprocessing.get_contrast_normalized_patches(patches)
    print normalized_patches.shape
    print normalized_patches[0]

    # get variance of normalized patches (each patch is reshaped along axis=1)
    print normalized_patches.reshape((normalized_patches.shape[0], -1)).var(axis=1)[:10]

    # get montage
    normalized_patches_montage = montage2d(normalized_patches)

    # show image
    imshow(normalized_patches_montage)

    # get giant normalized patch image array
    # giant_normalized_patch_image_array = get_giant_patch_image(normalized_patches, dtype='uint8', scale=True)
    # print giant_normalized_patch_image_array.shape
    # print giant_normalized_patch_image_array[:8,:8]

    # save giant normalized image
    # destination_filename = os.path.join(this_file_path, "./data/03_normalized_patches_giant_image.jpg")
    # save_image(giant_normalized_patch_image_array, destination_filename=destination_filename)

    return normalized_patches


def test_whitening(image_filename):
    # get instance of Preprocessing
    preprocessing = Preprocessing()

    # get normalized patches
    normalized_patches = test_contrast_normalization(image_filename)

    # get whitened patches
    print "Whitening"
    whitened_patches = preprocessing.zca_whitening(normalized_patches)
    print whitened_patches.shape
    print whitened_patches[0]

    # get montage
    whitened_patches_montage = montage2d(whitened_patches)

    # show image
    imshow(whitened_patches_montage)

    # get giant whitened patch image array
    giant_whitened_patch_image_array = get_giant_patch_image(whitened_patches, dtype='uint8', scale=True)
    print giant_whitened_patch_image_array.shape
    print giant_whitened_patch_image_array

    imshow(giant_whitened_patch_image_array)

    # save giant whitened image
    # destination_filename = os.path.join(this_file_path, "./data/04_whitened_patches_giant_image.jpg")
    # save_image(giant_whitened_patch_image_array, destination_filename=destination_filename, show=True)

    return whitened_patches
    

if __name__ == "__main__":
    image_filename = os.path.realpath(os.path.join(this_file_path, "./data/yaleB01_P00A-005E-10_64x64.pgm"))
    
    # test patch extraction
    # test_patch_extraction(image_filename)

    # test contrast normalization
    # test_contrast_normalization(image_filename)

    # test whitening
    test_whitening(image_filename)
    