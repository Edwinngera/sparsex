from ..customutils.customutils import save_image, get_image_from_file, get_giant_patch_image
from ..preprocessing.preprocessing import Preprocessing
from PIL import Image
import numpy as np

def test_patch_extraction(image_filename):
    # get instance of Preprocessing
    preprocessing = Preprocessing()

    # get original image array
    image_array = get_image_from_file(image_filename)
    print image_array.shape
    print image_array.dtype

    # save original image
    destination_filename = "/home/nitish/Desktop/01_original_image.jpg"
    save_image(image_array.astype('uint8'), destination_filename=destination_filename)
    
    # get patches
    print "Extracting patches"
    patches = preprocessing.extract_patches(image_array)
    print patches.shape

    # get giant patch image
    giant_patch_image_array = get_giant_patch_image(patches, dtype='uint8', scale=False)
    print giant_patch_image_array.shape

    # save giant image
    destination_filename = "/home/nitish/Desktop/02_patch_extraction_giant_image.jpg"
    save_image(giant_patch_image_array, destination_filename=destination_filename)

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

    # get giant normalized patch image array
    giant_normalized_patch_image_array = get_giant_patch_image(normalized_patches, dtype='uint8', scale=True)
    print giant_normalized_patch_image_array.shape
    print giant_normalized_patch_image_array[:8,:8]

    # save giant normalized image
    destination_filename = "/home/nitish/Desktop/03_normalized_patches_giant_image.jpg"
    save_image(giant_normalized_patch_image_array, destination_filename=destination_filename)

    return normalized_patches


def test_whitening(image_filename):
    # get instance of Preprocessing
    preprocessing = Preprocessing()

    # get normalized patches
    normalized_patches = test_contrast_normalization(image_filename)

    # get whitened patches
    whitened_patches = preprocessing.get_whitened_patches(normalized_patches)
    print whitened_patches.shape

    # get giant whitened patch image array
    giant_whitened_patch_image_array = get_giant_patch_image(whitened_patches, dtype='uint8', scale=True)
    print giant_whitened_patch_image_array.shape
    

if __name__ == "__main__":
    image_filename = "/home/nitish/mas_course_ss2015/assignments/sparsex/datasets/yale_face_b_ext_cropped/CroppedYale_64x64/yaleB01/yaleB01_P00A+000E+00.pgm"
    
    # test patch extraction
    #test_patch_extraction(image_filename)

    # test contrast normalization
    test_contrast_normalization(image_filename)

    # test whitening
    #test_whitening(image_filename)