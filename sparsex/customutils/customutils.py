from sklearn.feature_extraction.image import extract_patches_2d
from PIL import Image
import numpy as np

# source : http://stackoverflow.com/questions/13990465/3d-numpy-array-to-2d
def get_giant_patch_image(patches, dtype='uint8', scale=False):
    patches = patches.copy()
    if scale:
        original_patches_shape = patches.shape
        patches = patches.reshape((patches.shape[0], -1))
        patches = patches + np.abs(patches.max(axis=1)[:, np.newaxis]) + 0.01
        patches = patches * (255.0 / np.abs((patches.max(axis=1) - patches.min(axis=1))[:, np.newaxis] + 0.0001))
        patches = patches.reshape(original_patches_shape)
    sqrt_of_rows = int(np.sqrt(patches.shape[0]))
    broken_patches = patches.reshape(sqrt_of_rows, sqrt_of_rows, patches.shape[1], patches.shape[2])
    giant_patch_image = broken_patches.swapaxes(1,2).reshape(sqrt_of_rows * patches.shape[1],-1)
    return giant_patch_image.astype(dtype)


def save_image(image_array, destination_filename="~/home/nitish/Desktop/hello.jpg", dtype='uint8'):
    image_pil = Image.fromarray(image_array.astype(dtype))
    image_pil.save(destination_filename)


def get_image_from_file(image_filename, dtype='float'):
    image_pil = Image.open(image_filename)
    image = np.array(image_pil).astype(dtype)
    return image